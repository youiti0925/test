"""Decision Engine-based backtest.

Replays bars one at a time and feeds each one through the SAME rule chain
that production trade/analyze uses: Risk Gate → patterns → higher-TF
alignment → risk-reward floor → (advisory LLM, opt-in). A bar produces
a position only when `decide()` returns BUY/SELL — exactly mirroring
live behaviour.

Why a separate file from `backtest.py`?
---------------------------------------
The legacy `backtest.py` runs the technical-only signal and is useful as
a quick sanity baseline. This module stacks on the full safety chain so
you can directly compare:

  legacy "what would the indicators alone have done?"
    vs.
  engine "what would the production engine have done?"

Two backtests on the same bars; the gap between them tells you what the
gate is actually saving you from. Per spec §4 / §18, the engine version
is the one to trust before letting any rule change reach live.

Point-in-time correctness
-------------------------
Every step uses `df.iloc[: i + 1]` exclusively. We never peek past `i`.
The engine itself relies on `patterns.detect_swings`, which already pins
"no future leak" via test_patterns_no_future_leak. Higher-timeframe
trend uses the same window resampled — it will not consult bars to the
right of `i`.
"""
from __future__ import annotations

import dataclasses
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .calendar import Event
from .decision_engine import Decision, decide as decide_action
from .royal_road_decision import (
    PROFILE_NAME as ROYAL_ROAD_PROFILE,
    SUPPORTED_DECISION_PROFILES,
    compare_decisions as compare_royal_decisions,
    decide_royal_road,
    validate_decision_profile,
)
from .royal_road_decision_modes import (
    DEFAULT_ROYAL_ROAD_MODE,
    get_royal_road_mode_config,
)
from .royal_road_decision_v2 import (
    PROFILE_NAME_V2 as ROYAL_ROAD_PROFILE_V2,
    compare_v2_vs_current,
    compare_v2_vs_v1,
    decide_royal_road_v2,
)
from .royal_road_integrated_decision import (
    PROFILE_NAME_V2_INTEGRATED as ROYAL_ROAD_PROFILE_INTEGRATED,
    decide_royal_road_v2_integrated,
)
from .stop_modes import (
    DEFAULT_STOP_MODE,
    plan_stop,
    validate_stop_mode,
)
from .technical_confluence import build_technical_confluence
from .macro import MacroSnapshot
from .waveform_library import WaveformSample
from .decision_trace import (
    BarDecisionTrace,
    RunMetadata,
    TRACE_SCHEMA_VERSION,
    compute_data_snapshot_hash,
    compute_strategy_config_hash,
    get_commit_sha,
)
from .decision_trace_build import (
    build_atr_unavailable_trace as _build_atr_unavailable_trace,
    build_full_trace as _build_full_trace,
    build_run_id as _build_run_id,
    long_term_trend_slice as _long_term_trend_slice,
    populate_future_outcomes as _populate_future_outcomes,
)
from .higher_timeframe import HIGHER_INTERVAL_MAP
from .indicators import build_snapshot, technical_signal, technical_signal_reasons
from .patterns import TrendState, analyse as analyse_patterns
from .risk import atr as compute_atr
from .risk_gate import RiskState


@dataclass
class EnginePosition:
    side: str       # BUY | SELL
    entry: float
    entry_ts: pd.Timestamp
    size: float
    stop: float
    take_profit: float
    bars_held: int = 0
    # Stable id of the form f"{run_id}_T{N}" so the trade can be joined
    # back to the BarDecisionTrace entries at entry and exit.
    trade_id: str = ""


def _position_dict(p: "EnginePosition | None") -> dict | None:
    if p is None:
        return None
    return {
        "side": p.side,
        "entry": p.entry,
        "entry_ts": p.entry_ts,
        "size": p.size,
        "stop": p.stop,
        "take_profit": p.take_profit,
        "bars_held": p.bars_held,
        "trade_id": p.trade_id,
    }


@dataclass
class EngineTrade:
    side: str
    entry: float
    exit: float
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    pnl: float
    return_pct: float
    bars_held: int
    # Possible values: "stop" | "take_profit" | "max_holding" | "end_of_data".
    # NOTE: "flip" (close-on-opposite-signal) is INTENTIONALLY NOT
    # implemented — the engine relies on stop / TP / max_holding to
    # frame each trade. The legacy `backtest.py` is the flip-on-signal
    # path; pick that if you want flip semantics for comparison.
    exit_reason: str
    rule_chain: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    # Joins this trade to the corresponding ExecutionTraceSlice entries
    # (entry bar + exit bar). Empty default keeps existing tests/code that
    # construct EngineTrade directly working unchanged.
    trade_id: str = ""


@dataclass
class EngineBacktestResult:
    trades: list[EngineTrade] = field(default_factory=list)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    # Counts of why a bar produced no entry — invaluable for understanding
    # what the gate is filtering. Keys are decision.blocked_by codes plus
    # synthetic "hold_no_signal" / "hold_pattern" / "hold_other" buckets.
    hold_reasons: dict[str, int] = field(default_factory=dict)
    bars_processed: int = 0
    # Per-bar audit trail. Shape: list of BarDecisionTrace, one entry per
    # bar processed (warmup excluded — they don't go through decide()).
    # Empty when run_engine_backtest is called with capture_traces=False.
    decision_traces: list[BarDecisionTrace] = field(default_factory=list)
    # Reproducibility metadata — strategy hash, data hash, commit sha,
    # run_id linking every trace back to this run.
    run_metadata: RunMetadata | None = None

    def to_decision_trace_records(self) -> list[dict]:
        """Return all traces as JSON-serialisable dicts."""
        return [t.to_dict() for t in self.decision_traces]

    def to_run_metadata_dict(self) -> dict:
        """Return run metadata as a JSON-serialisable dict (or empty)."""
        return self.run_metadata.to_dict() if self.run_metadata else {}

    def metrics(self) -> dict:
        bars = self.bars_processed or 1
        execution_metadata = _synthetic_execution_metadata()
        if not self.trades:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "hold_rate": 1.0,
                "hold_reasons": dict(self.hold_reasons),
                "bars_processed": self.bars_processed,
                **execution_metadata,
            }
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses)) or 1e-9
        equity = np.array([eq for _, eq in self.equity_curve])
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        total_return = (equity[-1] / equity[0] - 1.0) * 100 if len(equity) else 0.0

        # Largest losing streak — survival metric, complements win_rate
        max_streak = streak = 0
        for t in self.trades:
            if t.pnl < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        return {
            "n_trades": len(self.trades),
            "win_rate": len(wins) / len(self.trades),
            "profit_factor": gross_win / gross_loss,
            "total_return_pct": float(total_return),
            "max_drawdown_pct": float(dd.min() * 100),
            "avg_win_pct": (
                float(np.mean([t.return_pct for t in wins])) if wins else 0.0
            ),
            "avg_loss_pct": (
                float(np.mean([t.return_pct for t in losses])) if losses else 0.0
            ),
            "max_consecutive_losses": max_streak,
            "avg_holding_bars": float(np.mean([t.bars_held for t in self.trades])),
            "hold_rate": 1.0 - (len(self.trades) / bars),
            "hold_reasons": dict(self.hold_reasons),
            "exit_reasons": _count_by_attr(self.trades, "exit_reason"),
            "bars_processed": self.bars_processed,
            **execution_metadata,
        }


def _synthetic_execution_metadata() -> dict[str, str | bool]:
    """Disclose execution assumptions used by this research backtest.

    The engine replay validates the decision rule chain, not real broker
    execution quality. These fields make reports self-describing so the
    result is not mistaken for spread/slippage/fill-aware live performance.
    """
    return {
        "synthetic_execution": True,
        "spread_mode": "not_modelled",
        "slippage_mode": "not_modelled",
        "fill_model": "close_price",
        "bid_ask_mode": "not_modelled",
        "sentiment_archive": "not_available",
    }


def _count_by_attr(trades: list[EngineTrade], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in trades:
        key = getattr(t, attr)
        counts[key] = counts.get(key, 0) + 1
    return counts


# A user-supplied function that, given the window up to bar i, returns
# either a TradeSignal-like advisory or None. Backtest doesn't call the
# real LLM — too expensive and non-deterministic — so this is normally
# left as `None` (advisor disabled). It exists for parity tests.
LLMSignalFn = Callable[[pd.DataFrame], object]


def _resample_higher_tf(df_window: pd.DataFrame, base_interval: str) -> str:
    """Compute higher-timeframe trend without re-fetching from yfinance.

    Resamples the in-memory window and re-runs `analyse` — keeps the
    backtest fully offline AND point-in-time. Returns "UNKNOWN" if the
    base interval has no entry in the map (e.g. unsupported TF).

    Map mirrors the live-side `HIGHER_INTERVAL_MAP` so 30m/2h/4h
    backtests don't silently degrade to UNKNOWN — pandas freq aliases
    are spelled here (e.g. "4H" not "4h" since pandas needs uppercase
    for the hour alias when not paired with "min"; "1W" for weekly).
    """
    rule = {
        "1m": "15min",
        "5m": "1h",
        "15m": "4h",
        "30m": "4h",
        "1h": "1D",
        "2h": "1D",
        "4h": "1W",
        "1d": "1W",
    }.get(base_interval)
    if rule is None or len(df_window) < 30:
        return "UNKNOWN"
    try:
        higher = df_window.resample(rule).agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"
    if len(higher) < 10:
        return "UNKNOWN"
    pat = analyse_patterns(higher)
    return pat.trend_state.value


def _bar_event_window(events: tuple[Event, ...], ts: pd.Timestamp) -> tuple[Event, ...]:
    """Events whose `when` is within ±48h of `ts` — fed to RiskGate."""
    if not events:
        return ()
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    target = ts.to_pydatetime()
    return tuple(
        e for e in events
        if abs((e.when - target).total_seconds()) <= 48 * 3600
    )


def _compute_waveform_bias_for_bar(
    df: pd.DataFrame,
    i: int,
    *,
    library: list[WaveformSample] | None,
    library_id: str | None,
    interval: str,
    window_bars: int,
    horizon_bars: int,
    method: str,
    min_score: float,
    min_samples: int,
    min_directional_share: float,
) -> dict | None:
    """Thin wrapper: returns None when no library is attached (so the
    trace stores a literal null and downstream tooling treats the bar
    as 'waveform lookup not requested'). Otherwise delegates to
    `decision_trace_build.waveform_bias_dict`, which itself returns a
    dict — possibly with only `unavailable_reason` if the lookup could
    not run for this bar (look-ahead filter empty, unknown interval,
    insufficient warmup, etc.).
    """
    if library is None:
        return None
    from .decision_trace_build import waveform_bias_dict
    return waveform_bias_dict(
        library, df, i,
        interval=interval,
        library_id=library_id,
        window_bars=window_bars,
        horizon_bars=horizon_bars,
        method=method,
        min_score=min_score,
        min_samples=min_samples,
        min_directional_share=min_directional_share,
    )


def _validate_and_concat_context(
    df_context: pd.DataFrame, df_test: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Concatenate `df_context + df_test` after strict validation. PR #17.

    Returns
    -------
    df_full : pd.DataFrame
        The concatenation, sorted by index, used **only** as input to
        `long_term_trend_slice`. The judgement path keeps using df_test.
    n_context_bars : int
        `len(df_context)` after validation. Surfaced in run_metadata so
        the audit reader can recover the boundary.

    Raises
    ------
    ValueError
        If df_context overlaps with df_test in time, or either has
        duplicate / non-monotonic index, or the tz state of the two
        indices is mismatched. We fail loud — silently dropping rows
        would corrupt long-term trend computation in ways that look
        plausible but are wrong.
    """
    if df_context.empty:
        return df_test, 0

    # tz-state must match — comparing tz-aware against tz-naive raises
    # in pandas, but the message is cryptic. Catch it here.
    ctx_tz = df_context.index.tz
    test_tz = df_test.index.tz
    if (ctx_tz is None) != (test_tz is None):
        raise ValueError(
            "df_context and df_test have mismatched timezone state "
            f"(context tz={ctx_tz}, test tz={test_tz}). Both must be "
            "tz-aware or both tz-naive — cannot compare."
        )

    # Sort defensively, but catch unexpected duplicates *before* the sort
    # masks them.
    if df_context.index.duplicated().any():
        n_dup = int(df_context.index.duplicated().sum())
        raise ValueError(
            f"df_context has {n_dup} duplicate index entries; refuse to "
            "silently dedupe (would corrupt SMA computation). Caller must "
            "clean upstream."
        )
    if df_test.index.duplicated().any():
        n_dup = int(df_test.index.duplicated().sum())
        raise ValueError(
            f"df_test has {n_dup} duplicate index entries; refuse to "
            "silently dedupe."
        )

    df_context_sorted = df_context.sort_index()
    df_test_sorted = df_test.sort_index()
    ctx_max = df_context_sorted.index.max()
    test_min = df_test_sorted.index.min()
    if ctx_max >= test_min:
        raise ValueError(
            "df_context overlaps df_test: "
            f"context.max={ctx_max!s} >= test.min={test_min!s}. "
            "context must be strictly before test_start."
        )

    df_full = pd.concat([df_context_sorted, df_test_sorted])
    # Defensive: post-concat duplicates would only happen if the two
    # frames share a timestamp on the seam, which the strict-less-than
    # check above already rejects. Belt and braces.
    if df_full.index.duplicated().any():  # pragma: no cover
        raise ValueError(
            "concatenated df_full has duplicate index after merge — "
            "context/test boundary likely overlapping."
        )
    return df_full, len(df_context_sorted)


def _build_context_metadata(
    *,
    df_test: pd.DataFrame,
    df_context: pd.DataFrame | None,
    n_context_bars: int,
    context_days: int | None,
) -> dict:
    """Readable context-period block for run_metadata. PR #17.

    Always emits all 9 fields regardless of whether context is attached —
    a missing field would be ambiguous between "old run" and "explicitly
    no context", and the reader has to defend against `KeyError` either
    way. ``trace_scope`` / ``metrics_scope`` are constants here because
    the engine ALWAYS confines trades and traces to the test frame, even
    when context is supplied (the judgement loop only iterates df_test).
    """
    enabled = n_context_bars > 0 and df_context is not None
    test_start_iso = (
        df_test.index.min().isoformat() if len(df_test) > 0 else None
    )
    test_end_iso = (
        df_test.index.max().isoformat() if len(df_test) > 0 else None
    )
    if enabled:
        # df_context has been validated and possibly sorted — recover its
        # earliest bar, which is the start of the historical window.
        context_start_iso = df_context.sort_index().index.min().isoformat()
    else:
        context_start_iso = None
    return {
        "context_enabled": enabled,
        "context_days": context_days if context_days is not None else 0,
        "context_start": context_start_iso,
        "test_start": test_start_iso,
        "test_end": test_end_iso,
        "n_context_bars": n_context_bars,
        "n_test_bars": len(df_test),
        "metrics_scope": "test_only",
        "trace_scope": "test_only",
        "long_term_context_attached": enabled,
    }


def _build_waveform_metadata(
    *,
    library: list[WaveformSample] | None,
    library_id: str | None,
    window_bars: int,
    horizon_bars: int,
    min_samples: int,
    min_score: float,
    min_share: float,
    method: str,
) -> dict:
    """Readable waveform-match config for run_metadata.json (PR #16).

    The same payload is fed into `compute_strategy_config_hash` so the hash
    still distinguishes runs with different libraries, but the readable
    copy here lets a future audit answer "which library / horizon / method
    produced this run?" without cracking open decision_traces.jsonl.
    Always emitted (waveform_enabled may be False) — a missing field would
    be ambiguous between "unset" and "explicitly disabled".
    """
    schema: str | None = None
    basename: str | None = None
    first_end_ts: str | None = None
    last_end_ts: str | None = None

    if library is not None and library_id is not None:
        # library_id format (PR-#15):
        #   "{basename}|schema={v1|v2|empty}|n=N|{first_end_ts}|{last_end_ts}|{sha8}"
        # Pre-PR-#15 callers may pass a non-conforming string — keep parsing
        # forgiving so we never crash run_metadata generation.
        parts = library_id.split("|")
        if len(parts) >= 1:
            basename = parts[0]
        for p in parts[1:]:
            if p.startswith("schema="):
                schema = p[len("schema="):]
                break
        if len(parts) >= 5:
            first_end_ts = parts[3] or None
            last_end_ts = parts[4] or None

    return {
        "waveform_enabled": library is not None,
        "waveform_library_id": library_id,
        "waveform_library_size": len(library) if library is not None else 0,
        "waveform_library_schema": schema,
        "waveform_library_path_basename": basename,
        "waveform_library_first_end_ts": first_end_ts,
        "waveform_library_last_end_ts": last_end_ts,
        "waveform_window_bars": window_bars,
        "waveform_horizon_bars": horizon_bars,
        "waveform_min_samples": min_samples,
        "waveform_min_score": min_score,
        "waveform_min_share": min_share,
        "waveform_method": method,
    }


def run_engine_backtest(
    df: pd.DataFrame,
    symbol: str,
    *,
    interval: str = "1h",
    initial_cash: float = 10_000.0,
    warmup: int = 50,
    stop_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
    max_holding_bars: int = 48,
    events: Iterable[Event] = (),
    llm_signal_fn: LLMSignalFn | None = None,
    use_higher_tf: bool = True,
    capture_traces: bool = True,
    compute_future_outcome: bool = True,
    data_source: str = "unknown",
    data_retrieved_at: datetime | None = None,
    macro: MacroSnapshot | None = None,
    # PR #15: waveform-match observability — values land in
    # trace.waveform.waveform_bias and run_metadata, but are NEVER
    # forwarded to decide_action (decisions remain unchanged whether
    # a library is attached or not).
    waveform_library: list[WaveformSample] | None = None,
    waveform_library_id: str | None = None,
    waveform_window_bars: int = 60,
    waveform_horizon_bars: int = 24,
    waveform_min_samples: int = 20,
    waveform_min_score: float = 0.55,
    waveform_min_share: float = 0.6,
    waveform_method: str = "dtw",
    # PR #17: optional historical OHLCV strictly BEFORE `df.index.min()`,
    # used **only** by long_term_trend_slice so SMA90/200 + monthly_trend
    # become valid from test_start. The judgement path (ATR / patterns /
    # _resample_higher_tf / waveform target signature / RSI / decide_action)
    # NEVER sees df_context — invariant pinned by
    # test_decisions_byte_identical_with_or_without_context.
    df_context: pd.DataFrame | None = None,
    context_days: int | None = None,
    # PR #18: optional pre-built calendar metadata block (events.json
    # freshness / coverage / window-policy version). Engine doesn't
    # touch the events themselves through this kwarg — events flow as
    # before via the `events=` tuple, keeping decision behaviour
    # invariant. This kwarg is purely for run_metadata.calendar.
    calendar: dict | None = None,
    # PR #19: optional pre-built parameter catalog block (baseline_id,
    # baseline_version, baseline_payload_hash, full baseline payload,
    # requested profile). METADATA-ONLY — engine reads no values from
    # here. The kwarg exists so the catalog reaches RunMetadata via
    # the same plumbing as `calendar`, without touching decide_action.
    parameters: dict | None = None,
    # PR #21: indicator-period / threshold overrides for runtime A/B.
    # All None = current_runtime defaults (byte-identical to PR #20
    # main). Non-None values flow to `indicators.build_snapshot` /
    # `technical_signal*` / `compute_atr`. Triggered ONLY by
    # backtest-engine `--apply-parameter-profile`. live cmd_trade
    # never calls `run_engine_backtest`, so these kwargs cannot
    # affect live behaviour.
    rsi_period: int | None = None,
    rsi_overbought: float | None = None,
    rsi_oversold: float | None = None,
    macd_fast: int | None = None,
    macd_slow: int | None = None,
    macd_signal_period: int | None = None,
    bb_period: int | None = None,
    bb_std: float | None = None,
    atr_period: int | None = None,
    # royal_road_decision_v1: opt-in decision profile. When equal to
    # the default "current_runtime", the decide_action chain is the
    # sole BUY/SELL/HOLD source (engine output byte-identical to
    # PR #21 main). When set to "royal_road_decision_v1", the
    # technical_confluence dict drives an alternative decision and
    # the trace gains a `royal_road_decision` slice with comparison
    # metadata. Live / OANDA / paper paths do not call this function
    # so this kwarg cannot affect live trading.
    decision_profile: str = "current_runtime",
    # royal_road_decision_v1 mode selector. Only consulted when
    # decision_profile == "royal_road_decision_v1". Defaults to
    # `balanced` (research candidate); `strict` reproduces the original
    # diagnostic heuristic; `exploratory` is for discovery and not for
    # adoption. See `royal_road_decision_modes.py`.
    royal_road_mode: str = DEFAULT_ROYAL_ROAD_MODE,
    # Integrated profile mode (consulted only when decision_profile=
    # "royal_road_decision_v2_integrated"). Two modes:
    #   integrated_balanced  required-data missing → WARN
    #   integrated_strict    required-data missing → HOLD
    # Live / OANDA / paper do not call this function, so this cannot
    # affect live trading.
    integrated_mode: str = "integrated_balanced",
    # v2 inputs (consulted only when decision_profile=
    # "royal_road_decision_v2"). df_lower_tf must be a DataFrame whose
    # bars are at a finer interval than `interval`. lower_tf_interval
    # is recorded in trace; the engine itself does not require it for
    # correctness. df_lower_tf=None → trigger emits unavailable_reason.
    df_lower_tf: pd.DataFrame | None = None,
    lower_tf_interval: str | None = None,
    # Stop placement mode. "atr" (default) reproduces PR #21 main.
    # "structure" / "hybrid" are opt-in and consult the v2 structure
    # stop layer. Live cmd_trade does not call this function so the
    # flag cannot affect live trading.
    stop_mode: str = DEFAULT_STOP_MODE,
) -> EngineBacktestResult:
    """Run a Decision Engine-driven backtest over `df`.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a tz-aware DatetimeIndex.
    interval:
        Base timeframe ("1h", "15m", ...). Used to compute the
        higher-timeframe trend in-memory.
    warmup:
        Number of leading bars skipped — the engine needs swings to form,
        and the gate's data_quality check requires ≥50 bars.
    stop_atr_mult / tp_atr_mult:
        Stop and take-profit are placed at `mult * ATR` from entry. Risk-
        reward fed to the engine is `tp_atr_mult / stop_atr_mult`, mirroring
        `risk.plan_trade`.
    max_holding_bars:
        Force-close after N bars even if neither stop nor TP fired.
        Without this, the win rate gets inflated by trades that never close.
    events:
        Tuple of macro events. The engine sees only events within ±48h of
        the bar — same window the live gate uses. (Calendar freshness is
        N/A in offline backtests.)
    llm_signal_fn:
        Optional callable producing an advisory TradeSignal per bar.
        Default `None` keeps the test deterministic.
    use_higher_tf:
        Compute higher-TF trend per bar (slightly slower). Default True.
    capture_traces:
        When True (default), every processed bar produces a BarDecisionTrace
        in result.decision_traces. Decision logic is identical with or
        without this flag — pinned by test_decisions_unchanged_with_trace_logging.
    compute_future_outcome:
        When True (default), a second pass after the bar loop fills the
        FutureOutcomeSlice on every trace. The decision pipeline never
        consults future_outcome — pinned by
        test_future_outcome_does_not_affect_decisions.
    data_source / data_retrieved_at:
        Provenance metadata copied into RunMetadata and MarketSlice. Pure
        labelling — no behaviour depends on these.
    """
    result = EngineBacktestResult()
    pos: EnginePosition | None = None
    cash = initial_cash
    equity = initial_cash
    pending_chain: tuple[str, ...] = ()
    pending_blocked: tuple[str, ...] = ()

    # PR #17: optional historical context for long_term_trend_slice ONLY.
    # The judgement path (ATR / patterns / higher_tf / RSI / waveform) sees
    # df only — never df_full. This is what keeps decisions byte-identical
    # with or without context (pinned by
    # test_decisions_byte_identical_with_or_without_context).
    df_full = df
    n_context_bars = 0
    if df_context is not None and len(df_context) > 0:
        df_full, n_context_bars = _validate_and_concat_context(df_context, df)

    def _long_term_trend_for_bar(i_test: int):
        """Closure: compute long_term_trend at test bar i_test, expanding
        the input window to include df_context if provided. PR #17.

        Returns None when no context is attached — preserves the existing
        behaviour where the trace builder falls back to
        long_term_trend_slice(df, i) on its own. We only emit an override
        when context actually changes the answer, keeping the no-context
        path byte-identical with pre-PR-#17 runs.
        """
        if n_context_bars == 0:
            return None
        # Map the test-frame bar index into the full-frame index. The
        # full frame is df_context (length n_context_bars) followed by df,
        # so test bar i_test corresponds to full bar i_test + n_context_bars.
        return _long_term_trend_slice(df_full, i_test + n_context_bars)

    # royal_road_decision_v1: validate the requested decision profile.
    # Unknown profile names raise — the CLI is expected to catch and
    # surface this with exit code 2. Default `current_runtime` keeps
    # the legacy path; other names route through the corresponding
    # profile module after the existing decide_action call.
    _decision_profile = validate_decision_profile(decision_profile)
    _royal_road_active = (_decision_profile == ROYAL_ROAD_PROFILE)
    _royal_road_v2_active = (_decision_profile == ROYAL_ROAD_PROFILE_V2)
    _royal_road_integrated_active = (
        _decision_profile == ROYAL_ROAD_PROFILE_INTEGRATED
    )
    # Validate the mode upfront (raises ValueError on unknown values).
    # The kwarg is harmless when the profile is current_runtime, but we
    # still want unknown values to surface immediately rather than
    # waiting for the first royal-road bar.
    _royal_road_mode = get_royal_road_mode_config(royal_road_mode).name
    _stop_mode = validate_stop_mode(stop_mode)
    # Integrated mode (only consulted when integrated profile active).
    # Validate upfront so a typo doesn't silently fall through.
    if _royal_road_integrated_active:
        from .royal_road_integrated_decision import validate_integrated_mode
        _integrated_mode = validate_integrated_mode(integrated_mode)
    else:
        _integrated_mode = integrated_mode

    # PR #21: resolve indicator-period overrides. None → existing
    # current_runtime defaults, non-None → flowed through.
    _atr_period_eff = atr_period if atr_period is not None else 14
    _rsi_period_eff = rsi_period if rsi_period is not None else 14
    _rsi_overbought_eff = rsi_overbought if rsi_overbought is not None else 70.0
    _rsi_oversold_eff = rsi_oversold if rsi_oversold is not None else 30.0
    _macd_fast_eff = macd_fast if macd_fast is not None else 12
    _macd_slow_eff = macd_slow if macd_slow is not None else 26
    _macd_signal_eff = macd_signal_period if macd_signal_period is not None else 9
    _bb_period_eff = bb_period if bb_period is not None else 20
    _bb_std_eff = bb_std if bb_std is not None else 2.0

    # PR #21: actually-used dict for trace audit. Always populated so
    # TechnicalSlice.*_used fields capture the run config (the values
    # equal the historical defaults when no profile is applied — this
    # is fine because the audit is meant to RECORD what ran, not just
    # what differed).
    _runtime_overrides_for_trace: dict = {
        "rsi_period": _rsi_period_eff,
        "rsi_overbought": _rsi_overbought_eff,
        "rsi_oversold": _rsi_oversold_eff,
        "macd_fast": _macd_fast_eff,
        "macd_slow": _macd_slow_eff,
        "macd_signal": _macd_signal_eff,
        "bb_period": _bb_period_eff,
        "bb_std": _bb_std_eff,
        "atr_period": _atr_period_eff,
        "stop_atr_mult": stop_atr_mult,
        "tp_atr_mult": tp_atr_mult,
        "max_holding_bars": max_holding_bars,
    }

    # ATR is monotonic-suffix — compute once over the full df. Each bar
    # only reads atr_series.iloc[i], no future leak.
    atr_series = compute_atr(df, period=_atr_period_eff)
    events_tuple = tuple(events)
    risk_reward = tp_atr_mult / stop_atr_mult

    # ── Trace metadata ──────────────────────────────────────────────────
    # Built up-front so every trace can carry the same run_id. bar_range
    # is finalised after the loop so it reflects what actually ran.
    run_metadata: RunMetadata | None = None
    trade_counter = 0
    if capture_traces:
        run_id = _build_run_id(symbol)
        # PR #16: derive readable waveform metadata once, then reuse for both
        # the strategy_config_hash payload and RunMetadata.waveform. The hash
        # alone cannot tell a future audit which library / horizon / method
        # produced these traces; without the readable copy you have to crack
        # open decision_traces.jsonl just to find the library_id.
        waveform_meta = _build_waveform_metadata(
            library=waveform_library,
            library_id=waveform_library_id,
            window_bars=waveform_window_bars,
            horizon_bars=waveform_horizon_bars,
            min_samples=waveform_min_samples,
            min_score=waveform_min_score,
            min_share=waveform_min_share,
            method=waveform_method,
        )
        # PR #17: backtest-engine context-period block. Always emitted for
        # backtest runs so a future audit can see "context_enabled=false"
        # explicitly rather than guessing from a missing field.
        context_meta = _build_context_metadata(
            df_test=df,
            df_context=df_context,
            n_context_bars=n_context_bars,
            context_days=context_days,
        )
        config_payload = {
            "interval": interval,
            "initial_cash": initial_cash,
            "warmup": warmup,
            "stop_atr_mult": stop_atr_mult,
            "tp_atr_mult": tp_atr_mult,
            "max_holding_bars": max_holding_bars,
            "use_higher_tf": use_higher_tf,
            "n_events": len(events_tuple),
            "llm_signal_fn_used": llm_signal_fn is not None,
            "macro_attached": macro is not None,
            "macro_slots": (
                sorted(macro.series.keys()) if macro is not None else []
            ),
            # PR #15 — waveform-match config. Saved here so a trace can be
            # re-derived from the same library + params later.
            **waveform_meta,
            # PR #17 — context-period config. Folded into the hash so a run
            # made WITH context is distinguishable from one WITHOUT, even
            # though decisions byte-match by design.
            "context_enabled": context_meta["context_enabled"],
            "context_days": context_meta["context_days"],
            "n_context_bars": context_meta["n_context_bars"],
        }
        sha, sha_status = get_commit_sha()
        run_metadata = RunMetadata(
            run_id=run_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            timeframe=interval,
            bar_range={"start": None, "end": None, "n_bars": 0, "warmup": warmup},
            trace_schema_version=TRACE_SCHEMA_VERSION,
            strategy_config_hash=compute_strategy_config_hash(config_payload),
            data_snapshot_hash=compute_data_snapshot_hash(df, symbol, interval),
            input_data_source=data_source,
            input_data_retrieved_at=(
                data_retrieved_at.isoformat() if data_retrieved_at else None
            ),
            synthetic_execution=True,
            commit_sha=sha,
            commit_sha_status=sha_status,
            execution_mode="synthetic_backtest",
            engine_version="backtest_engine.run_engine_backtest",
            timezone_name="UTC",
            waveform=dict(waveform_meta),
            context=dict(context_meta),
            calendar=dict(calendar) if calendar is not None else None,
            parameters=dict(parameters) if parameters is not None else None,
        )
        result.run_metadata = run_metadata
    else:
        run_id = ""

    for i in range(warmup, len(df)):
        result.bars_processed += 1
        window = df.iloc[: i + 1]
        ts = window.index[-1]
        price = float(window["close"].iloc[-1])

        # Snapshot before any state mutation on this bar, so the trace can
        # reproduce what the engine "saw" entering the bar.
        position_before = _position_dict(pos)
        bars_held_before = pos.bars_held if pos is not None else None
        had_open_position = pos is not None

        # Mark-to-market the existing position first; stops/TPs win over
        # any new signal on the same bar.
        bar_exit_event = False
        bar_exit_reason: str | None = None
        bar_exit_price: float | None = None
        bar_exit_trade_id: str | None = None
        if pos is not None:
            pos.bars_held += 1
            high = float(window["high"].iloc[-1])
            low = float(window["low"].iloc[-1])
            exit_reason: str | None = None
            exit_price: float | None = None
            if pos.side == "BUY":
                if low <= pos.stop:
                    exit_reason, exit_price = "stop", pos.stop
                elif high >= pos.take_profit:
                    exit_reason, exit_price = "take_profit", pos.take_profit
            else:  # SELL
                if high >= pos.stop:
                    exit_reason, exit_price = "stop", pos.stop
                elif low <= pos.take_profit:
                    exit_reason, exit_price = "take_profit", pos.take_profit

            if exit_reason is None and pos.bars_held >= max_holding_bars:
                exit_reason, exit_price = "max_holding", price

            if exit_reason is not None and exit_price is not None:
                direction = 1 if pos.side == "BUY" else -1
                pnl = (exit_price - pos.entry) * pos.size * direction
                ret_pct = 100 * (exit_price - pos.entry) / pos.entry * direction
                result.trades.append(
                    EngineTrade(
                        side=pos.side,
                        entry=pos.entry,
                        exit=exit_price,
                        entry_ts=pos.entry_ts,
                        exit_ts=ts,
                        pnl=pnl,
                        return_pct=ret_pct,
                        bars_held=pos.bars_held,
                        exit_reason=exit_reason,
                        rule_chain=pending_chain,
                        blocked_by=pending_blocked,
                        trade_id=pos.trade_id,
                    )
                )
                cash += pnl
                bar_exit_event = True
                bar_exit_reason = exit_reason
                bar_exit_price = exit_price
                bar_exit_trade_id = pos.trade_id
                pos = None

        # Build the engine inputs from the window only. Skip bars where
        # ATR is still NaN — engine will refuse the trade anyway.
        atr_value = atr_series.iloc[i]
        if pd.isna(atr_value) or atr_value <= 0:
            _bump(result.hold_reasons, "atr_unavailable")
            equity = cash + _unrealized(pos, price)
            result.equity_curve.append((ts, equity))
            if capture_traces:
                trace = _build_atr_unavailable_trace(
                    run_id=run_id,
                    df=df,
                    i=i,
                    ts=ts,
                    symbol=symbol,
                    interval=interval,
                    data_source=data_source,
                    events_tuple=events_tuple,
                    pos_after=pos,
                    position_before=position_before,
                    bars_held_before=bars_held_before,
                    had_open_position=had_open_position,
                    bar_exit_event=bar_exit_event,
                    bar_exit_reason=bar_exit_reason,
                    bar_exit_price=bar_exit_price,
                    bar_exit_trade_id=bar_exit_trade_id,
                    macro=macro,
                    waveform_bias=_compute_waveform_bias_for_bar(
                        df, i,
                        library=waveform_library,
                        library_id=waveform_library_id,
                        interval=interval,
                        window_bars=waveform_window_bars,
                        horizon_bars=waveform_horizon_bars,
                        method=waveform_method,
                        min_score=waveform_min_score,
                        min_samples=waveform_min_samples,
                        min_directional_share=waveform_min_share,
                    ),
                    long_term_trend_override=_long_term_trend_for_bar(i),
                    runtime_overrides=_runtime_overrides_for_trace,
                )
                result.decision_traces.append(trace)
            continue

        snap = build_snapshot(
            symbol, window,
            rsi_period=_rsi_period_eff,
            macd_fast=_macd_fast_eff,
            macd_slow=_macd_slow_eff,
            macd_signal=_macd_signal_eff,
            bb_period=_bb_period_eff,
            bb_std=_bb_std_eff,
        )
        tech = technical_signal(
            snap,
            rsi_overbought=_rsi_overbought_eff,
            rsi_oversold=_rsi_oversold_eff,
        )
        # Mirror call for trace only — action is delegated back to
        # technical_signal so it cannot drift. See indicators.py docstring.
        tech_action_for_trace, tech_reason_codes = technical_signal_reasons(
            snap,
            rsi_overbought=_rsi_overbought_eff,
            rsi_oversold=_rsi_oversold_eff,
        )
        pattern = analyse_patterns(window)
        higher_tf = (
            _resample_higher_tf(window, interval) if use_higher_tf else "UNKNOWN"
        )

        # Approximate sentiment is unavailable in backtest — leave None;
        # gate skips. Spread is None too (no Bid/Ask history). The point
        # of this backtest is the rule chain, not execution simulation.
        risk_state = RiskState(
            df=window,
            events=_bar_event_window(events_tuple, ts),
            spread_pct=None,
            sentiment_snapshot=None,
            now=ts.to_pydatetime() if ts.tzinfo else ts.tz_localize(
                timezone.utc
            ).to_pydatetime(),
        )

        llm_signal = llm_signal_fn(window) if llm_signal_fn is not None else None

        decision_current = decide_action(
            technical_signal=tech,
            pattern=pattern,
            higher_timeframe_trend=higher_tf,
            risk_reward=risk_reward,
            risk_state=risk_state,
            llm_signal=llm_signal,  # type: ignore[arg-type]
        )

        # royal_road_decision_v1: when the profile is active, we
        # additionally compute the royal-road decision from the
        # technical_confluence dict and use IT as the entry / exit
        # driver for this bar. The current_runtime decision is kept so
        # the trace can record what the legacy profile would have done
        # — this is what `compared_to_current_runtime` exposes.
        decision_royal: Decision | None = None
        royal_compare: dict | None = None
        decision_v2: Decision | None = None
        v2_compare_vs_current: dict | None = None
        v2_compare_vs_v1: dict | None = None
        v2_stop_plan = None
        decision_integrated: Decision | None = None
        if (
            _royal_road_active
            or _royal_road_v2_active
            or _royal_road_integrated_active
        ):
            confluence_dict = build_technical_confluence(
                df_window=df.iloc[: i + 1],
                snapshot=snap,
                pattern=pattern,
                atr_value=float(atr_value),
                technical_only_action=tech,
                stop_atr_mult=stop_atr_mult,
                tp_atr_mult=tp_atr_mult,
            )
            if _royal_road_active:
                decision_royal = decide_royal_road(
                    technical_signal=tech,
                    pattern=pattern,
                    higher_timeframe_trend=higher_tf,
                    risk_reward=risk_reward,
                    risk_state=risk_state,
                    technical_confluence=confluence_dict,
                    mode=_royal_road_mode,
                )
                royal_compare = compare_royal_decisions(
                    decision_current=decision_current,
                    decision_royal=decision_royal,
                )
                decision = decision_royal
            elif _royal_road_v2_active:
                # Always compute v1 for the v2 trace's compared_to_v1 block.
                decision_royal = decide_royal_road(
                    technical_signal=tech,
                    pattern=pattern,
                    higher_timeframe_trend=higher_tf,
                    risk_reward=risk_reward,
                    risk_state=risk_state,
                    technical_confluence=confluence_dict,
                    mode=_royal_road_mode,
                )
                royal_compare = compare_royal_decisions(
                    decision_current=decision_current,
                    decision_royal=decision_royal,
                )
                # Macro context dict for v2 (point-in-time).
                from .decision_trace_build import macro_context_slice
                _macro_slice = macro_context_slice(macro, ts)
                _macro_dict = (
                    _macro_slice.to_dict() if _macro_slice is not None else None
                )
                decision_v2 = decide_royal_road_v2(
                    df_window=df.iloc[: i + 1],
                    technical_confluence=confluence_dict,
                    pattern=pattern,
                    higher_timeframe_trend=higher_tf,
                    risk_reward=risk_reward,
                    risk_state=risk_state,
                    atr_value=float(atr_value),
                    last_close=float(price),
                    symbol=symbol,
                    macro_context=_macro_dict,
                    df_lower_tf=df_lower_tf,
                    lower_tf_interval=lower_tf_interval,
                    stop_mode=_stop_mode,
                    stop_atr_mult=stop_atr_mult,
                    tp_atr_mult=tp_atr_mult,
                    base_bar_close_ts=ts,
                    mode=_royal_road_mode,
                )
                v2_compare_vs_current = compare_v2_vs_current(
                    decision_current=decision_current,
                    decision_v2=decision_v2,
                )
                v2_compare_vs_v1 = compare_v2_vs_v1(
                    decision_v1=decision_royal,
                    decision_v2=decision_v2,
                )
                # Pull the stop plan back out of the advisory for engine
                # use (saves recomputation).
                v2_stop_plan = (decision_v2.advisory or {}).get(
                    "structure_stop_plan"
                )
                decision = decision_v2
            elif _royal_road_integrated_active:
                # Integrated profile: opt-in only. Same panel inputs as
                # v2 (so masterclass / source-pack panels match), but
                # the action is decided by integrated evidence axes.
                # NEVER reachable from live / OANDA / paper because
                # those paths don't call run_engine_backtest.
                from .decision_trace_build import macro_context_slice
                _macro_slice = macro_context_slice(macro, ts)
                _macro_dict = (
                    _macro_slice.to_dict() if _macro_slice is not None else None
                )
                decision_integrated = decide_royal_road_v2_integrated(
                    df_window=df.iloc[: i + 1],
                    technical_confluence=confluence_dict,
                    pattern=pattern,
                    higher_timeframe_trend=higher_tf,
                    risk_reward=risk_reward,
                    risk_state=risk_state,
                    atr_value=float(atr_value),
                    last_close=float(price),
                    symbol=symbol,
                    macro_context=_macro_dict,
                    df_lower_tf=df_lower_tf,
                    lower_tf_interval=lower_tf_interval,
                    stop_mode=_stop_mode,
                    stop_atr_mult=stop_atr_mult,
                    tp_atr_mult=tp_atr_mult,
                    base_bar_close_ts=ts,
                    mode=_integrated_mode,
                )
                v2_stop_plan = (decision_integrated.advisory or {}).get(
                    "structure_stop_plan"
                )
                # Reuse compare_v2_vs_current so the trace builder can
                # construct the v2 slice (it requires both decision and
                # comparison to be non-None).
                v2_compare_vs_current = compare_v2_vs_current(
                    decision_current=decision_current,
                    decision_v2=decision_integrated,
                )
                decision = decision_integrated
        else:
            decision = decision_current

        # Categorise the HOLD reason — the gate's blocked_by codes are
        # already a closed taxonomy; otherwise bucket by the last rule
        # chain step we got to before bailing.
        if decision.action == "HOLD":
            if decision.blocked_by:
                for code in decision.blocked_by:
                    _bump(result.hold_reasons, code)
            else:
                last_step = decision.rule_chain[-1] if decision.rule_chain else "unknown"
                _bump(result.hold_reasons, f"hold_{last_step}")

        # Don't open a second position if one is already running. A flip
        # from BUY to SELL on the same bar is rare and would over-trade
        # in a backtest — close on flip is handled at the NEXT bar via
        # mark-to-market once the new direction's stop/TP frames it.
        bar_entry_executed = False
        bar_entry_price: float | None = None
        bar_entry_skipped_reason = "entry_executed"  # set per case below
        if decision.action in ("BUY", "SELL") and pos is None:
            offset = stop_atr_mult * float(atr_value)
            tp_offset = tp_atr_mult * float(atr_value)
            if decision.action == "BUY":
                stop = price - offset
                tp = price + tp_offset
            else:
                stop = price + offset
                tp = price - tp_offset
            # v2 + stop_mode != "atr": consult the v2 structure stop plan
            # when valid. Default stop_mode="atr" keeps the legacy logic
            # unchanged (byte-identical for current_runtime / v1).
            if (
                _royal_road_v2_active and _stop_mode != "atr"
                and v2_stop_plan is not None
                and v2_stop_plan.get("stop_price") is not None
                and v2_stop_plan.get("take_profit_price") is not None
            ):
                stop = float(v2_stop_plan["stop_price"])
                tp = float(v2_stop_plan["take_profit_price"])
            # Position size: full cash / price — the legacy backtest's
            # convention. Real sizing is risk-fraction; we keep this
            # comparable to backtest.py for direct A/B reads.
            size = cash / price
            trade_counter += 1
            new_trade_id = f"{run_id}_T{trade_counter}" if run_id else ""
            pos = EnginePosition(
                side=decision.action,
                entry=price,
                entry_ts=ts,
                size=size,
                stop=stop,
                take_profit=tp,
                trade_id=new_trade_id,
            )
            pending_chain = decision.rule_chain
            pending_blocked = decision.blocked_by
            bar_entry_executed = True
            bar_entry_price = price
            bar_entry_skipped_reason = "entry_executed"
        elif decision.action in ("BUY", "SELL") and pos is not None:
            bar_entry_skipped_reason = "position_already_open"
        else:  # decision.action == "HOLD"
            bar_entry_skipped_reason = "decision_was_HOLD"

        equity = cash + _unrealized(pos, price)
        result.equity_curve.append((ts, equity))

        if capture_traces:
            trace = _build_full_trace(
                run_id=run_id,
                df=df,
                i=i,
                ts=ts,
                symbol=symbol,
                interval=interval,
                data_source=data_source,
                events_tuple=events_tuple,
                snap=snap,
                atr_value=float(atr_value),
                tech=tech,
                tech_reason_codes=tech_reason_codes,
                pattern=pattern,
                higher_tf=higher_tf,
                use_higher_tf=use_higher_tf,
                risk_state=risk_state,
                llm_signal=llm_signal,
                decision=decision,
                risk_reward=risk_reward,
                stop_atr_mult=stop_atr_mult,
                tp_atr_mult=tp_atr_mult,
                max_holding_bars=max_holding_bars,
                position_before=position_before,
                position_after=_position_dict(pos),
                bars_held_before=bars_held_before,
                had_open_position=had_open_position,
                bar_entry_executed=bar_entry_executed,
                bar_entry_price=bar_entry_price,
                bar_entry_skipped_reason=bar_entry_skipped_reason,
                bar_exit_event=bar_exit_event,
                bar_exit_reason=bar_exit_reason,
                bar_exit_price=bar_exit_price,
                bar_exit_trade_id=bar_exit_trade_id,
                macro=macro,
                waveform_bias=_compute_waveform_bias_for_bar(
                    df, i,
                    library=waveform_library,
                    library_id=waveform_library_id,
                    interval=interval,
                    window_bars=waveform_window_bars,
                    horizon_bars=waveform_horizon_bars,
                    method=waveform_method,
                    min_score=waveform_min_score,
                    min_samples=waveform_min_samples,
                    min_directional_share=waveform_min_share,
                ),
                long_term_trend_override=_long_term_trend_for_bar(i),
                runtime_overrides=_runtime_overrides_for_trace,
                royal_road_decision=decision_royal,
                royal_road_compare=royal_compare,
                # Route the integrated decision through the same v2
                # trace slot so visual_audit / decision_bridge can read
                # it via existing extractors. The slice's `profile`
                # field disambiguates ("royal_road_decision_v2" vs
                # "royal_road_decision_v2_integrated").
                royal_road_decision_v2=decision_v2 or decision_integrated,
                royal_road_v2_compare_vs_current=v2_compare_vs_current,
                royal_road_v2_compare_vs_v1=v2_compare_vs_v1,
            )
            result.decision_traces.append(trace)

    # Force-close anything still open at the end so equity / trade counts
    # don't omit the final position's PnL. The end_of_data exit must also
    # be reflected in the LAST decision_trace's execution_trace so the
    # trace is consistent with EngineTrade — pinned by
    # test_end_of_data_exit_recorded_in_last_trace.
    if pos is not None:
        ts = df.index[-1]
        price = float(df["close"].iloc[-1])
        direction = 1 if pos.side == "BUY" else -1
        pnl = (price - pos.entry) * pos.size * direction
        ret_pct = 100 * (price - pos.entry) / pos.entry * direction
        forced_trade_id = pos.trade_id
        result.trades.append(
            EngineTrade(
                side=pos.side,
                entry=pos.entry,
                exit=price,
                entry_ts=pos.entry_ts,
                exit_ts=ts,
                pnl=pnl,
                return_pct=ret_pct,
                bars_held=pos.bars_held,
                exit_reason="end_of_data",
                rule_chain=pending_chain,
                blocked_by=pending_blocked,
                trade_id=forced_trade_id,
            )
        )
        if capture_traces and result.decision_traces:
            last_trace = result.decision_traces[-1]
            new_exec = dataclasses.replace(
                last_trace.execution_trace,
                position_after=None,
                exit_event=True,
                exit_reason="end_of_data",
                exit_price=price,
                exit_trade_id=forced_trade_id,
                trade_id=(
                    last_trace.execution_trace.entry_trade_id
                    or forced_trade_id
                ),
            )
            # Patch the exit_check rule_check so it stays consistent with
            # the execution_trace — otherwise readers see exit_event=true
            # while exit_check still says "position remains open".
            new_checks = []
            for rc in last_trace.rule_checks:
                if rc.canonical_rule_id == "exit_check":
                    new_checks.append(dataclasses.replace(
                        rc,
                        result="PASS",
                        computed=True,
                        used_in_decision=True,
                        value={
                            "exited": True,
                            "exit_reason": "end_of_data",
                            "exit_price": price,
                            "forced_close_at_end_of_data": True,
                        },
                        reason="position force-closed at end_of_data",
                        source_chain_step="post_loop",
                    ))
                else:
                    new_checks.append(rc)
            last_trace.execution_trace = new_exec
            last_trace.rule_checks = tuple(new_checks)

    # Finalise run metadata now that we know the actual processed range,
    # then run the second pass that fills FutureOutcomeSlice. The second
    # pass runs over the completed traces — it cannot influence decisions.
    if capture_traces and result.run_metadata is not None:
        if result.decision_traces:
            result.run_metadata.bar_range = {
                "start": result.decision_traces[0].timestamp,
                "end": result.decision_traces[-1].timestamp,
                "n_bars": len(result.decision_traces),
                "warmup": warmup,
            }
        else:
            result.run_metadata.bar_range = {
                "start": None, "end": None, "n_bars": 0, "warmup": warmup,
            }

    if capture_traces and compute_future_outcome and result.decision_traces:
        _populate_future_outcomes(
            traces=result.decision_traces,
            df=df,
            interval=interval,
            stop_atr_mult=stop_atr_mult,
            tp_atr_mult=tp_atr_mult,
            max_holding_bars=max_holding_bars,
            atr_series=atr_series,
        )

    return result


def _unrealized(pos: EnginePosition | None, price: float) -> float:
    if pos is None:
        return 0.0
    direction = 1 if pos.side == "BUY" else -1
    return (price - pos.entry) * pos.size * direction


def _bump(d: dict[str, int], key: str) -> None:
    d[key] = d.get(key, 0) + 1


__all__ = [
    "EnginePosition",
    "EngineTrade",
    "EngineBacktestResult",
    "run_engine_backtest",
    "HIGHER_INTERVAL_MAP",
]
