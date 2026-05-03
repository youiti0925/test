"""Decision trace data model — `decision_trace_v1`.

Each bar processed by `backtest_engine.run_engine_backtest` produces one
`BarDecisionTrace`. The trace captures everything that was visible to the
engine at that bar AND what the engine actually did with it. The shape is
deliberately verbose: the goal is to make every HOLD/BUY/SELL decision
fully auditable after the fact, not to be compact.

Three correctness rules
-----------------------
1. **Decisions are not affected by trace logging.** Every field is read,
   not written, by the decision pipeline. Adding/removing trace fields
   must never change `result.trades` or `result.hold_reasons`.
2. **Future outcome is computed in a second pass.** `FutureOutcomeSlice`
   is filled AFTER the bar loop completes. The decision pipeline never
   sees future bars.
3. **Missing data is recorded, not silently dropped.** When a rule cannot
   be evaluated (e.g. spread is unobserved in offline backtest), the
   `RuleCheck` is still emitted with `result="SKIPPED"` and a reason.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

# Bumped only on backwards-incompatible schema changes. Pinned in
# test_run_metadata_attached_unique_and_complete so an accidental rename
# breaks immediately.
TRACE_SCHEMA_VERSION = "decision_trace_v1"


# ---------------------------------------------------------------------------
# Rule taxonomy
# ---------------------------------------------------------------------------
# 19 canonical rule_ids that EVERY trace must contain. The mapping name is
# the (canonical_rule_id -> rule_group) pair the user requested:
# canonical_rule_id is what code greps for; rule_group is the human bucket.

RULE_TAXONOMY: dict[str, str] = {
    # Risk-gate sub-checks (run inside risk_gate.evaluate)
    "data_quality": "data_health",
    "calendar_freshness": "calendar",
    "event_high": "event_risk",
    "spread_abnormal": "execution_quality",
    "daily_loss_cap": "pnl_safety",
    "consecutive_losses": "pnl_safety",
    "rule_unverified": "version_safety",
    "sentiment_spike": "sentiment_risk",
    # Indicator availability — checked by backtest_engine before decide()
    "atr_available": "indicator_health",
    # Decision-engine chain steps
    "technical_directionality": "technical",
    "pattern_check": "pattern",
    "higher_tf_alignment": "trend_alignment",
    "risk_reward_floor": "risk_reward",
    "llm_advisory": "advisory",
    "waveform_advisory": "advisory",
    # Position / execution rules — backtest_engine post-decision
    "position_state": "position",
    "entry_execution": "position",
    "exit_check": "position",
    # Terminal — final action confirmation
    "final_decision": "outcome",
}


# Closed taxonomy of entry_skipped_reason. The "entry_executed" sentinel
# means "no skip occurred" so the field is always populated, never null —
# every bar has a known disposition.
ENTRY_SKIPPED_REASONS: tuple[str, ...] = (
    "decision_was_HOLD",
    "position_already_open",
    "atr_unavailable",
    "invalid_price",
    "entry_executed",
)


# ---------------------------------------------------------------------------
# Future-outcome horizon table
# ---------------------------------------------------------------------------
# How many BARS make up a given wall-clock horizon, given the chart
# timeframe. None means the horizon is shorter than one bar of this
# interval and cannot be evaluated. Tested in
# test_future_outcome_horizon_bars_per_interval.

_INTERVAL_MINUTES: dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240,
    "1d": 1440,
}

_HORIZON_MINUTES: dict[str, int] = {"1h": 60, "4h": 240, "24h": 1440}


def horizon_to_bars(interval: str, horizon: str) -> int | None:
    """Convert ('1h', '24h') -> 24, ('4h', '1h') -> None."""
    if interval not in _INTERVAL_MINUTES or horizon not in _HORIZON_MINUTES:
        return None
    bars = _HORIZON_MINUTES[horizon] / _INTERVAL_MINUTES[interval]
    if bars < 1:
        return None
    return int(bars)


HORIZON_BARS_TABLE: dict[str, dict[str, int | None]] = {
    iv: {h: horizon_to_bars(iv, h) for h in _HORIZON_MINUTES}
    for iv in _INTERVAL_MINUTES
}


# ---------------------------------------------------------------------------
# Hash helpers — reproducibility metadata
# ---------------------------------------------------------------------------


def compute_strategy_config_hash(config: dict[str, Any]) -> str:
    """sha256 of a sorted-key JSON dump. Stable across runs with same params."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def compute_data_snapshot_hash(
    df: pd.DataFrame, symbol: str, interval: str
) -> str:
    """Strong fingerprint of the OHLCV data window.

    Includes every (timestamp, open, high, low, close, volume) row plus
    symbol/interval/length anchors. Two DataFrames produce the same hash
    iff their contents match exactly. Tested in
    test_data_snapshot_hash_stability and test_data_snapshot_hash_changes.
    """
    h = hashlib.sha256()
    h.update(symbol.encode())
    h.update(b"|")
    h.update(interval.encode())
    h.update(b"|")
    h.update(str(len(df)).encode())
    h.update(b"|")
    if len(df) > 0:
        h.update(str(df.index[0]).encode())
        h.update(b"|")
        h.update(str(df.index[-1]).encode())
        h.update(b"|")
        # Index timestamps (int64 ns since epoch) — order matters
        idx_ns = df.index.view("int64") if hasattr(df.index, "view") else (
            np.asarray(df.index.astype("int64"))
        )
        h.update(np.ascontiguousarray(idx_ns).tobytes())
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                arr = np.ascontiguousarray(df[col].to_numpy(dtype=np.float64))
                h.update(arr.tobytes())
                h.update(b"|")
    return h.hexdigest()


def get_commit_sha() -> tuple[str, str]:
    """Return (sha, status). status ∈ {resolved, unknown_no_git, unknown_subprocess_failed}."""
    if not os.path.exists(".git"):
        return ("unknown", "unknown_no_git")
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return (out.decode().strip(), "resolved")
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ("unknown", "unknown_subprocess_failed")


# ---------------------------------------------------------------------------
# JSON-safe scalar conversion
# ---------------------------------------------------------------------------


def _scalar(v: Any) -> Any:
    """Convert numpy/pandas scalars to vanilla Python for json.dumps."""
    if v is None:
        return None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def _scalar_dict(d: dict[str, Any]) -> dict[str, Any]:
    return {k: _scalar(v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# RunMetadata — 1 backtest = 1
# ---------------------------------------------------------------------------


@dataclass
class RunMetadata:
    run_id: str
    created_at: str
    symbol: str
    timeframe: str
    bar_range: dict[str, Any]
    trace_schema_version: str
    strategy_config_hash: str
    data_snapshot_hash: str
    input_data_source: str
    input_data_retrieved_at: str | None
    synthetic_execution: bool
    commit_sha: str
    commit_sha_status: str
    execution_mode: str
    engine_version: str
    timezone_name: str
    # PR #16: waveform-match run config in plain text. The same payload is
    # already folded into strategy_config_hash, but the hash is one-way —
    # without the readable copy you cannot tell from a stored run which
    # library / horizon / method produced its traces. Default None so
    # legacy callers (cmd_trade live exports etc.) continue to work.
    waveform: dict[str, Any] | None = None
    # PR #17: backtest --context-days run config. Records whether long-term
    # historical context was attached and over what window. The judgement
    # path NEVER reads this slice — context is observation-only via
    # `long_term_trend_slice` (decisions remain byte-identical with or
    # without context). Default None for live cmd_trade compatibility.
    context: dict[str, Any] | None = None
    # PR #18: events.json freshness / coverage / window-policy block. The
    # mtime-based freshness check no longer determines whether events flow
    # to the gate (research backtests use `backtest_warn_but_use` so
    # historical results are deterministic against the events.json
    # content). This metadata captures everything an audit needs to
    # reproduce a run: events_file_sha8, dropped_count, coverage, policy.
    # Default None so live cmd_trade RunMetadata construction stays
    # backwards compatible.
    calendar: dict[str, Any] | None = None
    # PR #19: literature-based parameter baseline catalog block. Records
    # baseline_id / baseline_version / baseline_payload_hash and the full
    # baseline payload + (optionally) the requested --parameter-profile
    # name. METADATA-ONLY: nothing here flows into decide_action or
    # risk_gate. `applied_to_runtime` is always False in PR #19; the
    # value exists so a future PR that DOES connect a profile to the
    # runtime can flip it without renaming the field. Default None so
    # live cmd_trade exports stay backwards compatible.
    parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar_range": dict(self.bar_range),
            "trace_schema_version": self.trace_schema_version,
            "strategy_config_hash": self.strategy_config_hash,
            "data_snapshot_hash": self.data_snapshot_hash,
            "input_data_source": self.input_data_source,
            "input_data_retrieved_at": self.input_data_retrieved_at,
            "synthetic_execution": self.synthetic_execution,
            "commit_sha": self.commit_sha,
            "commit_sha_status": self.commit_sha_status,
            "execution_mode": self.execution_mode,
            "engine_version": self.engine_version,
            "timezone": self.timezone_name,
            "waveform": dict(self.waveform) if self.waveform is not None else None,
            "context": dict(self.context) if self.context is not None else None,
            "calendar": dict(self.calendar) if self.calendar is not None else None,
            "parameters": dict(self.parameters) if self.parameters is not None else None,
        }


# ---------------------------------------------------------------------------
# Slice dataclasses — frozen=True so accidental writes during the bar
# loop become AttributeError. Mutability is reserved for FutureOutcomeSlice
# binding via dataclasses.replace in the second pass.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketSlice:
    open: float
    high: float
    low: float
    close: float
    volume: float
    data_source: str
    is_complete_bar: bool
    data_quality: str
    bars_available: int
    missing_ohlcv_count: int
    has_nan: bool
    index_monotonic: bool
    duplicate_timestamp_count: int
    timezone: str
    gap_detected: bool
    quality_reason: str | None

    def to_dict(self) -> dict:
        return _scalar_dict(self.__dict__)


@dataclass(frozen=True)
class TechnicalSlice:
    rsi_14: float
    macd: float
    macd_signal: float
    macd_hist: float
    sma_20: float
    sma_50: float
    ema_12: float
    bb_upper: float
    bb_lower: float
    bb_position: float
    change_pct_1h: float
    change_pct_24h: float
    atr_14: float | None
    technical_only_action: str
    technical_reason_codes: tuple[str, ...]
    reason_derivation: str
    # PR #21: parameter-runtime audit fields. The historical Snapshot
    # field name `rsi_14` does NOT change even when `rsi_period != 14`
    # — these `*_used` fields make the actually-used period explicit.
    # Default None preserves byte-identical to_dict() for pre-PR-#21
    # callers that don't pass values.
    rsi_period_used: int | None = None
    rsi_overbought_used: float | None = None
    rsi_oversold_used: float | None = None
    macd_fast_used: int | None = None
    macd_slow_used: int | None = None
    macd_signal_used: int | None = None
    bb_period_used: int | None = None
    bb_std_used: float | None = None
    atr_period_used: int | None = None
    stop_atr_mult_used: float | None = None
    tp_atr_mult_used: float | None = None
    max_holding_bars_used: int | None = None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["technical_reason_codes"] = list(self.technical_reason_codes)
        return _scalar_dict(d)


@dataclass(frozen=True)
class WaveformSlice:
    trend_state: str
    detected_pattern: str | None
    pattern_confidence: float
    neckline: float | None
    neckline_broken: bool
    distance_to_neckline_atr: float | None
    rsi_divergence: bool
    macd_momentum_weakening: bool
    swing_points_recent: tuple[dict, ...]
    waveform_bias: dict | None
    waveform_reason_codes: tuple[str, ...]

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["swing_points_recent"] = [_scalar_dict(s) for s in self.swing_points_recent]
        d["waveform_reason_codes"] = list(self.waveform_reason_codes)
        return _scalar_dict(d)


@dataclass(frozen=True)
class HigherTimeframeSlice:
    source_interval: str
    trend: str
    alignment: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class LongTermTrendSlice:
    """Multi-timeframe long-term context observable at this bar.

    All values are computed from `df.iloc[: i + 1]` only — never reach
    forward. Trend labels reuse `patterns.analyse` on resampled data.

    SMA convention (canonical "N-day SMA"):
      `sma_30d` / `sma_90d` / `sma_200d` are the **mean of the last N
      *daily* closes**. Hourly bars are first resampled to one close per
      calendar day (the day's last bar, weekends dropped), then the SMA
      is the simple mean of the trailing N daily closes. This matches
      market convention so that "200-day SMA" reads the way a chartist
      expects (≈ 200 trading-day average), independent of base interval.
      `unavailable_reasons` reports `"only K daily closes (need ≥N)"`
      whenever fewer than N daily closes exist yet.

    N-day return convention:
      `weekly_return_pct` / `monthly_return_pct` / `quarterly_return_pct`
      are `(close[ts] / close[ts − N days] − 1) × 100`. The reference
      close is the most recent close at-or-before `ts − N days`, so
      weekend gaps fall back to the prior trading day automatically.

    `unavailable_reasons` is populated when a value cannot be computed
    yet (warmup, sparse history). Consumers MUST treat None as "unknown",
    not as zero.
    """

    daily_trend: str
    weekly_trend: str
    monthly_trend: str

    sma_30d: float | None
    sma_90d: float | None
    sma_200d: float | None

    close_vs_sma_30d_pct: float | None
    close_vs_sma_90d_pct: float | None
    close_vs_sma_200d_pct: float | None

    weekly_return_pct: float | None
    monthly_return_pct: float | None
    quarterly_return_pct: float | None

    bars_available: int
    unavailable_reasons: dict

    # PR #20: SMA50 observability. literature_baseline_v1 records SMA50 as
    # the canonical fast SMA but pre-PR-#20 traces only carried 30/90/200.
    # Adding 50d alongside keeps backward compat (older traces still parse,
    # these fields default to None on the dataclass — populated by
    # `long_term_trend_slice`). NOT used by decide_action / risk_gate.
    sma_50d: float | None = None
    close_vs_sma_50d_pct: float | None = None
    sma_50_vs_sma_200_pct: float | None = None
    # `BULLISH` (sma50 >= sma200 × (1 + dead_band)) /
    # `BEARISH` (sma50 <= sma200 × (1 − dead_band)) /
    # `NEUTRAL` (within dead band) / `UNKNOWN` (data unavailable).
    # Dead band is `_SMA_50_200_DEAD_BAND_PCT` (default 0.5%) — observation
    # constant, NOT in PARAMETER_BASELINE_V1 (would change baseline hash).
    sma_50_200_state: str | None = None

    # PR #20: monthly_trend classification audit. The catalog re-verification
    # found 5666/5666 bars labelled RANGE — these fields expose the
    # classification inputs (return / slope / volatility) and the threshold
    # actually used so the all-RANGE pattern can be diagnosed without
    # changing the classification logic itself.
    monthly_volatility_pct: float | None = None
    monthly_slope_per_bar: float | None = None
    monthly_trend_classification_inputs: dict | None = None
    monthly_trend_classification_threshold: dict | None = None
    monthly_trend_classification_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "daily_trend": self.daily_trend,
            "weekly_trend": self.weekly_trend,
            "monthly_trend": self.monthly_trend,
            "sma_30d": self.sma_30d,
            "sma_50d": self.sma_50d,
            "sma_90d": self.sma_90d,
            "sma_200d": self.sma_200d,
            "close_vs_sma_30d_pct": self.close_vs_sma_30d_pct,
            "close_vs_sma_50d_pct": self.close_vs_sma_50d_pct,
            "close_vs_sma_90d_pct": self.close_vs_sma_90d_pct,
            "close_vs_sma_200d_pct": self.close_vs_sma_200d_pct,
            "sma_50_vs_sma_200_pct": self.sma_50_vs_sma_200_pct,
            "sma_50_200_state": self.sma_50_200_state,
            "weekly_return_pct": self.weekly_return_pct,
            "monthly_return_pct": self.monthly_return_pct,
            "quarterly_return_pct": self.quarterly_return_pct,
            "monthly_volatility_pct": self.monthly_volatility_pct,
            "monthly_slope_per_bar": self.monthly_slope_per_bar,
            "monthly_trend_classification_inputs": (
                dict(self.monthly_trend_classification_inputs)
                if self.monthly_trend_classification_inputs is not None else None
            ),
            "monthly_trend_classification_threshold": (
                dict(self.monthly_trend_classification_threshold)
                if self.monthly_trend_classification_threshold is not None else None
            ),
            "monthly_trend_classification_reason": (
                self.monthly_trend_classification_reason
            ),
            "bars_available": self.bars_available,
            "unavailable_reasons": dict(self.unavailable_reasons),
        }


@dataclass(frozen=True)
class MacroContextSlice:
    """Point-in-time macro / market context observable at this bar.

    Levels come from `MacroSnapshot.value_at(slot, ts)` (which uses
    Series.asof — strictly past-only). Deltas are computed against
    earlier `value_at(slot, ts - delta)` calls and are also strictly
    past-only.

    Units (single source of truth — fields are named to match):
      * `*_pct` fields are percent change (e.g. `dxy_change_5d_pct = -1.5`
        means DXY fell 1.5 % over the prior 5 days).
      * `*_bp` fields are **basis points** (1 bp = 0.01 percentage point).
        So `us10y_change_24h_bp = 10` means the 10-year yield rose by
        10 basis points (i.e. 0.10 percentage points). Yield deltas are
        always reported in bp to match market convention; do not mix.

    Every field is optional because:
      * the macro fetch may have failed at backtest start,
      * a particular slot's history may not cover this bar's timestamp,
      * deltas may not exist if the earlier reference point lacks data.
    Consumers MUST treat None as "unknown", not as zero.
    """

    # Levels at ts (point-in-time)
    us10y: float | None
    us_short_yield_proxy: float | None
    yield_spread_long_short: float | None
    dxy: float | None
    vix: float | None
    sp500: float | None
    nasdaq: float | None
    nikkei: float | None

    # Deltas — yields in basis points (bp), prices in percent (pct)
    dxy_change_24h_pct: float | None
    dxy_change_5d_pct: float | None
    us10y_change_24h_bp: float | None
    us10y_change_5d_bp: float | None
    yield_spread_change_5d_bp: float | None
    vix_change_24h_pct: float | None
    sp500_change_24h_pct: float | None
    nasdaq_change_24h_pct: float | None

    # Provenance
    available_slots: tuple[str, ...]
    missing_slots: tuple[str, ...]
    fetch_errors: dict

    # PR #20: DXY trend / return / z-score observability. R5
    # (DXY trend × USD exposure × outcome) was un-evaluable in the
    # PR-#19 verification because trace had only `dxy` level — no
    # historical context. These fields close that gap. ALL
    # observation-only — never read by decide_action / risk_gate.
    # Returns are calendar-day basis (not bars) per Q-clarification.
    # `*_unavailable_reason` carries the structured reason when a
    # field is None (e.g. "insufficient_history_for_20d", "stdev_zero").
    dxy_return_5d_pct: float | None = None
    dxy_return_20d_pct: float | None = None
    dxy_zscore_20d: float | None = None
    dxy_zscore_60d: float | None = None
    # Buckets: STRONG_DOWN (<-2%) / DOWN (-2..-0.5%) / FLAT (-0.5..0.5%) /
    # UP (0.5..2%) / STRONG_UP (>=2%). Thresholds live in
    # `_DXY_TREND_BUCKET_THRESHOLDS` in decision_trace_build.py.
    dxy_trend_5d_bucket: str | None = None
    dxy_trend_20d_bucket: str | None = None
    # Z-score bucket (uses 20d z-score by default):
    # EXTREME_LOW (<-2) / LOW (-2..-1) / NEUTRAL (-1..1) /
    # HIGH (1..2) / EXTREME_HIGH (>=2).
    dxy_zscore_bucket: str | None = None
    dxy_unavailable_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "us10y": self.us10y,
            "us_short_yield_proxy": self.us_short_yield_proxy,
            "yield_spread_long_short": self.yield_spread_long_short,
            "dxy": self.dxy,
            "vix": self.vix,
            "sp500": self.sp500,
            "nasdaq": self.nasdaq,
            "nikkei": self.nikkei,
            "dxy_change_24h_pct": self.dxy_change_24h_pct,
            "dxy_change_5d_pct": self.dxy_change_5d_pct,
            "dxy_return_5d_pct": self.dxy_return_5d_pct,
            "dxy_return_20d_pct": self.dxy_return_20d_pct,
            "dxy_zscore_20d": self.dxy_zscore_20d,
            "dxy_zscore_60d": self.dxy_zscore_60d,
            "dxy_trend_5d_bucket": self.dxy_trend_5d_bucket,
            "dxy_trend_20d_bucket": self.dxy_trend_20d_bucket,
            "dxy_zscore_bucket": self.dxy_zscore_bucket,
            "dxy_unavailable_reason": self.dxy_unavailable_reason,
            "us10y_change_24h_bp": self.us10y_change_24h_bp,
            "us10y_change_5d_bp": self.us10y_change_5d_bp,
            "yield_spread_change_5d_bp": self.yield_spread_change_5d_bp,
            "vix_change_24h_pct": self.vix_change_24h_pct,
            "sp500_change_24h_pct": self.sp500_change_24h_pct,
            "nasdaq_change_24h_pct": self.nasdaq_change_24h_pct,
            "available_slots": list(self.available_slots),
            "missing_slots": list(self.missing_slots),
            "fetch_errors": dict(self.fetch_errors),
        }


@dataclass(frozen=True)
class TechnicalConfluenceSlice:
    """Royal-road technical confluence observation (PR-C1).

    Wraps the dict returned by `technical_confluence.build_technical_confluence`.
    Stored on `BarDecisionTrace.technical_confluence` (Optional). All
    fields are observation-only: they are NEVER read by `decide_action`,
    `risk_gate`, or any RuleCheck. Adding/removing fields here MUST NOT
    change `result.trades` or `BarDecisionTrace.decision.final_action`
    — pinned by `tests/test_technical_confluence_trace_observation_only.py`.

    Schema (each value is itself a JSON-serializable dict; see
    `technical_confluence.py` for the canonical key list):
      policy_version       — str, e.g. "technical_confluence_v1"
      market_regime        — TREND_UP|TREND_DOWN|RANGE|VOLATILE|UNKNOWN
      dow_structure        — last swing levels, structure_code, BOS flags
      support_resistance   — nearest levels, distances, near/breakout flags
      candlestick_signal   — pinbar / engulfing / harami / strong body
      chart_pattern        — mirror of pattern.detected_pattern + neckline
      indicator_context    — RSI regime / BB lifecycle / MACD momentum
      risk_plan_obs        — ATR stop vs structure stop, RR observations
      vote_breakdown       — indicator vote audit + per-source alignment
      final_confluence     — STRONG_BUY_SETUP / WEAK_BUY_SETUP / NO_TRADE / ...
    """

    policy_version: str
    market_regime: str
    dow_structure: dict
    support_resistance: dict
    candlestick_signal: dict
    chart_pattern: dict
    indicator_context: dict
    risk_plan_obs: dict
    vote_breakdown: dict
    final_confluence: dict

    def to_dict(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "market_regime": self.market_regime,
            "dow_structure": dict(self.dow_structure),
            "support_resistance": dict(self.support_resistance),
            "candlestick_signal": dict(self.candlestick_signal),
            "chart_pattern": dict(self.chart_pattern),
            "indicator_context": dict(self.indicator_context),
            "risk_plan_obs": dict(self.risk_plan_obs),
            "vote_breakdown": dict(self.vote_breakdown),
            "final_confluence": dict(self.final_confluence),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TechnicalConfluenceSlice":
        return cls(
            policy_version=str(d.get("policy_version", "technical_confluence_v1")),
            market_regime=str(d.get("market_regime", "UNKNOWN")),
            dow_structure=dict(d.get("dow_structure", {})),
            support_resistance=dict(d.get("support_resistance", {})),
            candlestick_signal=dict(d.get("candlestick_signal", {})),
            chart_pattern=dict(d.get("chart_pattern", {})),
            indicator_context=dict(d.get("indicator_context", {})),
            risk_plan_obs=dict(d.get("risk_plan_obs", {})),
            vote_breakdown=dict(d.get("vote_breakdown", {})),
            final_confluence=dict(d.get("final_confluence", {})),
        )


@dataclass(frozen=True)
class RoyalRoadDecisionSlice:
    """Royal-road profile decision audit (PR royal_road_decision_v1).

    Emitted ONLY when `run_engine_backtest(decision_profile=
    "royal_road_decision_v1")` is requested. The default profile
    (`current_runtime`) leaves `BarDecisionTrace.royal_road_decision`
    as None — preserving byte-identical trace output for legacy /
    default callers.

    The slice records the royal-road profile's BUY/SELL/HOLD output
    AND a comparison block so post-hoc analyses can identify trades
    where the two profiles disagreed.
    """

    profile: str
    action: str
    confidence: float
    score: float
    reasons: list
    block_reasons: list
    compared_to_current_runtime: dict
    # royal_road_decision_v1 mode metadata (added when the profile is
    # split into strict / balanced / exploratory). Defaults preserve
    # backward compat for any caller that constructs the slice without
    # them — e.g. live cmd_trade legacy export paths that never produce
    # this slice anyway.
    mode: str = "balanced"
    mode_status: str = "heuristic_not_validated_default"
    mode_needs_validation: bool = True
    cautions: list = field(default_factory=list)
    evidence_axes: dict = field(default_factory=dict)
    evidence_axes_count: dict = field(default_factory=dict)
    min_evidence_axes_required: int | None = None

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "action": self.action,
            "confidence": float(self.confidence),
            "score": float(self.score),
            "reasons": list(self.reasons),
            "block_reasons": list(self.block_reasons),
            "compared_to_current_runtime": dict(
                self.compared_to_current_runtime
            ),
            "mode": self.mode,
            "mode_status": self.mode_status,
            "mode_needs_validation": bool(self.mode_needs_validation),
            "cautions": list(self.cautions),
            "evidence_axes": dict(self.evidence_axes),
            "evidence_axes_count": dict(self.evidence_axes_count),
            "min_evidence_axes_required": self.min_evidence_axes_required,
        }

    @classmethod
    def from_decision(
        cls,
        *,
        royal_decision,
        comparison: dict,
    ) -> "RoyalRoadDecisionSlice":
        adv = royal_decision.advisory or {}
        return cls(
            profile=str(adv.get("profile", "royal_road_decision_v1")),
            action=royal_decision.action,
            confidence=float(royal_decision.confidence),
            score=float(adv.get("score", 0.0)),
            reasons=list(adv.get("reasons") or []),
            block_reasons=list(adv.get("block_reasons") or []),
            compared_to_current_runtime=dict(comparison),
            mode=str(adv.get("mode", "balanced")),
            mode_status=str(
                adv.get("mode_status", "heuristic_not_validated_default")
            ),
            mode_needs_validation=bool(
                adv.get("mode_needs_validation", True)
            ),
            cautions=list(adv.get("cautions") or []),
            evidence_axes=dict(adv.get("evidence_axes") or {}),
            evidence_axes_count=dict(adv.get("evidence_axes_count") or {}),
            min_evidence_axes_required=adv.get("min_evidence_axes_required"),
        )


@dataclass(frozen=True)
class RoyalRoadDecisionV2Slice:
    """Royal-road v2 decision audit (royal_road_decision_v2 profile).

    Emitted ONLY when run_engine_backtest(decision_profile=
    "royal_road_decision_v2") is requested. Default current_runtime
    leaves this field None on every bar, preserving byte-identical
    trace output for legacy callers.

    Carries the full v2 audit:
      profile / mode / mode_status / mode_needs_validation
      action / confidence / score
      reasons / block_reasons / cautions
      evidence_axes / evidence_axes_count / min_evidence_axes_required
      support_resistance_v2 / trendline_context / chart_pattern_v2
      lower_tf_trigger / macro_alignment / structure_stop_plan
      compared_to_current_runtime / compared_to_royal_road_v1
    """

    profile: str
    mode: str
    mode_status: str
    mode_needs_validation: bool
    action: str
    confidence: float
    score: float
    reasons: list
    block_reasons: list
    cautions: list
    evidence_axes: dict
    evidence_axes_count: dict
    min_evidence_axes_required: int | None
    support_resistance_v2: dict
    trendline_context: dict
    chart_pattern_v2: dict
    lower_tf_trigger: dict
    macro_alignment: dict
    structure_stop_plan: dict | None
    compared_to_current_runtime: dict
    compared_to_royal_road_v1: dict | None
    # v2.2: extended trace fields. Defaults preserve backward compat.
    setup_candidates: list = field(default_factory=list)
    best_setup: dict | None = None
    reconstruction_quality: dict = field(default_factory=dict)
    multi_scale_chart: dict = field(default_factory=dict)
    # Phase F + G: integrated profile carries these. Defaults None
    # so legacy v2 / v1 / current_runtime traces keep their byte-
    # identical shape.
    integrated_decision: dict | None = None
    pattern_levels: dict | None = None
    entry_plan: dict | None = None
    breakout_quality_gate: dict | None = None
    fundamental_sidebar: dict | None = None
    wave_shape_review: dict | None = None
    wave_derived_lines: list = field(default_factory=list)
    # Phase I-1/I-2: entry-candidate observation layer (integrated
    # profile only). Empty defaults preserve byte-identical traces
    # for legacy v2 / v1 / current_runtime callers.
    entry_candidates: list = field(default_factory=list)
    selected_entry_candidate: dict = field(default_factory=dict)
    # Phase I follow-up: royal-road procedure checklist
    # (integrated profile only). Empty default again preserves
    # byte-identical legacy traces.
    royal_road_procedure_checklist: dict = field(default_factory=dict)
    # Phase I follow-up: structural lines snapshot (integrated
    # profile only). Empty default preserves byte-identical traces
    # for legacy v2 / v1 / current_runtime callers.
    structural_lines: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "mode": self.mode,
            "mode_status": self.mode_status,
            "mode_needs_validation": bool(self.mode_needs_validation),
            "action": self.action,
            "confidence": float(self.confidence),
            "score": float(self.score),
            "reasons": list(self.reasons),
            "block_reasons": list(self.block_reasons),
            "cautions": list(self.cautions),
            "evidence_axes": dict(self.evidence_axes),
            "evidence_axes_count": dict(self.evidence_axes_count),
            "min_evidence_axes_required": self.min_evidence_axes_required,
            "support_resistance_v2": dict(self.support_resistance_v2),
            "trendline_context": dict(self.trendline_context),
            "chart_pattern_v2": dict(self.chart_pattern_v2),
            "lower_tf_trigger": dict(self.lower_tf_trigger),
            "macro_alignment": dict(self.macro_alignment),
            "structure_stop_plan": (
                dict(self.structure_stop_plan)
                if self.structure_stop_plan is not None else None
            ),
            "compared_to_current_runtime": dict(
                self.compared_to_current_runtime
            ),
            "compared_to_royal_road_v1": (
                dict(self.compared_to_royal_road_v1)
                if self.compared_to_royal_road_v1 is not None else None
            ),
            "setup_candidates": list(self.setup_candidates),
            "best_setup": (
                dict(self.best_setup) if self.best_setup is not None else None
            ),
            "reconstruction_quality": dict(self.reconstruction_quality),
            "multi_scale_chart": dict(self.multi_scale_chart),
            # Phase F + G fields (integrated profile only)
            "integrated_decision": (
                dict(self.integrated_decision)
                if self.integrated_decision is not None else None
            ),
            "pattern_levels": (
                dict(self.pattern_levels)
                if self.pattern_levels is not None else None
            ),
            "entry_plan": (
                dict(self.entry_plan)
                if self.entry_plan is not None else None
            ),
            "breakout_quality_gate": (
                dict(self.breakout_quality_gate)
                if self.breakout_quality_gate is not None else None
            ),
            "fundamental_sidebar": (
                dict(self.fundamental_sidebar)
                if self.fundamental_sidebar is not None else None
            ),
            "wave_shape_review": (
                dict(self.wave_shape_review)
                if self.wave_shape_review is not None else None
            ),
            "wave_derived_lines": list(self.wave_derived_lines),
            # Phase I-1/I-2 entry-candidate observation layer.
            "entry_candidates": list(self.entry_candidates),
            "selected_entry_candidate": dict(self.selected_entry_candidate),
            # Phase I follow-up royal-road procedure checklist.
            "royal_road_procedure_checklist": dict(
                self.royal_road_procedure_checklist
            ),
            # Phase I follow-up structural lines snapshot.
            "structural_lines": dict(self.structural_lines),
        }

    @classmethod
    def from_decision(
        cls,
        *,
        v2_decision,
        comparison_vs_current: dict,
        comparison_vs_v1: dict | None,
    ) -> "RoyalRoadDecisionV2Slice":
        adv = v2_decision.advisory or {}
        return cls(
            profile=str(adv.get("profile", "royal_road_decision_v2")),
            mode=str(adv.get("mode", "balanced")),
            mode_status=str(
                adv.get("mode_status", "heuristic_not_validated_default")
            ),
            mode_needs_validation=bool(adv.get("mode_needs_validation", True)),
            action=v2_decision.action,
            confidence=float(v2_decision.confidence),
            score=float(adv.get("score", 0.0)),
            reasons=list(adv.get("reasons") or []),
            block_reasons=list(adv.get("block_reasons") or []),
            cautions=list(adv.get("cautions") or []),
            evidence_axes=dict(adv.get("evidence_axes") or {}),
            evidence_axes_count=dict(adv.get("evidence_axes_count") or {}),
            min_evidence_axes_required=adv.get("min_evidence_axes_required"),
            support_resistance_v2=dict(adv.get("support_resistance_v2") or {}),
            trendline_context=dict(adv.get("trendline_context") or {}),
            chart_pattern_v2=dict(adv.get("chart_pattern_v2") or {}),
            lower_tf_trigger=dict(adv.get("lower_tf_trigger") or {}),
            macro_alignment=dict(adv.get("macro_alignment") or {}),
            structure_stop_plan=adv.get("structure_stop_plan"),
            compared_to_current_runtime=dict(comparison_vs_current),
            compared_to_royal_road_v1=(
                dict(comparison_vs_v1) if comparison_vs_v1 else None
            ),
            setup_candidates=list(adv.get("setup_candidates") or []),
            best_setup=adv.get("best_setup"),
            reconstruction_quality=dict(adv.get("reconstruction_quality") or {}),
            multi_scale_chart=dict(adv.get("multi_scale_chart") or {}),
            # Phase F + G — only the integrated profile populates these.
            # Legacy v2 / v1 / current_runtime leave them None.
            integrated_decision=adv.get("integrated_decision"),
            pattern_levels=adv.get("pattern_levels"),
            entry_plan=adv.get("entry_plan"),
            breakout_quality_gate=adv.get("breakout_quality_gate"),
            fundamental_sidebar=adv.get("fundamental_sidebar"),
            wave_shape_review=adv.get("wave_shape_review"),
            wave_derived_lines=list(adv.get("wave_derived_lines") or []),
            # Phase I-1/I-2 entry-candidate observation layer.
            entry_candidates=list(adv.get("entry_candidates") or []),
            selected_entry_candidate=dict(
                adv.get("selected_entry_candidate") or {}
            ),
            # Phase I follow-up royal-road procedure checklist.
            royal_road_procedure_checklist=dict(
                adv.get("royal_road_procedure_checklist") or {}
            ),
            # Phase I follow-up structural lines snapshot.
            structural_lines=dict(adv.get("structural_lines") or {}),
        )


@dataclass(frozen=True)
class FundamentalSlice:
    nearby_events: tuple[dict, ...]
    blocking_events: tuple[dict, ...]
    warning_events: tuple[dict, ...]
    event_evidence_ids: tuple[str, ...]
    missing_event_evidence_reason: str
    macro_observations: tuple[dict, ...]
    macro_evidence_ids: tuple[str, ...]
    missing_macro_evidence_reason: str
    news_evidence_ids: tuple[str, ...]
    missing_news_evidence_reason: str
    data_provenance: dict

    def to_dict(self) -> dict:
        return {
            "nearby_events": [_scalar_dict(e) for e in self.nearby_events],
            "blocking_events": [_scalar_dict(e) for e in self.blocking_events],
            "warning_events": [_scalar_dict(e) for e in self.warning_events],
            "event_evidence_ids": list(self.event_evidence_ids),
            "missing_event_evidence_reason": self.missing_event_evidence_reason,
            "macro_observations": [_scalar_dict(o) for o in self.macro_observations],
            "macro_evidence_ids": list(self.macro_evidence_ids),
            "missing_macro_evidence_reason": self.missing_macro_evidence_reason,
            "news_evidence_ids": list(self.news_evidence_ids),
            "missing_news_evidence_reason": self.missing_news_evidence_reason,
            "data_provenance": dict(self.data_provenance),
        }


@dataclass(frozen=True)
class ExecutionAssumptionSlice:
    synthetic_execution: bool
    fill_model: str
    spread_mode: str
    slippage_mode: str
    bid_ask_mode: str
    sentiment_archive: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ExecutionTraceSlice:
    position_before: dict | None
    position_after: dict | None
    had_open_position: bool
    entry_signal: str
    entry_executed: bool
    entry_price: float | None
    entry_skipped_reason: str
    exit_event: bool
    exit_reason: str | None
    exit_price: float | None
    # Distinct from `trade_id`: when exit and entry happen on the same bar
    # (e.g. existing position hits stop and a fresh BUY/SELL fires) we need
    # both ids to reconstruct what closed and what opened. Either or both
    # may be None on bars that did not entry/exit.
    entry_trade_id: str | None
    exit_trade_id: str | None
    # Compat field — equal to entry_trade_id when an entry executed,
    # else exit_trade_id when an exit fired, else None. Older readers
    # that only know `trade_id` keep working; new readers should prefer
    # the explicit entry_trade_id / exit_trade_id pair.
    trade_id: str | None
    bars_held_before: int | None
    bars_held_after: int | None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["position_before"] = (
            _scalar_dict(self.position_before) if self.position_before else None
        )
        d["position_after"] = (
            _scalar_dict(self.position_after) if self.position_after else None
        )
        return _scalar_dict(d)


@dataclass(frozen=True)
class RuleCheck:
    canonical_rule_id: str
    rule_group: str
    result: str  # PASS | BLOCK | WARN | SKIPPED | NOT_REACHED | INFO
    computed: bool
    used_in_decision: bool
    value: Any
    threshold: Any
    evidence_ids: tuple[str, ...]
    reason: str
    source_chain_step: str | None

    def to_dict(self) -> dict:
        d = {
            "canonical_rule_id": self.canonical_rule_id,
            "rule_group": self.rule_group,
            "result": self.result,
            "computed": self.computed,
            "used_in_decision": self.used_in_decision,
            "value": _scalar(self.value) if not isinstance(self.value, dict) else _scalar_dict(self.value),
            "threshold": _scalar(self.threshold) if not isinstance(self.threshold, dict) else _scalar_dict(self.threshold),
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
            "source_chain_step": self.source_chain_step,
        }
        return d


@dataclass(frozen=True)
class DecisionSlice:
    technical_only_action: str
    final_action: str
    action_changed_by_engine: bool
    blocked_by: tuple[str, ...]
    rule_chain: tuple[str, ...]
    confidence: float
    reason: str
    advisory: dict

    def to_dict(self) -> dict:
        return {
            "technical_only_action": self.technical_only_action,
            "final_action": self.final_action,
            "action_changed_by_engine": self.action_changed_by_engine,
            "blocked_by": list(self.blocked_by),
            "rule_chain": list(self.rule_chain),
            "confidence": float(self.confidence),
            "reason": self.reason,
            "advisory": dict(self.advisory),
        }


@dataclass(frozen=True)
class FutureOutcomeSlice:
    horizons_bars: dict
    future_return_1h_if_buy_pct: float | None
    future_return_4h_if_buy_pct: float | None
    future_return_24h_if_buy_pct: float | None
    future_return_1h_if_sell_pct: float | None
    future_return_4h_if_sell_pct: float | None
    future_return_24h_if_sell_pct: float | None
    mfe_24h_if_buy_pct: float | None
    mae_24h_if_buy_pct: float | None
    mfe_24h_if_sell_pct: float | None
    mae_24h_if_sell_pct: float | None
    available_horizons: tuple[str, ...]
    unavailable_horizons: dict[str, str]
    outcome_if_technical_action_taken: str
    gate_effect: str
    # Hypothetical trade simulation: what would happen if we entered at
    # this bar's close in technical_only_action direction with the same
    # stop/TP/max_holding as the engine? Verification-only — never read
    # by decide() or any rule_check.
    hypothetical_technical_trade_exit_reason: str | None
    hypothetical_technical_trade_exit_price: float | None
    hypothetical_technical_trade_bars_held: int | None
    hypothetical_technical_trade_return_pct: float | None

    # PR #20: Lightweight shadow future outcome for blocked bars.
    # Populated when `decision.blocked_by` is non-empty so PR #19
    # verification's COST_OPPORTUNITY / PROTECTED / WIN_MISSED /
    # LOSS_AVOIDED tally can be computed retroactively. Forward-looking
    # by design — these fields are EX-POST DIAGNOSTIC ONLY and are
    # NEVER passed back to decide_action or risk_gate. Default None
    # so non-blocked bars and pre-PR-#20 traces keep the same shape.
    blocked_future_return_24h_pct: float | None = None
    blocked_future_return_24h_if_buy_pct: float | None = None
    blocked_future_return_24h_if_sell_pct: float | None = None
    # `UP` / `DOWN` / `FLAT`. Threshold for FLAT is
    # `_SHADOW_OUTCOME_THRESHOLD_PCT` (default 0.3%) in
    # decision_trace_build.py — observation constant, NOT in
    # PARAMETER_BASELINE_V1 (would change baseline hash).
    blocked_outcome_direction: str | None = None
    # Bucket relative to `decision.technical_only_action`:
    #   STRONG_FOR / FOR / NEUTRAL / AGAINST / STRONG_AGAINST.
    # Used by `r_candidates_v1` to compute PROTECTED / COST_OPPORTUNITY.
    blocked_outcome_bucket: str | None = None

    def to_dict(self) -> dict:
        return _scalar_dict(self.__dict__)


# ---------------------------------------------------------------------------
# BarDecisionTrace — root
# ---------------------------------------------------------------------------


@dataclass
class BarDecisionTrace:
    run_id: str
    trace_schema_version: str
    bar_id: str
    timestamp: str
    symbol: str
    timeframe: str
    bar_index: int

    market: MarketSlice
    technical: TechnicalSlice
    waveform: WaveformSlice
    higher_timeframe: HigherTimeframeSlice
    fundamental: FundamentalSlice
    execution_assumption: ExecutionAssumptionSlice
    execution_trace: ExecutionTraceSlice
    rule_checks: tuple[RuleCheck, ...]
    decision: DecisionSlice
    # Filled in second pass; never read by the decision pipeline.
    future_outcome: FutureOutcomeSlice | None = None
    # Optional multi-timeframe long-term context. Optional so older trace
    # JSONL files (pre-PR-A) still parse cleanly.
    long_term_trend: LongTermTrendSlice | None = None
    # Optional macro / market-context block (DXY, US10Y, VIX, indices).
    # Populated when backtest_engine receives a MacroSnapshot. Optional
    # for backward compat and for runs where macro fetch failed.
    macro_context: MacroContextSlice | None = None
    # Optional royal-road technical confluence observation (PR-C1).
    # Populated unconditionally by decision_trace_build for every bar
    # the trace is built (warmup bars get an `UNKNOWN`-shaped dict via
    # `technical_confluence.empty_technical_confluence`). Default None
    # preserves backward compatibility for older trace JSONL readers
    # and for live cmd_trace export paths that have not been wired yet.
    # OBSERVATION-ONLY — never read by decide_action / risk_gate.
    technical_confluence: TechnicalConfluenceSlice | None = None
    # Optional royal-road decision audit (royal_road_decision_v1
    # profile). Emitted ONLY when `run_engine_backtest(decision_profile
    # ="royal_road_decision_v1")` is requested. None when the default
    # `current_runtime` profile is in effect, which preserves
    # byte-identical trace output for legacy callers and the live
    # cmd_trade export path.
    royal_road_decision: RoyalRoadDecisionSlice | None = None
    # royal_road_decision_v2 profile audit. Emitted ONLY when
    # run_engine_backtest(decision_profile="royal_road_decision_v2").
    # Default None preserves trace shape for current_runtime / v1.
    royal_road_decision_v2: "RoyalRoadDecisionV2Slice | None" = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "trace_schema_version": self.trace_schema_version,
            "bar_id": self.bar_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "bar_index": self.bar_index,
            "market": self.market.to_dict(),
            "technical": self.technical.to_dict(),
            "waveform": self.waveform.to_dict(),
            "higher_timeframe": self.higher_timeframe.to_dict(),
            "long_term_trend": (
                self.long_term_trend.to_dict() if self.long_term_trend else None
            ),
            "macro_context": (
                self.macro_context.to_dict() if self.macro_context else None
            ),
            "fundamental": self.fundamental.to_dict(),
            "execution_assumption": self.execution_assumption.to_dict(),
            "execution_trace": self.execution_trace.to_dict(),
            "rule_checks": [rc.to_dict() for rc in self.rule_checks],
            "decision": self.decision.to_dict(),
            "future_outcome": (
                self.future_outcome.to_dict() if self.future_outcome else None
            ),
            "technical_confluence": (
                self.technical_confluence.to_dict()
                if self.technical_confluence else None
            ),
            "royal_road_decision": (
                self.royal_road_decision.to_dict()
                if self.royal_road_decision else None
            ),
            "royal_road_decision_v2": (
                self.royal_road_decision_v2.to_dict()
                if self.royal_road_decision_v2 else None
            ),
        }


# ---------------------------------------------------------------------------
# Builders / helpers
# ---------------------------------------------------------------------------


def make_skipped_check(
    rule_id: str,
    reason: str,
    *,
    threshold: Any = None,
    source_chain_step: str | None = None,
) -> RuleCheck:
    return RuleCheck(
        canonical_rule_id=rule_id,
        rule_group=RULE_TAXONOMY[rule_id],
        result="SKIPPED",
        computed=False,
        used_in_decision=False,
        value=None,
        threshold=threshold,
        evidence_ids=(),
        reason=reason,
        source_chain_step=source_chain_step,
    )


def make_not_reached_check(
    rule_id: str,
    reason: str,
    *,
    value: Any = None,
    threshold: Any = None,
    computed: bool = True,
) -> RuleCheck:
    """Used when an upstream rule blocked but the value WAS computed by
    backtest_engine before the engine bailed out (computed=true,
    used_in_decision=false per the design discussion)."""
    return RuleCheck(
        canonical_rule_id=rule_id,
        rule_group=RULE_TAXONOMY[rule_id],
        result="NOT_REACHED",
        computed=computed,
        used_in_decision=False,
        value=value,
        threshold=threshold,
        evidence_ids=(),
        reason=reason,
        source_chain_step=None,
    )


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "RULE_TAXONOMY",
    "ENTRY_SKIPPED_REASONS",
    "HORIZON_BARS_TABLE",
    "horizon_to_bars",
    "compute_strategy_config_hash",
    "compute_data_snapshot_hash",
    "get_commit_sha",
    "RunMetadata",
    "MarketSlice",
    "TechnicalSlice",
    "WaveformSlice",
    "HigherTimeframeSlice",
    "LongTermTrendSlice",
    "MacroContextSlice",
    "TechnicalConfluenceSlice",
    "RoyalRoadDecisionSlice",
    "RoyalRoadDecisionV2Slice",
    "FundamentalSlice",
    "ExecutionAssumptionSlice",
    "ExecutionTraceSlice",
    "RuleCheck",
    "DecisionSlice",
    "FutureOutcomeSlice",
    "BarDecisionTrace",
    "make_skipped_check",
    "make_not_reached_check",
]
