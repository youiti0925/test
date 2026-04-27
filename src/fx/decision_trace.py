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
            "fundamental": self.fundamental.to_dict(),
            "execution_assumption": self.execution_assumption.to_dict(),
            "execution_trace": self.execution_trace.to_dict(),
            "rule_checks": [rc.to_dict() for rc in self.rule_checks],
            "decision": self.decision.to_dict(),
            "future_outcome": (
                self.future_outcome.to_dict() if self.future_outcome else None
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
