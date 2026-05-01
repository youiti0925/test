"""R-candidate hypothesis-test helpers (PR #20, observation-only).

`compute_r_candidates_summary(traces, trades)` produces the
`r_candidates_summary` block embedded in `summary.json`. The 11
R-questions encode the hypotheses surfaced during the PR #19
verification — see `r_candidates_v1` schema below.

Versioning
----------
Schema is `r_candidates_v1`. R1–R11 cell definitions are FROZEN at this
version: changing the cell definition of an existing R-number requires
bumping to `r_candidates_v2`. Adding new R-numbers (R12, R13, …)
without changing R1–R11 is a non-breaking addition and stays at v1.

Design constraint — observation-only
------------------------------------
Nothing in this module is read by `decide_action` or `risk_gate`. The
output is purely diagnostic for offline verification. `sufficient_n`
(default ≥ 30) is recorded per cell so callers can mechanically
distinguish "real signal" from "n too small to act".
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence


SCHEMA_VERSION: str = "r_candidates_v1"
N_THRESHOLD_FOR_SUFFICIENT: int = 30

# Observation-only thresholds (NOT in PARAMETER_BASELINE_V1 — those drive
# trading decisions, these only label cells in the diagnostic summary).
_DXY_TREND_BUCKET_THRESHOLDS_HUMAN = {
    "STRONG_DOWN": "< -2.0%",
    "DOWN":        "-2.0..-0.5%",
    "FLAT":        "-0.5..0.5%",
    "UP":          "0.5..2.0%",
    "STRONG_UP":   ">= 2.0%",
}
_DXY_ZSCORE_BUCKET_THRESHOLDS_HUMAN = {
    "EXTREME_LOW":  "< -2",
    "LOW":          "-2..-1",
    "NEUTRAL":      "-1..1",
    "HIGH":         "1..2",
    "EXTREME_HIGH": ">= 2",
}
_SMA_50_200_DEAD_BAND_PCT = 0.5
_SMA_50_200_STATE_POLICY_VERSION = "sma50_200_state_v1"
_SHADOW_OUTCOME_THRESHOLD_PCT = 0.3
_SHADOW_OUTCOME_POLICY_VERSION = "shadow_outcome_v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    """Read `key` off a dict-like, falling back to dataclass attr."""
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


def _trade_is_win(t: Any) -> bool:
    return float(_safe_get(t, "return_pct", 0.0) or 0.0) > 0.0


def _cell(rows: Sequence[Any], *, label: str) -> dict:
    """Aggregate W/L/WR/sum_return/PF for a list of trade rows."""
    n = len(rows)
    if n == 0:
        return {
            "label": label, "n": 0, "wins": 0, "losses": 0,
            "win_rate": None, "sum_return_pct": 0.0, "profit_factor": None,
            "sufficient_n": False,
        }
    wins = sum(1 for r in rows if _trade_is_win(r))
    losses = n - wins
    sum_ret = sum(float(_safe_get(r, "return_pct", 0.0) or 0.0) for r in rows)
    sum_w = sum(
        float(_safe_get(r, "return_pct", 0.0) or 0.0)
        for r in rows if _trade_is_win(r)
    )
    sum_l = sum(
        -float(_safe_get(r, "return_pct", 0.0) or 0.0)
        for r in rows if not _trade_is_win(r)
    )
    pf = (sum_w / sum_l) if sum_l > 0 else None
    return {
        "label": label,
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / n,
        "sum_return_pct": sum_ret,
        "profit_factor": pf,
        "sufficient_n": n >= N_THRESHOLD_FOR_SUFFICIENT,
    }


def _trace_of(trade: Any) -> Any:
    """Trade's entry trace if attached, else None."""
    return _safe_get(trade, "entry_trace") or _safe_get(trade, "trace")


def _macro(trade: Any) -> dict:
    tr = _trace_of(trade)
    return _safe_get(tr, "macro_context", {}) or {}


def _wb(trade: Any) -> dict:
    tr = _trace_of(trade)
    wf = _safe_get(tr, "waveform", {}) or {}
    return _safe_get(wf, "waveform_bias", {}) or {}


def _lt(trade: Any) -> dict:
    tr = _trace_of(trade)
    return _safe_get(tr, "long_term_trend", {}) or {}


def _final_action(trade: Any) -> str | None:
    tr = _trace_of(trade)
    dec = _safe_get(tr, "decision", {}) or {}
    return _safe_get(dec, "final_action")


def _session_of_ts(ts: str) -> str:
    """Hour-bucket session label for an ISO-format timestamp.

    Buckets:
      ASIA       : 0..6 UTC
      LONDON_PRE : 6..13 UTC
      LONDON_NY  : 13..17 UTC (overlap)
      NY         : 17..24 UTC
    """
    try:
        h = int(ts[11:13])
    except (TypeError, ValueError, IndexError):
        return "UNKNOWN"
    if 0 <= h < 6:   return "ASIA"
    if 6 <= h < 13:  return "LONDON_PRE"
    if 13 <= h < 17: return "LONDON_NY"
    return "NY"


def _usd_direction(symbol: str | None, side: str | None) -> str:
    """LONG_USD / SHORT_USD / UNKNOWN per Q9 spec.

    USDJPY=X     → BUY = LONG_USD,   SELL = SHORT_USD
    EUR/GBP/AUD  → BUY = SHORT_USD, SELL = LONG_USD
    """
    if not symbol or side not in ("BUY", "SELL"):
        return "UNKNOWN"
    if symbol.startswith("USD"):
        return "LONG_USD" if side == "BUY" else "SHORT_USD"
    return "SHORT_USD" if side == "BUY" else "LONG_USD"


# ---------------------------------------------------------------------------
# Individual R-functions (R1–R11)
# ---------------------------------------------------------------------------


def _r1_vix_extreme_outcome(trades: Sequence[Any]) -> dict:
    high = [t for t in trades
            if (v := _macro(t).get("vix")) is not None and v >= 25]
    low  = [t for t in trades
            if (v := _macro(t).get("vix")) is not None and v < 25]
    return {
        "hypothesis": "VIX>=25 trades are weak (PR #19 candidate R1)",
        "cells": [_cell(high, label="VIX>=25"),
                  _cell(low,  label="VIX<25")],
    }


def _r2_vix_20_25_waveform_hold_outcome(trades: Sequence[Any]) -> dict:
    in_band = [t for t in trades
               if (v := _macro(t).get("vix")) is not None and 20 <= v < 25]
    wb_hold = [t for t in in_band if _wb(t).get("action") == "HOLD"]
    wb_act = [t for t in in_band if _wb(t).get("action") in ("BUY", "SELL")]
    return {
        "hypothesis": "VIX 20-25 with waveform_bias=HOLD is strong (R2)",
        "cells": [_cell(wb_hold, label="VIX 20-25 × wf=HOLD"),
                  _cell(wb_act,  label="VIX 20-25 × wf!=HOLD")],
    }


def _r3_waveform_high_conf_outcome(trades: Sequence[Any]) -> dict:
    high_conf = [t for t in trades
                 if (c := _wb(t).get("confidence")) is not None and c >= 0.5]
    concur = [t for t in high_conf
              if _wb(t).get("action") == _final_action(t)]
    disagree = [t for t in high_conf
                if _wb(t).get("action") != _final_action(t)]
    return {
        "hypothesis": "High-confidence waveform trades are weak (R3)",
        "cells": [
            _cell(high_conf, label="conf>=0.5 (any direction)"),
            _cell(concur,    label="conf>=0.5 × wf==final"),
            _cell(disagree,  label="conf>=0.5 × wf!=final"),
        ],
    }


def _r4_top_similarity_extreme_outcome(trades: Sequence[Any]) -> dict:
    high = [t for t in trades
            if (s := _wb(t).get("top_similarity")) is not None and s >= 0.85]
    low  = [t for t in trades
            if (s := _wb(t).get("top_similarity")) is not None and s < 0.85]
    return {
        "hypothesis": "top_similarity>=0.85 trades are weak (R4 — strongest PR #19 candidate)",
        "cells": [_cell(high, label="top_sim>=0.85"),
                  _cell(low,  label="top_sim<0.85")],
    }


def _r5_dxy_usd_exposure_outcome(trades: Sequence[Any]) -> dict:
    """DXY trend bucket × USD direction × outcome.

    PR #20 adds DXY trend buckets to the trace; the cells below pivot
    the trade pool by `dxy_trend_5d_bucket × usd_direction`. We also
    include the 20d variant for robustness.
    """
    cells_5d = []
    cells_20d = []
    for label_key, bucket_field in (
        ("5d", "dxy_trend_5d_bucket"),
        ("20d", "dxy_trend_20d_bucket"),
    ):
        for bucket in ("STRONG_DOWN", "DOWN", "FLAT", "UP", "STRONG_UP"):
            for usd in ("LONG_USD", "SHORT_USD"):
                rows = [
                    t for t in trades
                    if _macro(t).get(bucket_field) == bucket
                    and _usd_direction(_safe_get(t, "sym"),
                                       _safe_get(t, "side")) == usd
                ]
                cell = _cell(rows, label=f"DXY_{bucket}_{usd}")
                if label_key == "5d":
                    cells_5d.append(cell)
                else:
                    cells_20d.append(cell)
    return {
        "hypothesis": "DXY trend × USD exposure outcome (R5 — PR #19 was unevaluable; PR #20 added DXY buckets)",
        "policy_version": "r5_v1",
        "cells_dxy_5d": cells_5d,
        "cells_dxy_20d": cells_20d,
    }


def _r6_session_outcome(trades: Sequence[Any]) -> dict:
    g: dict[str, list] = defaultdict(list)
    for t in trades:
        g[_session_of_ts(_safe_get(t, "entry_ts", "") or "")].append(t)
    return {
        "hypothesis": "NY session trades are weak (R6)",
        "cells": [_cell(g[k], label=f"session={k}")
                  for k in ("ASIA", "LONDON_PRE", "LONDON_NY", "NY", "UNKNOWN")
                  if g[k]],
    }


def _r7_waveform_hold_vs_concur_outcome(trades: Sequence[Any]) -> dict:
    hold = [t for t in trades if _wb(t).get("action") == "HOLD"]
    concur = [t for t in trades
              if _wb(t).get("action") in ("BUY", "SELL")
              and _wb(t).get("action") == _final_action(t)]
    disagree = [t for t in trades
                if _wb(t).get("action") in ("BUY", "SELL")
                and _wb(t).get("action") != _final_action(t)]
    return {
        "hypothesis": "wf=HOLD vs wf=concur (PR #19 found no concur advantage; R7)",
        "cells": [
            _cell(hold,     label="wf=HOLD"),
            _cell(concur,   label="wf concur (action == final)"),
            _cell(disagree, label="wf disagree"),
        ],
    }


def _r8_exit_holding_distribution(trades: Sequence[Any]) -> dict:
    by_exit: dict[str, list] = defaultdict(list)
    for t in trades:
        by_exit[_safe_get(t, "exit_reason") or "unknown"].append(
            _safe_get(t, "bars_held") or 0
        )
    out_cells = []
    for k, vs in by_exit.items():
        n = len(vs)
        avg = (sum(vs) / n) if n else None
        mn = min(vs) if vs else None
        mx = max(vs) if vs else None
        out_cells.append({
            "label": f"exit={k}",
            "n": n,
            "avg_bars_held": avg,
            "min_bars_held": mn,
            "max_bars_held": mx,
            "sufficient_n": n >= N_THRESHOLD_FOR_SUFFICIENT,
        })
    return {
        "hypothesis": "Stop and TP avg holding differ (R8)",
        "cells": out_cells,
    }


def _close_vs_sma200_zone(pct: float | None) -> str:
    if pct is None:
        return "NA"
    if pct < -1.0:  return "below_sma200"
    if pct < 1.0:   return "near_sma200"
    return "above_sma200"


def _r9_sma200_above_sell_outcome(trades: Sequence[Any]) -> dict:
    rows = [t for t in trades
            if _close_vs_sma200_zone(_lt(t).get("close_vs_sma_200d_pct")) == "above_sma200"
            and _safe_get(t, "side") == "SELL"]
    return {
        "hypothesis": "above_SMA200 × SELL counter-trend works (R9)",
        "cells": [_cell(rows, label="above_sma200 × SELL")],
    }


def _r10_near_sma200_buy_outcome(trades: Sequence[Any]) -> dict:
    rows = [t for t in trades
            if _close_vs_sma200_zone(_lt(t).get("close_vs_sma_200d_pct")) == "near_sma200"
            and _safe_get(t, "side") == "BUY"]
    return {
        "hypothesis": "near_SMA200 × BUY is weak (R10)",
        "cells": [_cell(rows, label="near_sma200 × BUY")],
    }


def _r11_monthly_return_waveform_hold_outcome(trades: Sequence[Any]) -> dict:
    rows = [
        t for t in trades
        if (mr := _lt(t).get("monthly_return_pct")) is not None
        and -2.0 < mr < 0.0
        and _wb(t).get("action") == "HOLD"
    ]
    return {
        "hypothesis": "monthly_return -2..0% × wf=HOLD is strong (R11)",
        "cells": [_cell(rows, label="monthly_return -2..0% × wf=HOLD")],
    }


# ---------------------------------------------------------------------------
# R12-R16: technical_confluence_v1 hypothesis cells (PR-C2)
#
# These read trade['entry_trace']['technical_confluence']. When the
# slice is missing (legacy traces, live cmd_trade) the trade is simply
# excluded from the relevant cell.
# ---------------------------------------------------------------------------


_CONFLUENCE_LABELS: tuple[str, ...] = (
    "STRONG_BUY_SETUP",
    "WEAK_BUY_SETUP",
    "STRONG_SELL_SETUP",
    "WEAK_SELL_SETUP",
    "NO_TRADE",
    "AVOID_TRADE",
    "UNKNOWN",
)

_BULLISH_CANDLE_FLAGS: tuple[str, ...] = (
    "bullish_pinbar", "bullish_engulfing", "strong_bull_body",
)
_BEARISH_CANDLE_FLAGS: tuple[str, ...] = (
    "bearish_pinbar", "bearish_engulfing", "strong_bear_body",
)


def _tc(trade: Any) -> dict:
    """Trade's technical_confluence dict (or empty dict if absent)."""
    tr = _trace_of(trade)
    return _safe_get(tr, "technical_confluence", {}) or {}


def _tc_label(trade: Any) -> str:
    return (_tc(trade).get("final_confluence") or {}).get("label") or "UNKNOWN"


def _tc_sr(trade: Any) -> dict:
    return _tc(trade).get("support_resistance") or {}


def _tc_candle(trade: Any) -> dict:
    return _tc(trade).get("candlestick_signal") or {}


def _tc_indicator(trade: Any) -> dict:
    return _tc(trade).get("indicator_context") or {}


def _tc_risk(trade: Any) -> dict:
    return _tc(trade).get("risk_plan_obs") or {}


def _has_any_flag(d: dict, flags: tuple[str, ...]) -> bool:
    return any(bool(d.get(f)) for f in flags)


def _structure_stop_bucket(distance_atr: Any) -> str:
    if not isinstance(distance_atr, (int, float)):
        return "structure_stop_missing"
    d = float(distance_atr)
    if d < 0.5:                 return "structure_stop_lt_0_5_atr"
    if d < 1.5:                 return "structure_stop_0_5_to_1_5_atr"
    if d < 3.0:                 return "structure_stop_1_5_to_3_0_atr"
    return "structure_stop_gt_3_0_atr"


def _r12_confluence_label_outcome(trades: Sequence[Any]) -> dict:
    """Bucket trades by `final_confluence.label` and report W/L/WR/PF."""
    cells = []
    for label in _CONFLUENCE_LABELS:
        rows = [t for t in trades if _tc_label(t) == label]
        cells.append(_cell(rows, label=f"label={label}"))
    return {
        "hypothesis": (
            "technical_confluence label separates winners from losers (R12)"
        ),
        "policy_version": "r12_v1",
        "cells": cells,
    }


def _r13_sr_proximity_outcome(trades: Sequence[Any]) -> dict:
    """Trades pivoted by support/resistance proximity at entry."""
    near_support = [t for t in trades if _tc_sr(t).get("near_support")]
    near_resistance = [t for t in trades if _tc_sr(t).get("near_resistance")]
    not_near_any = [
        t for t in trades
        if not _tc_sr(t).get("near_support")
        and not _tc_sr(t).get("near_resistance")
    ]
    return {
        "hypothesis": (
            "near_support / near_resistance entries differ in WR vs "
            "level-agnostic entries (R13)"
        ),
        "policy_version": "r13_v1",
        "cells": [
            _cell(near_support,    label="near_support"),
            _cell(near_resistance, label="near_resistance"),
            _cell(not_near_any,    label="not_near_any_level"),
        ],
    }


def _r14_candlestick_confirmation_outcome(trades: Sequence[Any]) -> dict:
    """Compare entries with vs without a candlestick confirmation."""
    bullish = [
        t for t in trades
        if _has_any_flag(_tc_candle(t), _BULLISH_CANDLE_FLAGS)
    ]
    bearish = [
        t for t in trades
        if _has_any_flag(_tc_candle(t), _BEARISH_CANDLE_FLAGS)
    ]
    any_conf = [
        t for t in trades
        if (
            _has_any_flag(_tc_candle(t), _BULLISH_CANDLE_FLAGS)
            or _has_any_flag(_tc_candle(t), _BEARISH_CANDLE_FLAGS)
        )
    ]
    none_conf = [
        t for t in trades
        if not _has_any_flag(_tc_candle(t), _BULLISH_CANDLE_FLAGS)
        and not _has_any_flag(_tc_candle(t), _BEARISH_CANDLE_FLAGS)
    ]
    return {
        "hypothesis": (
            "lower-TF candlestick confirmation improves WR (R14)"
        ),
        "policy_version": "r14_v1",
        "cells": [
            _cell(bullish,   label="bullish_candle_confirmation"),
            _cell(bearish,   label="bearish_candle_confirmation"),
            _cell(any_conf,  label="any_candlestick_confirmation"),
            _cell(none_conf, label="no_candlestick_confirmation"),
        ],
    }


def _r15_rsi_regime_danger_outcome(trades: Sequence[Any]) -> dict:
    """Pivot trades by RSI regime danger flag."""
    danger_true = [
        t for t in trades if _tc_indicator(t).get("rsi_trend_danger") is True
    ]
    danger_false = [
        t for t in trades if _tc_indicator(t).get("rsi_trend_danger") is False
    ]
    range_valid = [
        t for t in trades if _tc_indicator(t).get("rsi_range_valid") is True
    ]
    return {
        "hypothesis": (
            "rsi_trend_danger=True (counter-trend RSI extreme during a "
            "directional regime) entries underperform (R15)"
        ),
        "policy_version": "r15_v1",
        "cells": [
            _cell(danger_true,  label="rsi_trend_danger_true"),
            _cell(danger_false, label="rsi_trend_danger_false"),
            _cell(range_valid,  label="rsi_range_valid_true"),
        ],
    }


def _r16_structure_stop_distance_outcome(trades: Sequence[Any]) -> dict:
    """Bucket trades by structure_stop_distance_atr."""
    buckets: dict[str, list] = defaultdict(list)
    for t in trades:
        b = _structure_stop_bucket(
            _tc_risk(t).get("structure_stop_distance_atr")
        )
        buckets[b].append(t)
    bucket_order = (
        "structure_stop_missing",
        "structure_stop_lt_0_5_atr",
        "structure_stop_0_5_to_1_5_atr",
        "structure_stop_1_5_to_3_0_atr",
        "structure_stop_gt_3_0_atr",
    )
    return {
        "hypothesis": (
            "structure-stop distance band predicts trade quality (R16)"
        ),
        "policy_version": "r16_v1",
        "cells": [
            _cell(buckets[b], label=b) for b in bucket_order
        ],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_r_candidates_summary(
    *,
    traces: Iterable[Any] = (),
    trades: Iterable[Any] = (),
    test_period: dict | None = None,
) -> dict:
    """Compute the r_candidates_v1 summary block.

    Parameters
    ----------
    traces:
        Bar-level decision traces (used for n_traces_total + future
        cells that need bar-level statistics — currently only the
        denominator).
    trades:
        Reconstructed trades. Each trade row should expose `entry_ts`,
        `exit_ts`, `side`, `return_pct`, `bars_held`, `exit_reason`,
        `sym`, plus `entry_trace` (or `trace`) carrying the
        BarDecisionTrace dict at entry.
    test_period:
        Optional `{"start": "...", "end": "..."}` echo for the report.

    Returns
    -------
    A dict matching the r_candidates_v1 schema. Always returns a dict
    with all 11 R-cells present; cells with n=0 are kept (they are
    deliberately visible so a reader sees "0 trades fit this band").
    """
    traces = list(traces)
    trades = list(trades)
    return {
        "schema_version": SCHEMA_VERSION,
        "test_period": dict(test_period) if test_period else None,
        "n_trades_total": len(trades),
        "n_traces_total": len(traces),
        "n_threshold_for_sufficient": N_THRESHOLD_FOR_SUFFICIENT,
        "thresholds": {
            "dxy_trend_bucket": dict(_DXY_TREND_BUCKET_THRESHOLDS_HUMAN),
            "dxy_zscore_bucket": dict(_DXY_ZSCORE_BUCKET_THRESHOLDS_HUMAN),
            "sma_50_200_dead_band_pct": _SMA_50_200_DEAD_BAND_PCT,
            "sma_50_200_state_policy_version":
                _SMA_50_200_STATE_POLICY_VERSION,
            "shadow_outcome_threshold_pct": _SHADOW_OUTCOME_THRESHOLD_PCT,
            "shadow_outcome_policy_version":
                _SHADOW_OUTCOME_POLICY_VERSION,
        },
        "r1_vix_extreme_outcome":              _r1_vix_extreme_outcome(trades),
        "r2_vix_20_25_waveform_hold_outcome":  _r2_vix_20_25_waveform_hold_outcome(trades),
        "r3_waveform_high_conf_outcome":       _r3_waveform_high_conf_outcome(trades),
        "r4_top_similarity_extreme_outcome":   _r4_top_similarity_extreme_outcome(trades),
        "r5_dxy_usd_exposure_outcome":         _r5_dxy_usd_exposure_outcome(trades),
        "r6_session_outcome":                  _r6_session_outcome(trades),
        "r7_waveform_hold_vs_concur_outcome":  _r7_waveform_hold_vs_concur_outcome(trades),
        "r8_exit_holding_distribution":        _r8_exit_holding_distribution(trades),
        "r9_sma200_above_sell_outcome":        _r9_sma200_above_sell_outcome(trades),
        "r10_near_sma200_buy_outcome":         _r10_near_sma200_buy_outcome(trades),
        "r11_monthly_return_waveform_hold_outcome":
            _r11_monthly_return_waveform_hold_outcome(trades),
        # PR-C2: technical_confluence_v1 hypothesis cells.
        "r12_confluence_label_outcome":
            _r12_confluence_label_outcome(trades),
        "r13_sr_proximity_outcome":
            _r13_sr_proximity_outcome(trades),
        "r14_candlestick_confirmation_outcome":
            _r14_candlestick_confirmation_outcome(trades),
        "r15_rsi_regime_danger_outcome":
            _r15_rsi_regime_danger_outcome(trades),
        "r16_structure_stop_distance_outcome":
            _r16_structure_stop_distance_outcome(trades),
    }
