"""technical_confluence_summary aggregator (PR-C2, observation-only).

`compute_technical_confluence_summary(traces)` produces the
`technical_confluence_summary` block embedded in `summary.json`.

It scans every BarDecisionTrace's optional `technical_confluence` slice
and counts label distribution, score histogram, reason frequencies, and
counts for each observable boolean field. The output is purely
diagnostic — never read by `decide_action`, `risk_gate`, or any
RuleCheck.

Schema versioning
-----------------
Schema is `technical_confluence_summary_v1`. Field renames or removals
require a version bump. Adding new aggregate keys is a non-breaking
change and stays at v1.

Fail-soft contract (matches `r_candidates_summary`)
---------------------------------------------------
The caller in `decision_trace_io._build_summary` wraps the call in a
try/except so an aggregator bug records `technical_confluence_error`
in the summary and leaves every other key intact.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


SCHEMA_VERSION: str = "technical_confluence_summary_v1"

# Score bins for `score_distribution`. Boundaries are inclusive on the
# lower edge: a score of -0.5 lands in "[-0.5, 0)".
_SCORE_BINS: tuple[tuple[str, float, float], ...] = (
    ("<-0.5",       float("-inf"), -0.5),
    ("[-0.5, 0)",   -0.5,           0.0),
    ("[0, 0.5]",     0.0,            0.5),
    (">0.5",         0.5,            float("inf")),
)

# How many top reasons to surface per direction. Reason space is small
# enough that the cap is practical, not theoretical.
_TOP_REASONS_LIMIT: int = 10


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _confluence_dict(trace: Any) -> dict | None:
    """Return the technical_confluence payload as a dict, or None.

    Accepts both BarDecisionTrace dataclass instances (attribute access)
    and dict representations (key access). Returns None when the trace
    has no confluence slice (live cmd_trade legacy export, etc.).
    """
    tc = _safe_get(trace, "technical_confluence")
    if tc is None:
        return None
    if isinstance(tc, dict):
        return tc
    # TechnicalConfluenceSlice instance
    to_dict = getattr(tc, "to_dict", None)
    return to_dict() if callable(to_dict) else None


def _bin_score(score: float) -> str:
    """Bin score into one of `<-0.5`, `[-0.5, 0)`, `[0, 0.5]`, `>0.5`.

    The middle-upper band is closed on both ends so a perfect 0.5 lands
    in `[0, 0.5]`, not `>0.5`.
    """
    if score < -0.5:
        return "<-0.5"
    if score < 0.0:
        return "[-0.5, 0)"
    if score <= 0.5:
        return "[0, 0.5]"
    return ">0.5"


def _initial_score_distribution() -> dict[str, int]:
    return {label: 0 for label, _lo, _hi in _SCORE_BINS}


def _initial_label_distribution() -> dict[str, int]:
    return {
        "STRONG_BUY_SETUP":  0,
        "WEAK_BUY_SETUP":    0,
        "STRONG_SELL_SETUP": 0,
        "WEAK_SELL_SETUP":   0,
        "NO_TRADE":          0,
        "AVOID_TRADE":       0,
        "UNKNOWN":           0,
    }


def _initial_candlestick_counts() -> dict[str, int]:
    return {
        "bullish_pinbar":      0,
        "bearish_pinbar":      0,
        "bullish_engulfing":   0,
        "bearish_engulfing":   0,
        "harami":              0,
        "strong_bull_body":    0,
        "strong_bear_body":    0,
        "rejection_wick":      0,
    }


def _initial_indicator_context_counts() -> dict[str, int]:
    return {
        "rsi_trend_danger_true":   0,
        "rsi_range_valid_true":    0,
        "bb_squeeze_true":         0,
        "bb_expansion_true":       0,
        "bb_band_walk_true":       0,
        "macd_momentum_up_true":   0,
        "macd_momentum_down_true": 0,
        "ma_trend_support_buy":    0,
        "ma_trend_support_sell":   0,
        "ma_trend_support_neutral": 0,
    }


def _initial_risk_plan_obs_summary() -> dict[str, Any]:
    return {
        "invalidation_clear_true_count": 0,
        "structure_stop_present_count":  0,
        "structure_stop_distance_atr_avg": None,
        "rr_structure_based_avg":          None,
    }


def compute_technical_confluence_summary(
    *,
    traces: Iterable[Any] = (),
) -> dict:
    """Aggregate technical_confluence_v1 fields across traces.

    Returns the full summary block even when the input is empty: counts
    are 0, averages are None, `top_*_reasons` are empty lists. This
    keeps `summary.json` shape-stable so downstream readers don't need
    to special-case the no-data case.
    """
    label_distribution = _initial_label_distribution()
    score_distribution = _initial_score_distribution()
    candlestick_counts = _initial_candlestick_counts()
    indicator_counts = _initial_indicator_context_counts()
    risk_plan_summary = _initial_risk_plan_obs_summary()

    bullish_reason_counter: Counter[str] = Counter()
    bearish_reason_counter: Counter[str] = Counter()
    avoid_reason_counter: Counter[str] = Counter()

    near_support_count = 0
    near_resistance_count = 0
    n_with_confluence = 0
    structure_stop_distances: list[float] = []
    rr_structure_values: list[float] = []

    for tr in traces:
        d = _confluence_dict(tr)
        if d is None:
            continue
        n_with_confluence += 1

        final = d.get("final_confluence") or {}
        label = final.get("label", "UNKNOWN") or "UNKNOWN"
        if label not in label_distribution:
            label_distribution[label] = 0
        label_distribution[label] += 1

        score = final.get("score")
        if isinstance(score, (int, float)):
            score_distribution[_bin_score(float(score))] += 1

        for reason in final.get("bullish_reasons", []) or []:
            bullish_reason_counter[str(reason)] += 1
        for reason in final.get("bearish_reasons", []) or []:
            bearish_reason_counter[str(reason)] += 1
        for reason in final.get("avoid_reasons", []) or []:
            avoid_reason_counter[str(reason)] += 1

        sr = d.get("support_resistance") or {}
        if sr.get("near_support"):
            near_support_count += 1
        if sr.get("near_resistance"):
            near_resistance_count += 1

        cs = d.get("candlestick_signal") or {}
        for k in candlestick_counts:
            if cs.get(k):
                candlestick_counts[k] += 1

        ic = d.get("indicator_context") or {}
        if ic.get("rsi_trend_danger") is True:
            indicator_counts["rsi_trend_danger_true"] += 1
        if ic.get("rsi_range_valid") is True:
            indicator_counts["rsi_range_valid_true"] += 1
        if ic.get("bb_squeeze") is True:
            indicator_counts["bb_squeeze_true"] += 1
        if ic.get("bb_expansion") is True:
            indicator_counts["bb_expansion_true"] += 1
        if ic.get("bb_band_walk") is True:
            indicator_counts["bb_band_walk_true"] += 1
        if ic.get("macd_momentum_up") is True:
            indicator_counts["macd_momentum_up_true"] += 1
        if ic.get("macd_momentum_down") is True:
            indicator_counts["macd_momentum_down_true"] += 1
        ma_state = ic.get("ma_trend_support")
        if ma_state == "BUY":
            indicator_counts["ma_trend_support_buy"] += 1
        elif ma_state == "SELL":
            indicator_counts["ma_trend_support_sell"] += 1
        elif ma_state == "NEUTRAL":
            indicator_counts["ma_trend_support_neutral"] += 1

        rp = d.get("risk_plan_obs") or {}
        if rp.get("invalidation_clear") is True:
            risk_plan_summary["invalidation_clear_true_count"] += 1
        if rp.get("structure_stop_price") is not None:
            risk_plan_summary["structure_stop_present_count"] += 1
        ssd = rp.get("structure_stop_distance_atr")
        if isinstance(ssd, (int, float)):
            structure_stop_distances.append(float(ssd))
        rr_s = rp.get("rr_structure_based")
        if isinstance(rr_s, (int, float)):
            rr_structure_values.append(float(rr_s))

    if structure_stop_distances:
        risk_plan_summary["structure_stop_distance_atr_avg"] = (
            sum(structure_stop_distances) / len(structure_stop_distances)
        )
    if rr_structure_values:
        risk_plan_summary["rr_structure_based_avg"] = (
            sum(rr_structure_values) / len(rr_structure_values)
        )

    return {
        "schema_version":               SCHEMA_VERSION,
        "n_traces_with_confluence":     n_with_confluence,
        "label_distribution":           label_distribution,
        "score_distribution":           score_distribution,
        "top_bullish_reasons":          _top_reasons(bullish_reason_counter),
        "top_bearish_reasons":          _top_reasons(bearish_reason_counter),
        "top_avoid_reasons":            _top_reasons(avoid_reason_counter),
        "near_support_count":           near_support_count,
        "near_resistance_count":        near_resistance_count,
        "candlestick_signal_counts":    candlestick_counts,
        "indicator_context_counts":     indicator_counts,
        "risk_plan_obs_summary":        risk_plan_summary,
    }


def _top_reasons(counter: Counter[str]) -> list[dict]:
    return [
        {"reason": r, "count": c}
        for r, c in counter.most_common(_TOP_REASONS_LIMIT)
    ]


__all__ = [
    "SCHEMA_VERSION",
    "compute_technical_confluence_summary",
]
