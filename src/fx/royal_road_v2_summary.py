"""royal_road_v2 summary aggregator (PR-V2 follow-up).

`compute_royal_road_v2_summary(traces)` produces the
`royal_road_v2_summary` block embedded in `summary.json` when the
royal_road_decision_v2 profile was active. Same fail-soft pattern as
`r_candidates_summary` and `technical_confluence_summary`: any
exception is caught upstream and recorded as
`royal_road_v2_summary_error` without breaking the rest of the file.

Output schema_version = "royal_road_v2_summary_v1". Adding new keys
is non-breaking; renames / removals require a version bump.

This module never reads decisions or runtime parameters; it only
aggregates already-emitted v2 trace fields.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


SCHEMA_VERSION: str = "royal_road_v2_summary_v1"
_TOP_REASONS_LIMIT: int = 15
_TOP_CAUTIONS_LIMIT: int = 15

# Distance / RR distribution buckets — heuristic, observation-only.
_DISTANCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("missing", float("-inf"), 0.0),
    ("[0,0.5)", 0.0, 0.5),
    ("[0.5,1.5)", 0.5, 1.5),
    ("[1.5,3.0)", 1.5, 3.0),
    (">=3.0", 3.0, float("inf")),
)
_RR_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("<1.0", float("-inf"), 1.0),
    ("[1.0,1.5)", 1.0, 1.5),
    ("[1.5,2.5)", 1.5, 2.5),
    (">=2.5", 2.5, float("inf")),
)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _v2_dict(trace: Any) -> dict | None:
    sl = _safe_get(trace, "royal_road_decision_v2")
    if sl is None:
        return None
    if isinstance(sl, dict):
        return sl
    to_dict = getattr(sl, "to_dict", None)
    return to_dict() if callable(to_dict) else None


def _bucket_distance(d: Any) -> str:
    if not isinstance(d, (int, float)):
        return "missing"
    v = float(d)
    if v < 0:
        return "missing"
    for label, lo, hi in _DISTANCE_BUCKETS:
        if lo <= v < hi:
            return label
    return ">=3.0"


def _bucket_rr(d: Any) -> str:
    if not isinstance(d, (int, float)):
        return "missing"
    v = float(d)
    for label, lo, hi in _RR_BUCKETS:
        if lo <= v < hi:
            return label
    return ">=2.5"


def _bucket_strength(s: Any) -> str:
    if not isinstance(s, (int, float)):
        return "missing"
    v = float(s)
    if v < 1.0:
        return "<1.0"
    if v < 2.0:
        return "[1.0,2.0)"
    if v < 4.0:
        return "[2.0,4.0)"
    return ">=4.0"


def compute_royal_road_v2_summary(*, traces: Iterable[Any] = ()) -> dict:
    """Aggregate v2 trace fields. Always returns the full schema even
    when no traces carry v2 data (counts = 0 / averages = None)."""

    n = 0
    action_dist = Counter()
    label_dist = Counter()      # final_confluence.label
    mode_dist = Counter()
    block_reasons = Counter()
    cautions = Counter()
    bullish_axes_total = Counter()
    bearish_axes_total = Counter()
    sr_strength_dist = Counter()
    trendline_signal_dist = Counter(
        bullish=0, bearish=0, neither=0
    )
    chart_pattern_dist = Counter()
    lower_tf_dist = Counter(
        unavailable=0, no_trigger=0, bullish_trigger=0,
        bearish_trigger=0, both=0,
    )
    macro_alignment_dist = Counter()    # currency_bias label
    macro_event_tone_dist = Counter()
    stop_mode_dist = Counter()
    selected_stop_distance_dist = Counter()
    rr_selected_dist = Counter()

    for tr in traces:
        d = _v2_dict(tr)
        if d is None:
            continue
        n += 1
        action_dist[d.get("action") or "UNKNOWN"] += 1
        mode_dist[d.get("mode") or "UNKNOWN"] += 1

        # final_confluence label is on the v2 trace's source slice
        # (technical_confluence). v2 advisory does not duplicate it,
        # so dig from the v1 trace if present; fall back to UNKNOWN.
        tc = _safe_get(tr, "technical_confluence")
        if tc is None:
            label_dist["UNKNOWN"] += 1
        else:
            tc_dict = (
                tc if isinstance(tc, dict)
                else (tc.to_dict() if hasattr(tc, "to_dict") else {})
            )
            label = (
                (tc_dict.get("final_confluence") or {}).get("label")
                or "UNKNOWN"
            )
            label_dist[label] += 1

        for r in d.get("block_reasons") or []:
            block_reasons[str(r)] += 1
        for c in d.get("cautions") or []:
            cautions[str(c)] += 1

        axes = d.get("evidence_axes") or {}
        for k, v in (axes.get("bullish") or {}).items():
            if v:
                bullish_axes_total[k] += 1
        for k, v in (axes.get("bearish") or {}).items():
            if v:
                bearish_axes_total[k] += 1

        sr = d.get("support_resistance_v2") or {}
        nearest_sup = sr.get("nearest_support") or {}
        nearest_res = sr.get("nearest_resistance") or {}
        for lvl in (nearest_sup, nearest_res):
            if not lvl:
                continue
            sr_strength_dist[_bucket_strength(lvl.get("strength_score"))] += 1

        tl = d.get("trendline_context") or {}
        if tl.get("bullish_signal"):
            trendline_signal_dist["bullish"] += 1
        elif tl.get("bearish_signal"):
            trendline_signal_dist["bearish"] += 1
        else:
            trendline_signal_dist["neither"] += 1

        cp = d.get("chart_pattern_v2") or {}
        for kind in ("head_and_shoulders", "inverse_head_and_shoulders",
                     "flag", "wedge", "triangle"):
            entry = cp.get(kind)
            if entry:
                chart_pattern_dist[entry.get("kind") or kind] += 1

        ltf = d.get("lower_tf_trigger") or {}
        if not ltf.get("available"):
            lower_tf_dist["unavailable"] += 1
        else:
            bull = bool(ltf.get("bullish_trigger"))
            bear = bool(ltf.get("bearish_trigger"))
            if bull and bear:
                lower_tf_dist["both"] += 1
            elif bull:
                lower_tf_dist["bullish_trigger"] += 1
            elif bear:
                lower_tf_dist["bearish_trigger"] += 1
            else:
                lower_tf_dist["no_trigger"] += 1

        macro = d.get("macro_alignment") or {}
        macro_alignment_dist[macro.get("currency_bias") or "UNKNOWN"] += 1
        macro_event_tone_dist[macro.get("event_tone") or "UNKNOWN"] += 1

        stop_plan = d.get("structure_stop_plan") or {}
        stop_mode_dist[stop_plan.get("chosen_mode") or "missing"] += 1
        sp_distance = stop_plan.get("structure_stop_distance_atr")
        selected_stop_distance_dist[_bucket_distance(sp_distance)] += 1
        rr_selected_dist[_bucket_rr(stop_plan.get("rr_realized"))] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "n_traces_with_v2": n,
        "action_distribution": dict(action_dist),
        "label_distribution": dict(label_dist),
        "mode_distribution": dict(mode_dist),
        "block_reasons_top": [
            {"reason": r, "count": c}
            for r, c in block_reasons.most_common(_TOP_REASONS_LIMIT)
        ],
        "cautions_top": [
            {"caution": c, "count": n_}
            for c, n_ in cautions.most_common(_TOP_CAUTIONS_LIMIT)
        ],
        "evidence_axes_counts": {
            "bullish": dict(bullish_axes_total),
            "bearish": dict(bearish_axes_total),
        },
        "support_resistance_strength_distribution": dict(sr_strength_dist),
        "trendline_signal_distribution": dict(trendline_signal_dist),
        "chart_pattern_distribution": dict(chart_pattern_dist),
        "lower_tf_trigger_distribution": dict(lower_tf_dist),
        "macro_alignment_distribution": dict(macro_alignment_dist),
        "macro_event_tone_distribution": dict(macro_event_tone_dist),
        "stop_mode_distribution": dict(stop_mode_dist),
        "selected_stop_distance_atr_distribution":
            dict(selected_stop_distance_dist),
        "rr_selected_distribution": dict(rr_selected_dist),
    }


__all__ = [
    "SCHEMA_VERSION",
    "compute_royal_road_v2_summary",
]
