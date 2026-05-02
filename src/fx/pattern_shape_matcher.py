"""Shape-based pattern matcher (observation-only).

Compares a `WaveSkeleton` (from `waveform_encoder`) against the idealised
templates in `pattern_templates` using a DTW (Dynamic Time Warping)
distance, then categorises the result into a `ShapeMatch`.

This module is observation-only. It does NOT participate in
`royal_road_decision_v2`'s final BUY/SELL/HOLD logic. The output is
consumed by the audit / visual_audit layer to give a human reviewer
a "the system thinks this LOOKS like a double bottom" report.

DTW vs simple correlation
-------------------------
DTW handles the case where two shapes are similar but stretched
along the time axis differently (e.g. a wide double-bottom vs a
narrow one). For a 64-point polyline the cost matrix is small and
DTW is fast; we still also compute Pearson correlation as a sanity
diagnostic exposed via `correlation_score`.

Score bands (heuristic, NOT validated)
--------------------------------------
shape_score ≥ 0.82  → strong candidate
shape_score ≥ 0.70  → candidate
shape_score <  0.70 → not_matched

These are first-pass thresholds. They are NOT validated against
multi-symbol multi-period data; downstream code surfaces this status
through `weakness_reasons` and `audit_notes`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .pattern_templates import (
    PatternTemplate,
    all_templates,
)
from .waveform_encoder import WaveSkeleton


SCHEMA_VERSION: Final[str] = "shape_match_v1"

# Heuristic thresholds (NOT validated)
STRONG_SHAPE_SCORE: Final[float] = 0.82
CANDIDATE_SHAPE_SCORE: Final[float] = 0.70


@dataclass(frozen=True)
class ShapeMatch:
    """Result of comparing a WaveSkeleton to one PatternTemplate.

    `status` is one of:
      "forming"          — shape matches but neckline / breakout
                           confirmation is not present yet
      "neckline_broken"  — latest skeleton extreme has crossed the
                           pattern's neckline in the expected direction
      "retested"         — after the break, the skeleton has come
                           back to the neckline at least once
      "invalidated"      — the shape was a candidate but the latest
                           pivot violates a required constraint
      "not_matched"      — shape_score below CANDIDATE_SHAPE_SCORE

    `matched_parts` maps required_parts (template-side) → skeleton
    pivot index (or None when the pivot count is insufficient).
    """

    schema_version: str
    kind: str
    template_id: str
    scale: str
    shape_score: float
    dtw_distance: float
    correlation_score: float | None
    side_bias: str
    status: str
    matched_parts: dict
    human_label: str
    human_explanation: str
    weakness_reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "template_id": self.template_id,
            "scale": self.scale,
            "shape_score": float(self.shape_score),
            "dtw_distance": float(self.dtw_distance),
            "correlation_score": (
                float(self.correlation_score)
                if self.correlation_score is not None else None
            ),
            "side_bias": str(self.side_bias),
            "status": str(self.status),
            "matched_parts": dict(self.matched_parts),
            "human_label": str(self.human_label),
            "human_explanation": str(self.human_explanation),
            "weakness_reasons": list(self.weakness_reasons),
        }


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Standard DTW with absolute cost. `a` and `b` are 1-D float arrays
    (we feed in y-only, both already aligned to a common x grid).

    Returns the path-length-normalised cumulative distance, so equal-
    length input is comparable across template kinds.
    """
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return float("inf")
    INF = float("inf")
    cost = np.full((n + 1, m + 1), INF, dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            d = abs(ai - b[j - 1])
            cost[i, j] = d + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )
        # path normalisation will divide by (n+m); we leave that for the
        # caller.
    return float(cost[n, m] / (n + m))


def _resample_template(
    template: PatternTemplate, n: int,
) -> np.ndarray:
    """Resample template's y-values onto an n-point uniform grid in [0,1]."""
    if not template.points:
        return np.zeros(n, dtype=float)
    xs = np.array([p[0] for p in template.points], dtype=float)
    ys = np.array([p[1] for p in template.points], dtype=float)
    target = np.linspace(0.0, 1.0, n)
    return np.interp(target, xs, ys)


def _skeleton_y(skel: WaveSkeleton, n: int) -> np.ndarray:
    if not skel.normalized_points:
        return np.zeros(n, dtype=float)
    xs = np.array([p[0] for p in skel.normalized_points], dtype=float)
    ys = np.array([p[1] for p in skel.normalized_points], dtype=float)
    if xs[0] == xs[-1]:
        return np.full(n, ys[-1] if len(ys) else 0.5, dtype=float)
    target = np.linspace(xs[0], xs[-1], n)
    return np.interp(target, xs, ys)


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size != b.size or a.size < 3:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    c = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(c):
        return None
    return c


# ---------------------------------------------------------------------------
# Part mapping (template's required_parts -> skeleton pivot index)
# ---------------------------------------------------------------------------


def _map_parts(template: PatternTemplate, skel: WaveSkeleton) -> dict:
    """Map each `required_part` to the skeleton pivot index whose normalised x
    is closest to the template point at that part's position.

    The mapping is intentionally simple. When the skeleton has fewer
    pivots than the template has parts, the parts that fall outside
    skeleton extent map to None.
    """
    parts = template.required_parts
    if not parts or not skel.pivots:
        return {p: None for p in parts}
    if not template.points or len(template.points) != len(parts):
        # Template parts list mismatched with point count; keep mapping
        # to None defensively.
        return {p: None for p in parts}

    pivot_indices = [int(p.index) for p in skel.pivots]
    if not pivot_indices:
        return {p: None for p in parts}

    idx_min = pivot_indices[0]
    idx_max = pivot_indices[-1]
    span = max(1, idx_max - idx_min)

    out: dict = {}
    for part_name, (px, _py) in zip(parts, template.points):
        target_idx_float = idx_min + px * span
        # find nearest pivot
        nearest = min(
            pivot_indices, key=lambda v: abs(v - target_idx_float)
        )
        out[part_name] = int(nearest)
    return out


# ---------------------------------------------------------------------------
# Status classification (heuristic)
# ---------------------------------------------------------------------------


def _classify_status(
    *,
    template: PatternTemplate,
    skel: WaveSkeleton,
    shape_score: float,
) -> tuple[str, tuple[str, ...]]:
    """Coarse status assignment.

    Heuristics:
      - shape_score < CANDIDATE  → not_matched
      - skeleton has fewer than 4 pivots and template requires ≥5 parts
        → forming, weakness "insufficient_pivots"
      - if breakout part maps to last pivot AND last pivot kind matches
        the expected direction (BUY → H, SELL → L for tops): treat as
        "neckline_broken"
      - otherwise default to "forming"
    """
    weakness: list[str] = []
    if shape_score < CANDIDATE_SHAPE_SCORE:
        return "not_matched", ("shape_score_below_threshold",)

    parts = template.required_parts
    pivots = skel.pivots
    if len(pivots) < min(4, len(parts)):
        weakness.append("insufficient_pivots")
        return "forming", tuple(weakness)

    # neckline-style breakout heuristic:
    breakout_part = "breakout"
    if breakout_part in parts:
        # last template point is breakout (by convention these templates
        # put it at x=1.0 last). Compare last skeleton pivot's kind:
        last = pivots[-1]
        if template.side_bias == "BUY":
            # breakout up: last pivot should be a high above the neckline
            if last.kind == "H":
                # use the highest of the pivots before last as neckline_peak
                prior = [p.price for p in pivots[:-1]]
                if prior and last.price > max(prior):
                    return "neckline_broken", ("breakout_up",)
            weakness.append("breakout_not_confirmed")
            return "forming", tuple(weakness)
        if template.side_bias == "SELL":
            last = pivots[-1]
            if last.kind == "L":
                prior = [p.price for p in pivots[:-1]]
                if prior and last.price < min(prior):
                    return "neckline_broken", ("breakout_down",)
            weakness.append("breakout_not_confirmed")
            return "forming", tuple(weakness)

    # Symmetric triangles etc. — break direction unknown.
    if template.side_bias == "NEUTRAL":
        weakness.append("breakout_direction_unknown")
        return "forming", tuple(weakness)

    return "forming", tuple(weakness)


# ---------------------------------------------------------------------------
# Human-readable text
# ---------------------------------------------------------------------------


_HUMAN_LABEL_JA: Final[dict] = {
    "double_bottom": "ダブルボトム候補",
    "double_top": "ダブルトップ候補",
    "head_and_shoulders": "三尊候補",
    "inverse_head_and_shoulders": "逆三尊候補",
    "bullish_flag": "上昇フラッグ候補",
    "bearish_flag": "下降フラッグ候補",
    "rising_wedge": "上昇ウェッジ候補",
    "falling_wedge": "下降ウェッジ候補",
    "ascending_triangle": "上昇三角形候補",
    "descending_triangle": "下降三角形候補",
    "symmetric_triangle": "対称三角形候補",
}


_STATUS_LABEL_JA: Final[dict] = {
    "forming": "形成中",
    "neckline_broken": "ネックラインブレイク済み",
    "retested": "リターンムーブ確認済み",
    "invalidated": "形が崩れた",
    "not_matched": "形として弱い",
}


def _human_explanation(
    template: PatternTemplate,
    status: str,
    shape_score: float,
    weakness: tuple[str, ...],
) -> str:
    base = template.description_ja
    status_ja = _STATUS_LABEL_JA.get(status, status)
    if status == "forming":
        weak_str = ""
        if "breakout_not_confirmed" in weakness:
            weak_str = " ただし、ネックライン (またはブレイクライン) はまだ明確に超えていません。"
        elif "insufficient_pivots" in weakness:
            weak_str = " ただし、波形のピボット数が少なく、形は仮判定です。"
        elif "breakout_direction_unknown" in weakness:
            weak_str = " ブレイク方向はまだ未確定です。"
        return f"{base} 現在は{status_ja}。{weak_str}".strip()
    if status == "neckline_broken":
        return f"{base} 現在は{status_ja}。リターンムーブ (押し戻し再確認) はまだです。"
    if status == "retested":
        return f"{base} {status_ja}。"
    if status == "invalidated":
        return f"{base} {status_ja}。"
    return f"{base} {status_ja} (shape_score={shape_score:.2f})。"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def match_skeleton_to_template(
    skel: WaveSkeleton,
    template: PatternTemplate,
    *,
    n_resample: int = 64,
) -> ShapeMatch:
    """Compare a single `WaveSkeleton` to a single `PatternTemplate`."""
    if not skel.normalized_points or not template.points:
        return ShapeMatch(
            schema_version=SCHEMA_VERSION,
            kind=template.kind, template_id=template.template_id,
            scale=skel.scale, shape_score=0.0,
            dtw_distance=float("inf"), correlation_score=None,
            side_bias=template.side_bias, status="not_matched",
            matched_parts={p: None for p in template.required_parts},
            human_label=_HUMAN_LABEL_JA.get(template.kind, template.kind),
            human_explanation="波形不足のため照合不可。",
            weakness_reasons=("empty_skeleton_or_template",),
        )
    a = _skeleton_y(skel, n_resample)
    b = _resample_template(template, n_resample)
    dtw = _dtw_distance(a, b)
    # Map DTW distance to a 0..1 score. The ceiling 0.5 is a heuristic
    # (worst case for normalised y-difference is 1.0; typical good
    # match is < 0.1).
    shape_score = float(max(0.0, 1.0 - 2.0 * dtw))
    corr = _correlation(a, b)

    status, weakness = _classify_status(
        template=template, skel=skel, shape_score=shape_score,
    )
    matched_parts = _map_parts(template, skel)
    label = _HUMAN_LABEL_JA.get(template.kind, template.kind)
    explanation = _human_explanation(
        template, status, shape_score, weakness,
    )
    return ShapeMatch(
        schema_version=SCHEMA_VERSION,
        kind=template.kind,
        template_id=template.template_id,
        scale=skel.scale,
        shape_score=shape_score,
        dtw_distance=dtw,
        correlation_score=corr,
        side_bias=template.side_bias,
        status=status,
        matched_parts=matched_parts,
        human_label=label,
        human_explanation=explanation,
        weakness_reasons=weakness,
    )


def match_skeleton(
    skel: WaveSkeleton,
    *,
    templates: tuple[PatternTemplate, ...] | None = None,
    n_resample: int = 64,
    keep_top_per_kind: bool = True,
) -> tuple[ShapeMatch, ...]:
    """Compare a skeleton against all templates and return the matches
    sorted by shape_score descending. By default keeps only the top
    match per kind to avoid duplicate variants in the report."""
    pool = templates if templates is not None else all_templates()
    if not skel.normalized_points:
        return ()
    raw: list[ShapeMatch] = [
        match_skeleton_to_template(skel, t, n_resample=n_resample)
        for t in pool
    ]
    if keep_top_per_kind:
        best_by_kind: dict[str, ShapeMatch] = {}
        for m in raw:
            cur = best_by_kind.get(m.kind)
            if cur is None or m.shape_score > cur.shape_score:
                best_by_kind[m.kind] = m
        raw = list(best_by_kind.values())
    raw.sort(key=lambda m: m.shape_score, reverse=True)
    return tuple(raw)


__all__ = [
    "SCHEMA_VERSION",
    "STRONG_SHAPE_SCORE",
    "CANDIDATE_SHAPE_SCORE",
    "ShapeMatch",
    "match_skeleton_to_template",
    "match_skeleton",
]
