"""Idealised pattern templates for shape-based audit (observation-only).

Each template is a small ordered sequence of normalised (x, y) points
in [0, 1] x [0, 1] that represents the canonical shape of one pattern
(e.g. double bottom, head-and-shoulders, ascending triangle, etc.).

The matcher in `pattern_shape_matcher` resamples both the live wave
skeleton and these templates to a fixed number of points, then computes
DTW distance between the two polylines.

`required_parts` enumerates the meaningful pieces of the shape so the
audit layer can dissect a matched pattern into labels (B1 / B2 / NL /
BR / ...). Each part name maps to a template-point index (declared
implicitly: parts are listed in the same chronological order as the
template points; see `_PART_INDEX_MAP` below for the mapping helper).

Conventions
-----------
- The leftmost point is `x = 0.0`, the rightmost is `x = 1.0`.
- Points are listed left-to-right by chronological order.
- `y` is normalised so the lowest pivot of the template = 0.0 and
  the highest = 1.0. Where a point sits between extremes, the y value
  reflects its rough proportion in canonical chart literature.
- `side_bias` is the directional reading once the pattern triggers:
  "BUY" for bottom reversals, "SELL" for top reversals,
  "NEUTRAL" for symmetric continuations.

Heuristic note
--------------
These templates are first-pass canonical shapes drawn from visual
trader literature. They have NOT been validated against multi-symbol
multi-period backtests. Downstream code must surface this status to
the user via `audit_notes` (see pattern_shape_review).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


SCHEMA_VERSION: Final[str] = "pattern_templates_v1"


@dataclass(frozen=True)
class PatternTemplate:
    """One idealised pattern shape.

    `template_id` is unique across the registry (e.g.
    "double_bottom_v1_standard"). `kind` is the family
    ("double_bottom"). Multiple variants share a kind.
    """

    template_id: str
    kind: str
    side_bias: str  # "BUY" | "SELL" | "NEUTRAL"
    points: tuple[tuple[float, float], ...]
    required_parts: tuple[str, ...]
    description_ja: str

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "kind": self.kind,
            "side_bias": self.side_bias,
            "points": [[float(x), float(y)] for x, y in self.points],
            "required_parts": list(self.required_parts),
            "description_ja": self.description_ja,
        }


# ---------------------------------------------------------------------------
# Reversal patterns — bottoms (BUY bias)
# ---------------------------------------------------------------------------


DOUBLE_BOTTOM_V1_STANDARD: Final[PatternTemplate] = PatternTemplate(
    template_id="double_bottom_v1_standard",
    kind="double_bottom",
    side_bias="BUY",
    points=(
        (0.00, 0.70),  # prior_high   — initial decline starts
        (0.20, 0.05),  # first_bottom — B1
        (0.45, 0.55),  # neckline_peak — middle peak / NL
        (0.70, 0.10),  # second_bottom — B2 (slightly higher than B1 ok)
        (1.00, 0.90),  # breakout      — closes above NL
    ),
    required_parts=(
        "prior_high", "first_bottom", "neckline_peak",
        "second_bottom", "breakout",
    ),
    description_ja=(
        "ダブルボトム (標準形)。2回底を試した後、ネックライン (中央の山) "
        "を上抜けて上昇する形。"
    ),
)


DOUBLE_BOTTOM_V2_SHALLOW_SECOND: Final[PatternTemplate] = PatternTemplate(
    template_id="double_bottom_v2_shallow_second",
    kind="double_bottom",
    side_bias="BUY",
    points=(
        (0.00, 0.65),
        (0.22, 0.08),
        (0.50, 0.50),
        (0.72, 0.18),  # second bottom is slightly shallower (early signal)
        (1.00, 0.85),
    ),
    required_parts=(
        "prior_high", "first_bottom", "neckline_peak",
        "second_bottom", "breakout",
    ),
    description_ja=(
        "ダブルボトム (2回目の底が浅い形)。買い圧の早期化を示唆する。"
    ),
)


DOUBLE_BOTTOM_V3_WIDE: Final[PatternTemplate] = PatternTemplate(
    template_id="double_bottom_v3_wide",
    kind="double_bottom",
    side_bias="BUY",
    points=(
        (0.00, 0.60),
        (0.15, 0.10),
        (0.50, 0.55),
        (0.80, 0.12),  # wide spacing between bottoms
        (1.00, 0.80),
    ),
    required_parts=(
        "prior_high", "first_bottom", "neckline_peak",
        "second_bottom", "breakout",
    ),
    description_ja="ダブルボトム (横幅が広い形)。底値圏で時間をかけて形成する。",
)


INVERSE_HEAD_AND_SHOULDERS_V1: Final[PatternTemplate] = PatternTemplate(
    template_id="inverse_head_and_shoulders_v1_standard",
    kind="inverse_head_and_shoulders",
    side_bias="BUY",
    points=(
        (0.00, 0.65),
        (0.15, 0.25),  # left_shoulder (LS)
        (0.30, 0.45),  # right side of LS (between LS and head)
        (0.50, 0.05),  # head (H) — lowest
        (0.70, 0.45),  # left side of RS (between head and RS)
        (0.85, 0.25),  # right_shoulder (RS)
        (1.00, 0.90),  # breakout above neckline
    ),
    required_parts=(
        "prior_high", "left_shoulder", "neckline_left_peak",
        "head", "neckline_right_peak", "right_shoulder", "breakout",
    ),
    description_ja=(
        "逆三尊 (Inverse Head and Shoulders)。中央の頭が最も深く、"
        "両肩はほぼ同じ高さ。ネックライン上抜けで完成。"
    ),
)


# ---------------------------------------------------------------------------
# Reversal patterns — tops (SELL bias)
# ---------------------------------------------------------------------------


DOUBLE_TOP_V1_STANDARD: Final[PatternTemplate] = PatternTemplate(
    template_id="double_top_v1_standard",
    kind="double_top",
    side_bias="SELL",
    points=(
        (0.00, 0.30),  # prior_low
        (0.20, 0.95),  # first_top — P1
        (0.45, 0.45),  # neckline_trough — middle trough / NL
        (0.70, 0.90),  # second_top — P2
        (1.00, 0.10),  # breakout — closes below NL
    ),
    required_parts=(
        "prior_low", "first_top", "neckline_trough",
        "second_top", "breakout",
    ),
    description_ja=(
        "ダブルトップ (標準形)。2回天井を試した後、ネックライン (中央の谷) "
        "を下抜けて下落する形。"
    ),
)


DOUBLE_TOP_V2_SHALLOW_SECOND: Final[PatternTemplate] = PatternTemplate(
    template_id="double_top_v2_shallow_second",
    kind="double_top",
    side_bias="SELL",
    points=(
        (0.00, 0.35),
        (0.22, 0.92),
        (0.50, 0.50),
        (0.72, 0.82),  # second top slightly lower (early signal)
        (1.00, 0.15),
    ),
    required_parts=(
        "prior_low", "first_top", "neckline_trough",
        "second_top", "breakout",
    ),
    description_ja="ダブルトップ (2回目の天井が低い形)。売り優勢の早期化を示唆。",
)


DOUBLE_TOP_V3_WIDE: Final[PatternTemplate] = PatternTemplate(
    template_id="double_top_v3_wide",
    kind="double_top",
    side_bias="SELL",
    points=(
        (0.00, 0.40),
        (0.15, 0.90),
        (0.50, 0.45),
        (0.80, 0.88),
        (1.00, 0.20),
    ),
    required_parts=(
        "prior_low", "first_top", "neckline_trough",
        "second_top", "breakout",
    ),
    description_ja="ダブルトップ (横幅が広い形)。天井圏で時間をかけて形成する。",
)


HEAD_AND_SHOULDERS_V1: Final[PatternTemplate] = PatternTemplate(
    template_id="head_and_shoulders_v1_standard",
    kind="head_and_shoulders",
    side_bias="SELL",
    points=(
        (0.00, 0.35),
        (0.15, 0.75),  # left_shoulder (LS)
        (0.30, 0.55),  # left neckline trough
        (0.50, 0.95),  # head (H) — highest
        (0.70, 0.55),  # right neckline trough
        (0.85, 0.75),  # right_shoulder (RS)
        (1.00, 0.10),  # breakout below neckline
    ),
    required_parts=(
        "prior_low", "left_shoulder", "neckline_left_trough",
        "head", "neckline_right_trough", "right_shoulder", "breakout",
    ),
    description_ja=(
        "三尊 (Head and Shoulders)。中央の頭が最も高く、両肩はほぼ同じ高さ。"
        "ネックライン下抜けで完成。"
    ),
)


# ---------------------------------------------------------------------------
# Continuation patterns — flags (directional)
# ---------------------------------------------------------------------------


BULLISH_FLAG_V1: Final[PatternTemplate] = PatternTemplate(
    template_id="bullish_flag_v1_standard",
    kind="bullish_flag",
    side_bias="BUY",
    points=(
        (0.00, 0.05),  # impulse_start
        (0.30, 0.85),  # impulse_end (sharp rise)
        (0.45, 0.65),  # consolidation_high
        (0.60, 0.70),  # consolidation_mid
        (0.75, 0.55),  # consolidation_low
        (1.00, 0.95),  # breakout
    ),
    required_parts=(
        "impulse_start", "impulse_end",
        "consolidation_high", "consolidation_mid",
        "consolidation_low", "breakout",
    ),
    description_ja=(
        "上昇フラッグ。強い上昇 (ポール) の後、わずかに下向きの平行レンジ"
        "を形成し、再度上抜ける継続パターン。"
    ),
)


BEARISH_FLAG_V1: Final[PatternTemplate] = PatternTemplate(
    template_id="bearish_flag_v1_standard",
    kind="bearish_flag",
    side_bias="SELL",
    points=(
        (0.00, 0.95),  # impulse_start
        (0.30, 0.15),  # impulse_end (sharp drop)
        (0.45, 0.35),  # consolidation_low
        (0.60, 0.30),  # consolidation_mid
        (0.75, 0.45),  # consolidation_high
        (1.00, 0.05),  # breakout
    ),
    required_parts=(
        "impulse_start", "impulse_end",
        "consolidation_low", "consolidation_mid",
        "consolidation_high", "breakout",
    ),
    description_ja=(
        "下降フラッグ。強い下落 (ポール) の後、わずかに上向きの平行レンジ"
        "を形成し、再度下抜ける継続パターン。"
    ),
)


# ---------------------------------------------------------------------------
# Wedges
# ---------------------------------------------------------------------------


RISING_WEDGE_V1: Final[PatternTemplate] = PatternTemplate(
    template_id="rising_wedge_v1_standard",
    kind="rising_wedge",
    side_bias="SELL",
    points=(
        (0.00, 0.10),
        (0.15, 0.55),
        (0.30, 0.30),
        (0.45, 0.70),
        (0.60, 0.55),  # converging from below
        (0.75, 0.85),
        (0.90, 0.78),  # apex approach
        (1.00, 0.20),  # breakout downward
    ),
    required_parts=(
        "lower_anchor_1", "upper_anchor_1",
        "lower_anchor_2", "upper_anchor_2",
        "lower_anchor_3", "upper_anchor_3",
        "apex", "breakout",
    ),
    description_ja=(
        "上昇ウェッジ (Rising Wedge)。安値・高値とも切り上げるが、"
        "上値が抑えられて収束し、最終的に下方ブレイクする弱気パターン。"
    ),
)


FALLING_WEDGE_V1: Final[PatternTemplate] = PatternTemplate(
    template_id="falling_wedge_v1_standard",
    kind="falling_wedge",
    side_bias="BUY",
    points=(
        (0.00, 0.90),
        (0.15, 0.45),
        (0.30, 0.70),
        (0.45, 0.30),
        (0.60, 0.45),
        (0.75, 0.15),
        (0.90, 0.22),  # apex approach
        (1.00, 0.80),  # breakout upward
    ),
    required_parts=(
        "upper_anchor_1", "lower_anchor_1",
        "upper_anchor_2", "lower_anchor_2",
        "upper_anchor_3", "lower_anchor_3",
        "apex", "breakout",
    ),
    description_ja=(
        "下降ウェッジ (Falling Wedge)。高値・安値とも切り下げるが、"
        "下値が支えられて収束し、最終的に上方ブレイクする強気パターン。"
    ),
)


# ---------------------------------------------------------------------------
# Triangles
# ---------------------------------------------------------------------------


ASCENDING_TRIANGLE_V1: Final[PatternTemplate] = PatternTemplate(
    template_id="ascending_triangle_v1_standard",
    kind="ascending_triangle",
    side_bias="BUY",
    points=(
        (0.00, 0.20),
        (0.20, 0.85),  # touches flat top
        (0.35, 0.40),  # ascending bottom 1
        (0.50, 0.85),  # touches flat top
        (0.65, 0.55),  # ascending bottom 2 (higher)
        (0.80, 0.85),
        (1.00, 0.95),  # breakout up
    ),
    required_parts=(
        "lower_anchor_1", "upper_anchor_1",
        "lower_anchor_2", "upper_anchor_2",
        "lower_anchor_3", "upper_anchor_3",
        "breakout",
    ),
    description_ja=(
        "上昇三角形 (Ascending Triangle)。上値が水平で、安値が切り上がる。"
        "最終的に上方ブレイクしやすい強気パターン。"
    ),
)


DESCENDING_TRIANGLE_V1: Final[PatternTemplate] = PatternTemplate(
    template_id="descending_triangle_v1_standard",
    kind="descending_triangle",
    side_bias="SELL",
    points=(
        (0.00, 0.80),
        (0.20, 0.15),
        (0.35, 0.60),
        (0.50, 0.15),
        (0.65, 0.45),
        (0.80, 0.15),
        (1.00, 0.05),  # breakout down
    ),
    required_parts=(
        "upper_anchor_1", "lower_anchor_1",
        "upper_anchor_2", "lower_anchor_2",
        "upper_anchor_3", "lower_anchor_3",
        "breakout",
    ),
    description_ja=(
        "下降三角形 (Descending Triangle)。下値が水平で、高値が切り下がる。"
        "最終的に下方ブレイクしやすい弱気パターン。"
    ),
)


SYMMETRIC_TRIANGLE_V1: Final[PatternTemplate] = PatternTemplate(
    template_id="symmetric_triangle_v1_standard",
    kind="symmetric_triangle",
    side_bias="NEUTRAL",
    points=(
        (0.00, 0.10),
        (0.18, 0.85),
        (0.35, 0.30),
        (0.52, 0.70),
        (0.68, 0.45),
        (0.84, 0.60),
        (1.00, 0.50),  # apex (no directional break)
    ),
    required_parts=(
        "lower_anchor_1", "upper_anchor_1",
        "lower_anchor_2", "upper_anchor_2",
        "lower_anchor_3", "upper_anchor_3",
        "apex",
    ),
    description_ja=(
        "対称三角形 (Symmetric Triangle)。高値切り下げ・安値切り上げで収束。"
        "ブレイク方向はどちらにも出るため、方向は中立。"
    ),
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_ALL_TEMPLATES: Final[tuple[PatternTemplate, ...]] = (
    DOUBLE_BOTTOM_V1_STANDARD,
    DOUBLE_BOTTOM_V2_SHALLOW_SECOND,
    DOUBLE_BOTTOM_V3_WIDE,
    INVERSE_HEAD_AND_SHOULDERS_V1,
    DOUBLE_TOP_V1_STANDARD,
    DOUBLE_TOP_V2_SHALLOW_SECOND,
    DOUBLE_TOP_V3_WIDE,
    HEAD_AND_SHOULDERS_V1,
    BULLISH_FLAG_V1,
    BEARISH_FLAG_V1,
    RISING_WEDGE_V1,
    FALLING_WEDGE_V1,
    ASCENDING_TRIANGLE_V1,
    DESCENDING_TRIANGLE_V1,
    SYMMETRIC_TRIANGLE_V1,
)


def all_templates() -> tuple[PatternTemplate, ...]:
    """Return the full template registry, frozen tuple."""
    return _ALL_TEMPLATES


def templates_by_kind(kind: str) -> tuple[PatternTemplate, ...]:
    return tuple(t for t in _ALL_TEMPLATES if t.kind == kind)


def template_by_id(template_id: str) -> PatternTemplate | None:
    for t in _ALL_TEMPLATES:
        if t.template_id == template_id:
            return t
    return None


__all__ = [
    "SCHEMA_VERSION",
    "PatternTemplate",
    "all_templates",
    "templates_by_kind",
    "template_by_id",
    "DOUBLE_BOTTOM_V1_STANDARD",
    "DOUBLE_BOTTOM_V2_SHALLOW_SECOND",
    "DOUBLE_BOTTOM_V3_WIDE",
    "INVERSE_HEAD_AND_SHOULDERS_V1",
    "DOUBLE_TOP_V1_STANDARD",
    "DOUBLE_TOP_V2_SHALLOW_SECOND",
    "DOUBLE_TOP_V3_WIDE",
    "HEAD_AND_SHOULDERS_V1",
    "BULLISH_FLAG_V1",
    "BEARISH_FLAG_V1",
    "RISING_WEDGE_V1",
    "FALLING_WEDGE_V1",
    "ASCENDING_TRIANGLE_V1",
    "DESCENDING_TRIANGLE_V1",
    "SYMMETRIC_TRIANGLE_V1",
]
