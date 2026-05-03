"""Multi-scale chart reconstruction (royal_road_decision_v2 follow-up).

A human chart reader rarely commits to a single zoom level. They flip
between micro / short / medium / long timeframe scales to confirm
swings, levels, trendlines, and patterns. This module provides the
structural API for that:

    ScaleAnalysis (per-scale snapshot of:
        swing_points
        level_zone_candidates
        trendline_candidates
        pattern_candidates
        wave_structure
        visual_quality)

    reconstruct_chart_multi_scale(df, atr_value, last_close)
        -> {"micro": ScaleAnalysis, "short": ..., "medium": ..., "long": ...}

Future-leak rule
----------------
Each scale slices `df.iloc[-bars:]` from the END of the supplied frame.
The caller is expected to pass `df.iloc[: i + 1]` (visible window),
identical to what the rest of the v2 detectors require. No bar with
ts > parent_bar_ts is ever consulted.

Scale lengths (heuristic, tunable):
  micro  : 30 bars
  short  : 100 bars
  medium : 300 bars
  long   : 1000 bars

When the visible window is shorter than a scale's bar count, the
scale is reported with `available=False, unavailable_reason="short_history"`.
The other scales still populate.

This module is observation-only. It does not consult `decide_action`
and the result is for trace + diagnostic consumption only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from .chart_patterns import detect_patterns
from .patterns import detect_swings
from .support_resistance import detect_levels
from .trendlines import detect_trendlines


SCHEMA_VERSION: Final[str] = "chart_reconstruction_v2"

SCALES: Final[dict[str, int]] = {
    "micro": 30,
    "short": 100,
    "medium": 300,
    "long": 1000,
}


@dataclass(frozen=True)
class ScaleAnalysis:
    scale: str
    bars_required: int
    bars_used: int
    available: bool
    unavailable_reason: str | None
    swing_points: list      # list of swing dicts (truncated to last 20)
    level_zone_candidates: list
    trendline_candidates: list
    pattern_candidates: list
    wave_structure: dict
    visual_quality: float

    def to_dict(self) -> dict:
        return {
            "scale": self.scale,
            "bars_required": int(self.bars_required),
            "bars_used": int(self.bars_used),
            "available": bool(self.available),
            "unavailable_reason": self.unavailable_reason,
            "swing_points": list(self.swing_points),
            "level_zone_candidates": list(self.level_zone_candidates),
            "trendline_candidates": list(self.trendline_candidates),
            "pattern_candidates": list(self.pattern_candidates),
            "wave_structure": dict(self.wave_structure),
            "visual_quality": float(self.visual_quality),
        }


def empty_scale(scale: str, reason: str) -> ScaleAnalysis:
    return ScaleAnalysis(
        scale=scale,
        bars_required=SCALES.get(scale, 0),
        bars_used=0,
        available=False,
        unavailable_reason=reason,
        swing_points=[],
        level_zone_candidates=[],
        trendline_candidates=[],
        pattern_candidates=[],
        wave_structure={},
        visual_quality=0.0,
    )


def _swing_to_dict(sw, kind_str: str) -> dict:
    return {
        "index": int(sw.index),
        "ts": (
            sw.ts.isoformat() if hasattr(sw.ts, "isoformat") else str(sw.ts)
        ),
        "price": float(sw.price),
        "kind": kind_str,
    }


def _wave_structure_summary(highs, lows) -> dict:
    """Compact HH/HL/LH/LL summary."""
    if not highs and not lows:
        return {"sequence": [], "trend": "UNKNOWN"}
    swings = sorted(highs + lows, key=lambda s: s.index)
    if not swings:
        return {"sequence": [], "trend": "UNKNOWN"}
    seq: list[str] = []
    last_high = None
    last_low = None
    for sw in swings:
        if sw.kind == "H":
            if last_high is None:
                last_high = sw.price
            else:
                seq.append("HH" if sw.price > last_high else "LH")
                last_high = sw.price
        else:
            if last_low is None:
                last_low = sw.price
            else:
                seq.append("HL" if sw.price > last_low else "LL")
                last_low = sw.price
    last4 = seq[-4:] if seq else []
    up = sum(1 for s in last4 if s in ("HH", "HL"))
    down = sum(1 for s in last4 if s in ("LL", "LH"))
    if up >= 3 and down <= 1:
        trend = "UP"
    elif down >= 3 and up <= 1:
        trend = "DOWN"
    elif up >= 2 and down >= 2:
        trend = "MIXED"
    else:
        trend = "RANGE"
    return {"sequence": seq, "last_4": last4, "trend": trend}


def _visual_quality(
    *,
    n_swings: int, n_levels: int, n_patterns: int,
    n_trendlines: int,
) -> float:
    """Rough heuristic: more confirmed swings + levels + trendlines +
    patterns = higher visual quality at this scale. Capped at [0, 1]."""
    raw = (
        min(1.0, n_swings / 12.0) * 0.3
        + min(1.0, n_levels / 5.0) * 0.3
        + min(1.0, n_trendlines / 2.0) * 0.2
        + min(1.0, n_patterns / 3.0) * 0.2
    )
    return float(max(0.0, min(1.0, raw)))


def _analyse_scale(
    *, scale: str, df: pd.DataFrame, atr_value: float, last_close: float,
) -> ScaleAnalysis:
    bars_required = SCALES[scale]
    if df is None or len(df) == 0:
        return empty_scale(scale, "no_data")
    if len(df) < bars_required:
        return empty_scale(scale, "short_history")
    sub = df.iloc[-bars_required:]
    bars_used = int(len(sub))

    highs, lows = detect_swings(sub, lookback=3)
    sr = detect_levels(sub, atr_value=atr_value, last_close=last_close)
    tl = detect_trendlines(sub, atr_value=atr_value, last_close=last_close)
    cp = detect_patterns(sub, atr_value=atr_value, last_close=last_close)

    swing_points = [
        _swing_to_dict(sw, "H") for sw in highs[-10:]
    ] + [
        _swing_to_dict(sw, "L") for sw in lows[-10:]
    ]
    level_zone_candidates = [
        lvl.to_dict() for lvl in sr.selected_level_zones_top5
    ]
    trendline_candidates = [
        tline.to_dict() for tline in tl.selected_trendlines_top3
    ]
    pattern_candidates = [
        m.to_dict() for m in cp.selected_patterns_top5
    ]
    wave_structure = _wave_structure_summary(highs, lows)
    quality = _visual_quality(
        n_swings=len(highs) + len(lows),
        n_levels=len(level_zone_candidates),
        n_patterns=len(pattern_candidates),
        n_trendlines=len(trendline_candidates),
    )
    return ScaleAnalysis(
        scale=scale,
        bars_required=bars_required,
        bars_used=bars_used,
        available=True,
        unavailable_reason=None,
        swing_points=swing_points,
        level_zone_candidates=level_zone_candidates,
        trendline_candidates=trendline_candidates,
        pattern_candidates=pattern_candidates,
        wave_structure=wave_structure,
        visual_quality=quality,
    )


def reconstruct_chart_multi_scale(
    df: pd.DataFrame | None,
    *,
    atr_value: float | None,
    last_close: float | None,
) -> dict:
    """Run the four-scale reconstruction and return a dict mapping
    scale name → ScaleAnalysis.to_dict(). Always includes all 4 scales
    (unavailable ones report `unavailable_reason`)."""
    out: dict = {"schema_version": SCHEMA_VERSION, "scales": {}}
    if df is None or atr_value is None or atr_value <= 0 or last_close is None:
        for sc in SCALES.keys():
            out["scales"][sc] = empty_scale(sc, "missing_inputs").to_dict()
        return out
    for sc in SCALES.keys():
        out["scales"][sc] = _analyse_scale(
            scale=sc, df=df, atr_value=atr_value, last_close=last_close,
        ).to_dict()
    return out


__all__ = [
    "SCHEMA_VERSION",
    "SCALES",
    "ScaleAnalysis",
    "empty_scale",
    "reconstruct_chart_multi_scale",
]
