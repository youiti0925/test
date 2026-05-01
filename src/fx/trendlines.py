"""Trendline fitting from confirmed swings.

Used by royal_road_decision_v2 only. Future-leak safe: relies on
`patterns.detect_swings()` confirmed output.

This is a v2-minimal implementation. It fits 2-anchor lines (the most
recent two swing highs for resistance, the most recent two swing lows
for support), reports slope / intercept / projected level / distance.
A v3 expansion could include 3+ anchor confirmation, log-scale fits,
or dynamic re-anchoring.

Heuristic constants (NOT validated, NOT in PARAMETER_BASELINE_V1):
  _MIN_BARS_BETWEEN_ANCHORS  : two anchors must be >= N bars apart
  _NEAR_LINE_ATR             : within this many ATR = "near the line"
  _BROKEN_BAND_ATR           : close beyond by this many ATR = broken
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from .patterns import detect_swings


_MIN_BARS_BETWEEN_ANCHORS: Final[int] = 5
_NEAR_LINE_ATR: Final[float] = 0.5
_BROKEN_BAND_ATR: Final[float] = 0.5


TrendlineKind = Literal["ascending_support", "descending_resistance"]


@dataclass(frozen=True)
class Trendline:
    kind: TrendlineKind
    slope: float                # price per bar
    intercept: float
    anchor_indices: tuple[int, int]
    anchor_prices: tuple[float, float]
    touch_count: int
    broken: bool
    distance_to_line_atr: float | None
    strength_score: float

    def projected_price(self, bar_index: int) -> float:
        return float(self.slope * bar_index + self.intercept)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "slope": float(self.slope),
            "intercept": float(self.intercept),
            "anchor_indices": list(self.anchor_indices),
            "anchor_prices": [float(p) for p in self.anchor_prices],
            "touch_count": int(self.touch_count),
            "broken": bool(self.broken),
            "distance_to_line_atr": (
                float(self.distance_to_line_atr)
                if self.distance_to_line_atr is not None else None
            ),
            "strength_score": float(self.strength_score),
        }


@dataclass(frozen=True)
class TrendlineContext:
    schema_version: str
    ascending_support: Trendline | None
    descending_resistance: Trendline | None
    bullish_signal: bool        # near asc support OR descending resistance broken up
    bearish_signal: bool        # near desc resistance OR ascending support broken down
    reason: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "ascending_support": (
                self.ascending_support.to_dict()
                if self.ascending_support else None
            ),
            "descending_resistance": (
                self.descending_resistance.to_dict()
                if self.descending_resistance else None
            ),
            "bullish_signal": bool(self.bullish_signal),
            "bearish_signal": bool(self.bearish_signal),
            "reason": self.reason,
        }


def empty_context(reason: str = "insufficient_data") -> TrendlineContext:
    return TrendlineContext(
        schema_version="trendline_context_v2",
        ascending_support=None,
        descending_resistance=None,
        bullish_signal=False,
        bearish_signal=False,
        reason=reason,
    )


def _fit_two_anchor_line(
    sw_a, sw_b,
    *,
    closes: np.ndarray,
    atr_value: float,
    kind: TrendlineKind,
) -> Trendline | None:
    """Linear fit through two swings; check broken state vs current close."""
    if abs(sw_b.index - sw_a.index) < _MIN_BARS_BETWEEN_ANCHORS:
        return None
    dx = sw_b.index - sw_a.index
    dy = sw_b.price - sw_a.price
    slope = dy / dx if dx != 0 else 0.0
    intercept = sw_a.price - slope * sw_a.index

    # Validate the line direction matches the kind.
    if kind == "ascending_support" and slope <= 0:
        return None
    if kind == "descending_resistance" and slope >= 0:
        return None

    # Touch count = swings other than the anchors that lie within
    # _NEAR_LINE_ATR * atr of the line. We don't have a list of all
    # swings here without a re-run, so we keep touch_count = 2 (the
    # two anchors). A future PR can scan all swings.
    touch_count = 2

    last_index = len(closes) - 1
    last_close = float(closes[-1])
    projected_at_last = slope * last_index + intercept
    distance_atr = abs(last_close - projected_at_last) / atr_value

    if kind == "ascending_support":
        # Broken DOWN through support: close < line - 0.5 ATR
        broken = bool(last_close < projected_at_last - _BROKEN_BAND_ATR * atr_value)
    else:
        # descending_resistance broken UP through resistance.
        broken = bool(last_close > projected_at_last + _BROKEN_BAND_ATR * atr_value)

    strength_score = float(touch_count * (1.0 if not broken else 0.5))
    return Trendline(
        kind=kind,
        slope=float(slope),
        intercept=float(intercept),
        anchor_indices=(int(sw_a.index), int(sw_b.index)),
        anchor_prices=(float(sw_a.price), float(sw_b.price)),
        touch_count=touch_count,
        broken=broken,
        distance_to_line_atr=float(distance_atr),
        strength_score=strength_score,
    )


def detect_trendlines(
    df: pd.DataFrame,
    *,
    atr_value: float | None,
    last_close: float | None,
    lookback: int = 3,
) -> TrendlineContext:
    if (
        df is None or len(df) < 2 * lookback + 8
        or atr_value is None or atr_value <= 0
        or last_close is None
    ):
        return empty_context("insufficient_data")

    highs, lows = detect_swings(df, lookback=lookback)
    closes = df["close"].to_numpy()

    asc: Trendline | None = None
    if len(lows) >= 2:
        asc = _fit_two_anchor_line(
            lows[-2], lows[-1],
            closes=closes, atr_value=atr_value, kind="ascending_support",
        )
    desc: Trendline | None = None
    if len(highs) >= 2:
        desc = _fit_two_anchor_line(
            highs[-2], highs[-1],
            closes=closes, atr_value=atr_value, kind="descending_resistance",
        )

    if asc is None and desc is None:
        return empty_context("no_valid_trendlines")

    bullish_signal = bool(
        (asc is not None and not asc.broken
         and asc.distance_to_line_atr is not None
         and asc.distance_to_line_atr <= _NEAR_LINE_ATR
         and last_close >= asc.projected_price(len(closes) - 1))
        or (desc is not None and desc.broken
            and last_close > desc.projected_price(len(closes) - 1))
    )
    bearish_signal = bool(
        (desc is not None and not desc.broken
         and desc.distance_to_line_atr is not None
         and desc.distance_to_line_atr <= _NEAR_LINE_ATR
         and last_close <= desc.projected_price(len(closes) - 1))
        or (asc is not None and asc.broken
            and last_close < asc.projected_price(len(closes) - 1))
    )

    reason_parts = []
    if asc:
        reason_parts.append(
            f"asc_support@slope={asc.slope:.6f}"
            + (" broken" if asc.broken else "")
        )
    if desc:
        reason_parts.append(
            f"desc_resistance@slope={desc.slope:.6f}"
            + (" broken" if desc.broken else "")
        )
    reason = "; ".join(reason_parts) or "no_lines"

    return TrendlineContext(
        schema_version="trendline_context_v2",
        ascending_support=asc,
        descending_resistance=desc,
        bullish_signal=bullish_signal,
        bearish_signal=bearish_signal,
        reason=reason,
    )


__all__ = [
    "Trendline",
    "TrendlineContext",
    "detect_trendlines",
    "empty_context",
]
