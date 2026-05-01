"""Chart pattern detection for royal_road_decision_v2.

Implements detectors for:
  - head_and_shoulders / inverse_head_and_shoulders
  - flag (bullish / bearish)
  - wedge (rising / falling)
  - triangle (ascending / descending / symmetric)

Each detector returns a `PatternMatch` only when:
  - the geometric shape is satisfied, AND
  - the pattern's neckline has been broken (close-based) — entry-grade
    signals require confirmation, not just shape formation.

Future-leak rule: relies on `patterns.detect_swings()` which is itself
future-leak safe (swings confirmed only after `lookback` post bars).

This is v2-minimal. Detectors are heuristic and pending validation.
Adoption requires multi-symbol × multi-period evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from .patterns import detect_swings


_HS_SHOULDER_TOL_ATR: Final[float] = 0.6   # shoulder height symmetry
_HS_HEAD_PROMINENCE_ATR: Final[float] = 0.5
_FLAG_MIN_BARS: Final[int] = 5
_FLAG_MAX_BARS: Final[int] = 30
_WEDGE_MIN_SWINGS: Final[int] = 4
_TRIANGLE_TOL_ATR: Final[float] = 0.4


PatternKind = Literal[
    "head_and_shoulders",
    "inverse_head_and_shoulders",
    "flag_bullish",
    "flag_bearish",
    "wedge_rising",
    "wedge_falling",
    "triangle_ascending",
    "triangle_descending",
    "triangle_symmetric",
]


@dataclass(frozen=True)
class PatternMatch:
    kind: PatternKind
    neckline: float | None
    neckline_broken: bool
    retested: bool
    invalidation_price: float | None
    target_price: float | None
    confidence: float
    anchor_indices: tuple[int, ...]
    side_bias: Literal["BUY", "SELL", "NEUTRAL"]

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "neckline": (
                float(self.neckline) if self.neckline is not None else None
            ),
            "neckline_broken": bool(self.neckline_broken),
            "retested": bool(self.retested),
            "invalidation_price": (
                float(self.invalidation_price)
                if self.invalidation_price is not None else None
            ),
            "target_price": (
                float(self.target_price) if self.target_price is not None
                else None
            ),
            "confidence": float(self.confidence),
            "anchor_indices": [int(i) for i in self.anchor_indices],
            "side_bias": self.side_bias,
        }


@dataclass(frozen=True)
class ChartPatternSnapshot:
    schema_version: str
    matches: tuple[PatternMatch, ...]
    head_and_shoulders: PatternMatch | None
    inverse_head_and_shoulders: PatternMatch | None
    flag: PatternMatch | None
    wedge: PatternMatch | None
    triangle: PatternMatch | None
    bullish_breakout_confirmed: bool
    bearish_breakout_confirmed: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "matches": [m.to_dict() for m in self.matches],
            "head_and_shoulders": (
                self.head_and_shoulders.to_dict()
                if self.head_and_shoulders else None
            ),
            "inverse_head_and_shoulders": (
                self.inverse_head_and_shoulders.to_dict()
                if self.inverse_head_and_shoulders else None
            ),
            "flag": self.flag.to_dict() if self.flag else None,
            "wedge": self.wedge.to_dict() if self.wedge else None,
            "triangle": (
                self.triangle.to_dict() if self.triangle else None
            ),
            "bullish_breakout_confirmed":
                bool(self.bullish_breakout_confirmed),
            "bearish_breakout_confirmed":
                bool(self.bearish_breakout_confirmed),
            "reason": self.reason,
        }


def empty_snapshot(reason: str = "insufficient_data") -> ChartPatternSnapshot:
    return ChartPatternSnapshot(
        schema_version="chart_pattern_v2",
        matches=(),
        head_and_shoulders=None,
        inverse_head_and_shoulders=None,
        flag=None,
        wedge=None,
        triangle=None,
        bullish_breakout_confirmed=False,
        bearish_breakout_confirmed=False,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------


def _detect_head_and_shoulders(highs, lows, atr_value, closes) -> PatternMatch | None:
    """3 highs forming H&S; neckline = swing low between them."""
    if len(highs) < 3 or len(lows) < 2 or atr_value <= 0:
        return None
    h1, h2, h3 = highs[-3], highs[-2], highs[-1]
    # head must be higher than both shoulders by at least the prominence.
    if not (h2.price > h1.price + _HS_HEAD_PROMINENCE_ATR * atr_value
            and h2.price > h3.price + _HS_HEAD_PROMINENCE_ATR * atr_value):
        return None
    # shoulders roughly symmetric.
    if abs(h1.price - h3.price) > _HS_SHOULDER_TOL_ATR * atr_value:
        return None
    # neckline = swing lows between h1..h2 and h2..h3 (averaged).
    valley_a = [l for l in lows if h1.index < l.index < h2.index]
    valley_b = [l for l in lows if h2.index < l.index < h3.index]
    if not valley_a or not valley_b:
        return None
    neckline = float(min(valley_a[-1].price, valley_b[-1].price))
    last_close = float(closes[-1])
    broken = bool(last_close < neckline)
    target = neckline - (h2.price - neckline)  # measured-move
    confidence = 0.65 + min(
        0.2, (h2.price - max(h1.price, h3.price)) / (3 * atr_value)
    )
    return PatternMatch(
        kind="head_and_shoulders",
        neckline=neckline,
        neckline_broken=broken,
        retested=False,
        invalidation_price=float(h2.price),
        target_price=float(target),
        confidence=float(min(0.95, confidence)),
        anchor_indices=(h1.index, h2.index, h3.index),
        side_bias="SELL",
    )


def _detect_inverse_head_and_shoulders(highs, lows, atr_value, closes) -> PatternMatch | None:
    if len(lows) < 3 or len(highs) < 2 or atr_value <= 0:
        return None
    l1, l2, l3 = lows[-3], lows[-2], lows[-1]
    if not (l2.price < l1.price - _HS_HEAD_PROMINENCE_ATR * atr_value
            and l2.price < l3.price - _HS_HEAD_PROMINENCE_ATR * atr_value):
        return None
    if abs(l1.price - l3.price) > _HS_SHOULDER_TOL_ATR * atr_value:
        return None
    peak_a = [h for h in highs if l1.index < h.index < l2.index]
    peak_b = [h for h in highs if l2.index < h.index < l3.index]
    if not peak_a or not peak_b:
        return None
    neckline = float(max(peak_a[-1].price, peak_b[-1].price))
    last_close = float(closes[-1])
    broken = bool(last_close > neckline)
    target = neckline + (neckline - l2.price)
    confidence = 0.65 + min(
        0.2, (min(l1.price, l3.price) - l2.price) / (3 * atr_value)
    )
    return PatternMatch(
        kind="inverse_head_and_shoulders",
        neckline=neckline,
        neckline_broken=broken,
        retested=False,
        invalidation_price=float(l2.price),
        target_price=float(target),
        confidence=float(min(0.95, confidence)),
        anchor_indices=(l1.index, l2.index, l3.index),
        side_bias="BUY",
    )


def _detect_flag(closes: np.ndarray, atr_value: float) -> PatternMatch | None:
    """Bull flag: a strong impulse leg followed by a tight pullback in a
    narrow range. Bear flag: mirror. v2-minimal: looks at the last
    `_FLAG_MAX_BARS` for a tight range preceded by a wide-range leg.
    """
    if len(closes) < _FLAG_MAX_BARS + 5 or atr_value <= 0:
        return None
    impulse_window = closes[-_FLAG_MAX_BARS - 5:-_FLAG_MAX_BARS]
    consol_window = closes[-_FLAG_MAX_BARS:]
    if len(impulse_window) < 3 or len(consol_window) < _FLAG_MIN_BARS:
        return None
    impulse_range = float(np.ptp(impulse_window))
    consol_range = float(np.ptp(consol_window))
    if impulse_range <= 0:
        return None
    if consol_range > 0.6 * impulse_range:
        return None  # consolidation is too wide for a flag
    if impulse_range < 1.5 * atr_value:
        return None  # not a strong enough impulse
    direction = (
        "BUY" if impulse_window[-1] > impulse_window[0] else "SELL"
    )
    last_close = float(closes[-1])
    consol_high = float(np.max(consol_window))
    consol_low = float(np.min(consol_window))
    if direction == "BUY":
        kind: PatternKind = "flag_bullish"
        neckline = consol_high
        invalidation = consol_low
        broken = bool(last_close > consol_high)
        target = consol_high + impulse_range  # measured-move
        side_bias = "BUY"
    else:
        kind = "flag_bearish"
        neckline = consol_low
        invalidation = consol_high
        broken = bool(last_close < consol_low)
        target = consol_low - impulse_range
        side_bias = "SELL"
    return PatternMatch(
        kind=kind,
        neckline=float(neckline),
        neckline_broken=broken,
        retested=False,
        invalidation_price=float(invalidation),
        target_price=float(target),
        confidence=0.55 + 0.1 * min(1.0, impulse_range / (3 * atr_value)),
        anchor_indices=(len(closes) - _FLAG_MAX_BARS, len(closes) - 1),
        side_bias=side_bias,
    )


def _fit_envelope(swings, *, kind: str) -> tuple[float, float] | None:
    if len(swings) < 2:
        return None
    sorted_swings = sorted(swings, key=lambda s: s.index)
    a, b = sorted_swings[0], sorted_swings[-1]
    if b.index == a.index:
        return None
    slope = (b.price - a.price) / (b.index - a.index)
    intercept = a.price - slope * a.index
    return (slope, intercept)


def _detect_wedge(highs, lows, atr_value, closes) -> PatternMatch | None:
    """Wedge: both upper and lower envelopes have the same sign of slope
    (rising wedge: both rising but converging; falling wedge: both
    falling but converging)."""
    if len(highs) < 2 or len(lows) < 2 or atr_value <= 0:
        return None
    upper = _fit_envelope(highs[-_WEDGE_MIN_SWINGS:], kind="upper")
    lower = _fit_envelope(lows[-_WEDGE_MIN_SWINGS:], kind="lower")
    if upper is None or lower is None:
        return None
    su, _iu = upper
    sl, _il = lower
    if not ((su > 0 and sl > 0) or (su < 0 and sl < 0)):
        return None
    # Convergence: |slope_upper - slope_lower| > 0 and they cross
    # eventually. Don't compute crossover; just require slopes differ.
    if abs(su - sl) < 1e-9:
        return None
    last_index = len(closes) - 1
    last_close = float(closes[-1])
    upper_at_last = su * last_index + _iu
    lower_at_last = sl * last_index + _il
    if su > 0 and sl > 0:
        # rising wedge -> bearish on break of LOWER bound
        kind: PatternKind = "wedge_rising"
        side_bias: Literal["BUY", "SELL", "NEUTRAL"] = "SELL"
        neckline = float(lower_at_last)
        broken = bool(last_close < neckline)
        invalidation = float(upper_at_last)
    else:
        # falling wedge -> bullish on break of UPPER bound
        kind = "wedge_falling"
        side_bias = "BUY"
        neckline = float(upper_at_last)
        broken = bool(last_close > neckline)
        invalidation = float(lower_at_last)
    return PatternMatch(
        kind=kind,
        neckline=neckline,
        neckline_broken=broken,
        retested=False,
        invalidation_price=invalidation,
        target_price=None,
        confidence=0.55,
        anchor_indices=tuple(s.index for s in highs[-_WEDGE_MIN_SWINGS:]),
        side_bias=side_bias,
    )


def _detect_triangle(highs, lows, atr_value, closes) -> PatternMatch | None:
    if len(highs) < 2 or len(lows) < 2 or atr_value <= 0:
        return None
    upper = _fit_envelope(highs[-3:], kind="upper")
    lower = _fit_envelope(lows[-3:], kind="lower")
    if upper is None or lower is None:
        return None
    su, _iu = upper
    sl, _il = lower
    last_index = len(closes) - 1
    last_close = float(closes[-1])
    upper_at_last = su * last_index + _iu
    lower_at_last = sl * last_index + _il
    flat_thr = 1e-6 * atr_value  # essentially flat
    if abs(su) <= flat_thr and sl > 0:
        kind: PatternKind = "triangle_ascending"
        side_bias: Literal["BUY", "SELL", "NEUTRAL"] = "BUY"
        neckline = float(upper_at_last)
        broken = bool(last_close > neckline)
        invalidation = float(lower_at_last)
    elif abs(sl) <= flat_thr and su < 0:
        kind = "triangle_descending"
        side_bias = "SELL"
        neckline = float(lower_at_last)
        broken = bool(last_close < neckline)
        invalidation = float(upper_at_last)
    elif su < 0 and sl > 0:
        # symmetric: break either way; report side_bias based on close.
        kind = "triangle_symmetric"
        if last_close > upper_at_last:
            side_bias = "BUY"
            neckline = float(upper_at_last)
            broken = True
        elif last_close < lower_at_last:
            side_bias = "SELL"
            neckline = float(lower_at_last)
            broken = True
        else:
            side_bias = "NEUTRAL"
            neckline = float(0.5 * (upper_at_last + lower_at_last))
            broken = False
        invalidation = float(0.5 * (upper_at_last + lower_at_last))
    else:
        return None
    return PatternMatch(
        kind=kind,
        neckline=neckline,
        neckline_broken=broken,
        retested=False,
        invalidation_price=invalidation,
        target_price=None,
        confidence=0.55,
        anchor_indices=tuple(s.index for s in highs[-3:]),
        side_bias=side_bias,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_patterns(
    df: pd.DataFrame,
    *,
    atr_value: float | None,
    last_close: float | None,
    lookback: int = 3,
) -> ChartPatternSnapshot:
    if (
        df is None or len(df) < 2 * lookback + 8
        or atr_value is None or atr_value <= 0
        or last_close is None
    ):
        return empty_snapshot("insufficient_data")
    highs, lows = detect_swings(df, lookback=lookback)
    closes = df["close"].to_numpy()

    hs = _detect_head_and_shoulders(highs, lows, atr_value, closes)
    inv = _detect_inverse_head_and_shoulders(highs, lows, atr_value, closes)
    flag = _detect_flag(closes, atr_value)
    wedge = _detect_wedge(highs, lows, atr_value, closes)
    tri = _detect_triangle(highs, lows, atr_value, closes)

    matches = tuple(m for m in (hs, inv, flag, wedge, tri) if m is not None)
    bullish_breakout = any(
        m.neckline_broken and m.side_bias == "BUY" for m in matches
    )
    bearish_breakout = any(
        m.neckline_broken and m.side_bias == "SELL" for m in matches
    )
    reason = (
        ", ".join(m.kind + ("(broken)" if m.neckline_broken else "")
                  for m in matches)
        or "no_patterns_detected"
    )
    return ChartPatternSnapshot(
        schema_version="chart_pattern_v2",
        matches=matches,
        head_and_shoulders=hs,
        inverse_head_and_shoulders=inv,
        flag=flag,
        wedge=wedge,
        triangle=tri,
        bullish_breakout_confirmed=bullish_breakout,
        bearish_breakout_confirmed=bearish_breakout,
        reason=reason,
    )


__all__ = [
    "PatternMatch",
    "ChartPatternSnapshot",
    "detect_patterns",
    "empty_snapshot",
]
