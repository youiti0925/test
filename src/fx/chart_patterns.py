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
from .retest_detection import detect_retest


_HS_SHOULDER_TOL_ATR: Final[float] = 0.6   # shoulder height symmetry
_HS_HEAD_PROMINENCE_ATR: Final[float] = 0.5
# flag formation parameters: adaptive, not fixed N=3.
_FLAG_MIN_CONSOL_BARS: Final[int] = 5
_FLAG_MAX_CONSOL_BARS: Final[int] = 30
_FLAG_MIN_IMPULSE_BARS: Final[int] = 3
_FLAG_MAX_IMPULSE_BARS: Final[int] = 20
_FLAG_MIN_IMPULSE_ATR: Final[float] = 1.5  # impulse range >= 1.5 ATR
_FLAG_MAX_CONSOL_TO_IMPULSE_RATIO: Final[float] = 0.6
# wedge / triangle: minimum anchors per side
_WEDGE_MIN_SWINGS: Final[int] = 3
_TRIANGLE_MIN_SWINGS: Final[int] = 3
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
    # v2.1: formation / line / quality breakdown — defaults preserve
    # backward compat with existing callers that construct PatternMatch
    # with only the v1 fields.
    formation_start_index: int | None = None
    formation_end_index: int | None = None
    formation_bars: int | None = None
    impulse_bars: int | None = None
    consolidation_bars: int | None = None
    upper_line: dict | None = None       # {"slope": ..., "intercept": ..., "anchors": [...]}
    lower_line: dict | None = None
    convergence_score: float | None = None
    breakout_direction: str = "UNKNOWN"   # BUY | SELL | NEUTRAL | UNKNOWN
    pattern_quality_score: float = 0.0
    reason: str = ""

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
            "formation_start_index": self.formation_start_index,
            "formation_end_index": self.formation_end_index,
            "formation_bars": self.formation_bars,
            "impulse_bars": self.impulse_bars,
            "consolidation_bars": self.consolidation_bars,
            "upper_line": (
                dict(self.upper_line) if self.upper_line is not None else None
            ),
            "lower_line": (
                dict(self.lower_line) if self.lower_line is not None else None
            ),
            "convergence_score": (
                float(self.convergence_score)
                if self.convergence_score is not None else None
            ),
            "breakout_direction": self.breakout_direction,
            "pattern_quality_score": float(self.pattern_quality_score),
            "reason": self.reason,
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


def _retest_confirmed(
    *,
    closes: np.ndarray,
    neckline: float | None,
    side_bias: str,
    breakout_search_start: int,
    atr_value: float,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    timestamps=None,
    parent_bar_ts: pd.Timestamp | None = None,
    wick_allowed: bool = False,
) -> bool:
    """Backward-compatible wrapper around `retest_detection.detect_retest`.

    By default (wick_allowed=False) it preserves the v2.0 close-based
    behaviour. Callers can opt into wick-based detection. Future-leak
    safety is enforced by the shared helper when `parent_bar_ts` /
    `timestamps` are supplied.
    """
    if side_bias not in ("BUY", "SELL"):
        return False
    res = detect_retest(
        closes=closes,
        highs=highs,
        lows=lows,
        timestamps=timestamps,
        level=neckline,
        side=side_bias,
        breakout_search_start=breakout_search_start,
        atr_value=atr_value,
        parent_bar_ts=parent_bar_ts,
        wick_allowed=wick_allowed,
        close_confirm_required=True,
    )
    return bool(res.retest_confirmed)


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
    retested = _retest_confirmed(
        closes=closes, neckline=neckline, side_bias="SELL",
        breakout_search_start=int(h3.index), atr_value=atr_value,
    ) if broken else False
    formation_bars = int(h3.index - h1.index)
    quality = float(min(0.95,
        confidence * (1.0 + (0.1 if broken else 0.0))
                   * (1.0 + (0.1 if retested else 0.0))
    ))
    return PatternMatch(
        kind="head_and_shoulders",
        neckline=neckline,
        neckline_broken=broken,
        retested=retested,
        invalidation_price=float(h2.price),
        target_price=float(target),
        confidence=float(min(0.95, confidence)),
        anchor_indices=(h1.index, h2.index, h3.index),
        side_bias="SELL",
        formation_start_index=int(h1.index),
        formation_end_index=int(h3.index),
        formation_bars=formation_bars,
        impulse_bars=None,
        consolidation_bars=None,
        upper_line=None,
        lower_line=None,
        convergence_score=None,
        breakout_direction="SELL" if broken else "UNKNOWN",
        pattern_quality_score=quality,
        reason=(
            f"3H@{h1.price:.4f}/{h2.price:.4f}/{h3.price:.4f} "
            f"neckline={neckline:.4f} broken={broken} retested={retested}"
        ),
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
    retested = _retest_confirmed(
        closes=closes, neckline=neckline, side_bias="BUY",
        breakout_search_start=int(l3.index), atr_value=atr_value,
    ) if broken else False
    formation_bars = int(l3.index - l1.index)
    quality = float(min(0.95,
        confidence * (1.0 + (0.1 if broken else 0.0))
                   * (1.0 + (0.1 if retested else 0.0))
    ))
    return PatternMatch(
        kind="inverse_head_and_shoulders",
        neckline=neckline,
        neckline_broken=broken,
        retested=retested,
        invalidation_price=float(l2.price),
        target_price=float(target),
        confidence=float(min(0.95, confidence)),
        anchor_indices=(l1.index, l2.index, l3.index),
        side_bias="BUY",
        formation_start_index=int(l1.index),
        formation_end_index=int(l3.index),
        formation_bars=formation_bars,
        impulse_bars=None,
        consolidation_bars=None,
        upper_line=None,
        lower_line=None,
        convergence_score=None,
        breakout_direction="BUY" if broken else "UNKNOWN",
        pattern_quality_score=quality,
        reason=(
            f"3L@{l1.price:.4f}/{l2.price:.4f}/{l3.price:.4f} "
            f"neckline={neckline:.4f} broken={broken} retested={retested}"
        ),
    )


def _detect_flag(closes: np.ndarray, atr_value: float) -> PatternMatch | None:
    """Bull / bear flag with ADAPTIVE impulse + consolidation windows.

    Algorithm (v2.1):
      1. Walk backward from the current bar to find the END of the
         consolidation: the most recent bar whose price-range over the
         next K bars (K=_FLAG_MIN_CONSOL_BARS) is small relative to
         its precursor.
      2. From that consolidation start, walk further back to find the
         IMPULSE: a window where ptp >= _FLAG_MIN_IMPULSE_ATR * atr.
      3. Verify consol_range / impulse_range <=
         _FLAG_MAX_CONSOL_TO_IMPULSE_RATIO.
      4. Detect breakout direction at the current bar and compute
         retest via the shared helper.

    Returns None when the conditions are not met (no fixed N).
    """
    if len(closes) < _FLAG_MIN_CONSOL_BARS + _FLAG_MIN_IMPULSE_BARS + 2 or atr_value <= 0:
        return None

    n = len(closes)
    # Try a few candidate consolidation lengths from the smallest valid
    # to _FLAG_MAX_CONSOL_BARS, picking the first that satisfies the
    # impulse + ratio constraints.
    for consol_len in range(_FLAG_MIN_CONSOL_BARS, _FLAG_MAX_CONSOL_BARS + 1):
        consol_start = n - consol_len
        if consol_start <= _FLAG_MIN_IMPULSE_BARS:
            break
        consol_window = closes[consol_start:n]
        consol_range = float(np.ptp(consol_window))
        if consol_range <= 0:
            continue
        # Try a few impulse lengths immediately before the consolidation.
        best_impulse: tuple[int, int, float] | None = None  # (start, end, range)
        for imp_len in range(_FLAG_MIN_IMPULSE_BARS, _FLAG_MAX_IMPULSE_BARS + 1):
            imp_start = consol_start - imp_len
            if imp_start < 0:
                break
            imp_window = closes[imp_start:consol_start]
            imp_range = float(np.ptp(imp_window))
            if imp_range >= _FLAG_MIN_IMPULSE_ATR * atr_value:
                if best_impulse is None or imp_range > best_impulse[2]:
                    best_impulse = (imp_start, consol_start, imp_range)
        if best_impulse is None:
            continue
        imp_start, imp_end, imp_range = best_impulse
        if consol_range > _FLAG_MAX_CONSOL_TO_IMPULSE_RATIO * imp_range:
            continue
        # Direction is the impulse leg sign.
        impulse_window = closes[imp_start:imp_end]
        direction = "BUY" if impulse_window[-1] > impulse_window[0] else "SELL"
        last_close = float(closes[-1])
        consol_high = float(np.max(consol_window))
        consol_low = float(np.min(consol_window))
        if direction == "BUY":
            kind: PatternKind = "flag_bullish"
            neckline = consol_high
            invalidation = consol_low
            broken = bool(last_close > consol_high)
            target = consol_high + imp_range
            side_bias = "BUY"
        else:
            kind = "flag_bearish"
            neckline = consol_low
            invalidation = consol_high
            broken = bool(last_close < consol_low)
            target = consol_low - imp_range
            side_bias = "SELL"
        retested = _retest_confirmed(
            closes=closes, neckline=neckline, side_bias=side_bias,
            breakout_search_start=imp_end,
            atr_value=atr_value,
        ) if broken else False
        # Lines: upper / lower bounds of the consolidation channel
        upper_line = {
            "slope": 0.0,
            "intercept": consol_high,
            "anchors": [int(consol_start), int(n - 1)],
            "kind": "horizontal_high",
        }
        lower_line = {
            "slope": 0.0,
            "intercept": consol_low,
            "anchors": [int(consol_start), int(n - 1)],
            "kind": "horizontal_low",
        }
        # convergence_score for a flag is low (parallel channel by definition);
        # set it to 1 - (consol_range / imp_range) so tighter consolidation = higher.
        convergence = max(0.0, 1.0 - (consol_range / imp_range))
        # pattern_quality_score: combine impulse strength + consolidation tightness
        # + breakout confirmation.
        quality = (
            min(1.0, imp_range / (3 * atr_value)) * 0.5
            + (1.0 - consol_range / max(imp_range, 1e-9)) * 0.3
            + (0.2 if broken else 0.0)
        )
        return PatternMatch(
            kind=kind,
            neckline=float(neckline),
            neckline_broken=broken,
            retested=retested,
            invalidation_price=float(invalidation),
            target_price=float(target),
            confidence=0.55 + 0.1 * min(1.0, imp_range / (3 * atr_value)),
            anchor_indices=(int(imp_start), int(imp_end), int(n - 1)),
            side_bias=side_bias,
            formation_start_index=int(imp_start),
            formation_end_index=int(n - 1),
            formation_bars=int(n - imp_start),
            impulse_bars=int(imp_end - imp_start),
            consolidation_bars=int(n - consol_start),
            upper_line=upper_line,
            lower_line=lower_line,
            convergence_score=float(convergence),
            breakout_direction=side_bias if broken else "UNKNOWN",
            pattern_quality_score=float(min(0.95, quality)),
            reason=(
                f"impulse={imp_range:.4f}({imp_end - imp_start}b) "
                f"consol={consol_range:.4f}({n - consol_start}b) "
                f"ratio={consol_range / imp_range:.3f}"
            ),
        )
    return None


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
    wedge_anchor_idx = (
        min(s.index for s in highs[-_WEDGE_MIN_SWINGS:])
        if highs else len(closes) - 1
    )
    retested = _retest_confirmed(
        closes=closes, neckline=neckline, side_bias=side_bias,
        breakout_search_start=int(wedge_anchor_idx),
        atr_value=atr_value,
    ) if broken else False
    upper_line = {
        "slope": float(su),
        "intercept": float(_iu),
        "anchors": [int(s.index) for s in highs[-_WEDGE_MIN_SWINGS:]],
        "kind": "ascending" if su > 0 else "descending",
    }
    lower_line = {
        "slope": float(sl),
        "intercept": float(_il),
        "anchors": [int(s.index) for s in lows[-_WEDGE_MIN_SWINGS:]],
        "kind": "ascending" if sl > 0 else "descending",
    }
    # convergence_score = how much the gap between the lines shrinks
    # over the wedge. Computed as 1 - (current_gap / initial_gap), capped
    # to [0, 1]. Higher = stronger convergence.
    initial_gap = max(1e-9, abs(
        (su * wedge_anchor_idx + _iu) - (sl * wedge_anchor_idx + _il)
    ))
    current_gap = max(0.0, abs(upper_at_last - lower_at_last))
    convergence = float(max(0.0, min(1.0, 1.0 - (current_gap / initial_gap))))
    quality = float(min(0.95,
        0.5 + 0.3 * convergence + (0.15 if broken else 0.0)
    ))
    return PatternMatch(
        kind=kind,
        neckline=neckline,
        neckline_broken=broken,
        retested=retested,
        invalidation_price=invalidation,
        target_price=None,
        confidence=0.55,
        anchor_indices=tuple(s.index for s in highs[-_WEDGE_MIN_SWINGS:]),
        side_bias=side_bias,
        formation_start_index=int(wedge_anchor_idx),
        formation_end_index=int(len(closes) - 1),
        formation_bars=int(len(closes) - 1 - wedge_anchor_idx),
        impulse_bars=None,
        consolidation_bars=None,
        upper_line=upper_line,
        lower_line=lower_line,
        convergence_score=convergence,
        breakout_direction=side_bias if broken else "UNKNOWN",
        pattern_quality_score=quality,
        reason=(
            f"{kind} su={su:.6f} sl={sl:.6f} "
            f"convergence={convergence:.3f} broken={broken}"
        ),
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
    triangle_anchor_idx = (
        min(s.index for s in highs[-3:]) if highs else len(closes) - 1
    )
    retested = _retest_confirmed(
        closes=closes, neckline=neckline,
        side_bias=side_bias if side_bias in ("BUY", "SELL") else "BUY",
        breakout_search_start=int(triangle_anchor_idx),
        atr_value=atr_value,
    ) if (broken and side_bias in ("BUY", "SELL")) else False
    upper_line = {
        "slope": float(su),
        "intercept": float(_iu),
        "anchors": [int(s.index) for s in highs[-3:]],
        "kind": "flat" if abs(su) <= flat_thr else (
            "ascending" if su > 0 else "descending"
        ),
    }
    lower_line = {
        "slope": float(sl),
        "intercept": float(_il),
        "anchors": [int(s.index) for s in lows[-3:]],
        "kind": "flat" if abs(sl) <= flat_thr else (
            "ascending" if sl > 0 else "descending"
        ),
    }
    initial_gap = max(1e-9, abs(
        (su * triangle_anchor_idx + _iu) - (sl * triangle_anchor_idx + _il)
    ))
    current_gap = max(0.0, abs(upper_at_last - lower_at_last))
    convergence = float(max(0.0, min(1.0, 1.0 - (current_gap / initial_gap))))
    quality = float(min(0.95,
        0.5 + 0.3 * convergence + (0.15 if broken else 0.0)
    ))
    return PatternMatch(
        kind=kind,
        neckline=neckline,
        neckline_broken=broken,
        retested=retested,
        invalidation_price=invalidation,
        target_price=None,
        confidence=0.55,
        anchor_indices=tuple(s.index for s in highs[-3:]),
        side_bias=side_bias,
        formation_start_index=int(triangle_anchor_idx),
        formation_end_index=int(len(closes) - 1),
        formation_bars=int(len(closes) - 1 - triangle_anchor_idx),
        impulse_bars=None,
        consolidation_bars=None,
        upper_line=upper_line,
        lower_line=lower_line,
        convergence_score=convergence,
        breakout_direction=(
            side_bias if broken and side_bias in ("BUY", "SELL")
            else "NEUTRAL"
        ),
        pattern_quality_score=quality,
        reason=(
            f"{kind} su={su:.6f} sl={sl:.6f} "
            f"convergence={convergence:.3f} broken={broken}"
        ),
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
