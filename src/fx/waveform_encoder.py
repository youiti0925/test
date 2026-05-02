"""Waveform skeleton encoder (observation-only).

Converts an OHLC visible window into a normalised "waveform skeleton" —
a small ordered list of pivot highs/lows plus a 0..1 normalised polyline
suitable for template matching (`pattern_shape_matcher`).

This is observation-only. It does NOT feed `royal_road_decision_v2`'s
final BUY/SELL/HOLD logic. The output is consumed by:

  - `chart_reconstruction.reconstruct_chart_multi_scale` (per scale)
  - `visual_audit` (waveform review section + pattern dissection)

Future-leak rule
----------------
The caller passes a visible df (already truncated to bars whose
ts ≤ parent_bar_ts). This module never reads beyond `df.iloc[-1]`.
The latest "developing" pivot, if reported, is explicitly tagged
`source="developing"` so consumers know it is not yet confirmed and
can be ignored for hard-gate decisions.

Heuristic parameters (initial defaults)
---------------------------------------
- min_reversal_atr   = 0.8   ZigZag reversal threshold in ATR multiples
- min_pivot_gap_bars = 3     Minimum bars between consecutive pivots
- max_pivots         = 12    Cap on retained pivots (most-recent biased)
- resample_points    = 64    Length of normalised polyline

These are heuristics. They have NOT been validated against multi-symbol
multi-period data; the reason field flags this so downstream callers
can filter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable

import numpy as np
import pandas as pd

from .patterns import detect_swings


SCHEMA_VERSION: Final[str] = "wave_skeleton_v1"

DEFAULT_MIN_REVERSAL_ATR: Final[float] = 0.8
DEFAULT_MIN_PIVOT_GAP_BARS: Final[int] = 3
DEFAULT_MAX_PIVOTS: Final[int] = 12
DEFAULT_RESAMPLE_POINTS: Final[int] = 64


@dataclass(frozen=True)
class WavePivot:
    """One pivot on the skeleton.

    `kind`:    "H" (high) or "L" (low)
    `source`:  "swing"        — confirmed by detect_swings
               "zigzag"       — promoted to skeleton pivot after
                                ATR reversal filter
               "developing"   — tentative latest extreme since the
                                last confirmed pivot. NOT confirmed.
    `strength`: 0..1 prominence proxy (relative reversal magnitude
                versus median pivot reversal in the same skeleton).
    """

    index: int
    ts: str
    price: float
    kind: str
    source: str
    strength: float

    def to_dict(self) -> dict:
        return {
            "index": int(self.index),
            "ts": str(self.ts),
            "price": float(self.price),
            "kind": str(self.kind),
            "source": str(self.source),
            "strength": float(self.strength),
        }


@dataclass(frozen=True)
class WaveSkeleton:
    """Normalised waveform skeleton.

    `normalized_points` is a tuple of (x, y) pairs in [0, 1] x [0, 1].
    `x` is the linear interpolation of pivot `index` between the first
    and last pivot index; `y` is `(price - price_min) / (price_max -
    price_min)` so 0 = lowest pivot price, 1 = highest pivot price.

    For pure-flat skeletons (price_max == price_min), `y` is 0.5
    everywhere.
    """

    schema_version: str
    scale: str
    bars_used: int
    pivots: tuple[WavePivot, ...]
    normalized_points: tuple[tuple[float, float], ...]
    price_min: float
    price_max: float
    atr_value: float | None
    trend_hint: str
    reason: str
    # Initial heuristic parameters used to build this skeleton, copied
    # so downstream audit consumers can show the user what thresholds
    # produced this view.
    params: dict

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "scale": self.scale,
            "bars_used": int(self.bars_used),
            "pivots": [p.to_dict() for p in self.pivots],
            "normalized_points": [
                [float(x), float(y)] for x, y in self.normalized_points
            ],
            "price_min": float(self.price_min),
            "price_max": float(self.price_max),
            "atr_value": (
                float(self.atr_value) if self.atr_value is not None else None
            ),
            "trend_hint": str(self.trend_hint),
            "reason": str(self.reason),
            "params": dict(self.params),
        }


def empty_skeleton(scale: str, reason: str) -> WaveSkeleton:
    return WaveSkeleton(
        schema_version=SCHEMA_VERSION,
        scale=scale,
        bars_used=0,
        pivots=(),
        normalized_points=(),
        price_min=0.0,
        price_max=0.0,
        atr_value=None,
        trend_hint="UNKNOWN",
        reason=reason,
        params={},
    )


def _ts_to_str(ts) -> str:
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def _zigzag_filter(
    swings_chronological: list,  # list of patterns.Swing
    *,
    atr_value: float,
    min_reversal_atr: float,
    min_pivot_gap_bars: int,
) -> list:
    """Drop swings whose reversal magnitude vs the previously-kept
    swing is below `min_reversal_atr * atr` OR which sit within
    `min_pivot_gap_bars` of the previous kept swing.

    Operates left-to-right on a chronologically merged sequence of
    confirmed swings (highs and lows). Always keeps the first swing.
    Output preserves chronological order.

    The filter intentionally does NOT consult bars later than
    swings_chronological[-1].index, so passing
    detect_swings(visible_df) keeps the future-leak guarantee.
    """
    if not swings_chronological:
        return []
    threshold = max(0.0, float(min_reversal_atr) * float(atr_value))
    kept: list = [swings_chronological[0]]
    for sw in swings_chronological[1:]:
        prev = kept[-1]
        gap = int(sw.index) - int(prev.index)
        if gap < int(min_pivot_gap_bars):
            # Replace the previous if the same kind and more extreme,
            # otherwise drop.
            if sw.kind == prev.kind:
                more_extreme = (
                    (sw.kind == "H" and sw.price > prev.price)
                    or (sw.kind == "L" and sw.price < prev.price)
                )
                if more_extreme:
                    kept[-1] = sw
            continue
        reversal = abs(float(sw.price) - float(prev.price))
        if reversal < threshold and sw.kind != prev.kind:
            # too small a reversal — skip
            continue
        if sw.kind == prev.kind:
            # consecutive same-kind swings: keep the more extreme one
            more_extreme = (
                (sw.kind == "H" and sw.price > prev.price)
                or (sw.kind == "L" and sw.price < prev.price)
            )
            if more_extreme:
                kept[-1] = sw
            continue
        kept.append(sw)
    return kept


def _atr_proxy(df: pd.DataFrame, period: int = 14) -> float | None:
    if df is None or len(df) < period + 1:
        return None
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    if tr.size < period:
        return None
    # Simple rolling mean — matches existing patterns.py prominence proxy
    # closely enough for this heuristic encoder.
    atr_arr = pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()
    last = atr_arr[-1]
    if not np.isfinite(last) or last <= 0:
        return None
    return float(last)


def _trend_hint(pivots: list[WavePivot]) -> str:
    """Coarse HH/HL/LH/LL count over the last 4 pivots."""
    if len(pivots) < 2:
        return "UNKNOWN"
    last_high: float | None = None
    last_low: float | None = None
    seq: list[str] = []
    for p in pivots:
        if p.kind == "H":
            if last_high is not None:
                seq.append("HH" if p.price > last_high else "LH")
            last_high = p.price
        else:
            if last_low is not None:
                seq.append("HL" if p.price > last_low else "LL")
            last_low = p.price
    tail = seq[-4:]
    if not tail:
        return "UNKNOWN"
    up = sum(1 for s in tail if s in ("HH", "HL"))
    down = sum(1 for s in tail if s in ("LL", "LH"))
    if up >= 3 and down <= 1:
        return "UP"
    if down >= 3 and up <= 1:
        return "DOWN"
    if up >= 2 and down >= 2:
        return "MIXED"
    return "RANGE"


def _resample(points: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Resample a polyline (x, y) to exactly `n` evenly-spaced points
    along its x dimension. `points` must be sorted by x ascending.
    """
    if not points or n <= 0:
        return []
    if len(points) == 1:
        return [(points[0][0], points[0][1]) for _ in range(n)]
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    target = np.linspace(xs[0], xs[-1], n)
    interp = np.interp(target, xs, ys)
    return [(float(t), float(v)) for t, v in zip(target, interp)]


def _developing_pivot(
    df: pd.DataFrame,
    last_kept,
    atr_value: float,
    *,
    min_reversal_atr: float,
) -> WavePivot | None:
    """If the most-recent unconfirmed bar represents a tentative
    extreme since the last confirmed pivot, return it as a "developing"
    WavePivot. Otherwise None.

    A developing high is reported when the post-pivot max(high) exceeds
    `last_kept.price + min_reversal_atr * atr` (last pivot was a low
    → developing high). Symmetrically for developing low.
    """
    if last_kept is None or atr_value <= 0:
        return None
    after = df.iloc[int(last_kept.index) + 1:]
    if len(after) == 0:
        return None
    threshold = float(min_reversal_atr) * float(atr_value)
    if last_kept.kind == "L":
        post_high = after["high"].max()
        if not np.isfinite(post_high):
            return None
        if post_high - last_kept.price < threshold:
            return None
        idx_local = after["high"].idxmax()
        full_idx = df.index.get_loc(idx_local)
        return WavePivot(
            index=int(full_idx),
            ts=_ts_to_str(idx_local),
            price=float(post_high),
            kind="H",
            source="developing",
            strength=0.3,
        )
    else:
        post_low = after["low"].min()
        if not np.isfinite(post_low):
            return None
        if last_kept.price - post_low < threshold:
            return None
        idx_local = after["low"].idxmin()
        full_idx = df.index.get_loc(idx_local)
        return WavePivot(
            index=int(full_idx),
            ts=_ts_to_str(idx_local),
            price=float(post_low),
            kind="L",
            source="developing",
            strength=0.3,
        )


def encode_wave_skeleton(
    df: pd.DataFrame | None,
    *,
    scale: str,
    atr_value: float | None = None,
    min_reversal_atr: float = DEFAULT_MIN_REVERSAL_ATR,
    min_pivot_gap_bars: int = DEFAULT_MIN_PIVOT_GAP_BARS,
    max_pivots: int = DEFAULT_MAX_PIVOTS,
    resample_points: int = DEFAULT_RESAMPLE_POINTS,
    include_developing: bool = True,
    swing_lookback: int = 3,
) -> WaveSkeleton:
    """Encode `df` (visible window) into a wave skeleton.

    `df` must already be truncated to bars with ts ≤ parent_bar_ts.
    No bar at index > len(df) - 1 is consulted.

    Returns an empty skeleton (with `reason` populated) when the input
    is too short to extract two distinct pivots.
    """
    params = {
        "min_reversal_atr": float(min_reversal_atr),
        "min_pivot_gap_bars": int(min_pivot_gap_bars),
        "max_pivots": int(max_pivots),
        "resample_points": int(resample_points),
        "swing_lookback": int(swing_lookback),
        "include_developing": bool(include_developing),
        "thresholds_status": "heuristic_not_validated",
    }
    if df is None or len(df) == 0:
        return WaveSkeleton(
            schema_version=SCHEMA_VERSION, scale=scale, bars_used=0,
            pivots=(), normalized_points=(),
            price_min=0.0, price_max=0.0,
            atr_value=None, trend_hint="UNKNOWN",
            reason="empty_input", params=params,
        )

    bars_used = int(len(df))
    atr = (
        float(atr_value)
        if (atr_value is not None and atr_value > 0)
        else _atr_proxy(df)
    )
    if atr is None or atr <= 0:
        return WaveSkeleton(
            schema_version=SCHEMA_VERSION, scale=scale, bars_used=bars_used,
            pivots=(), normalized_points=(),
            price_min=0.0, price_max=0.0,
            atr_value=atr, trend_hint="UNKNOWN",
            reason="atr_unavailable", params=params,
        )

    highs, lows = detect_swings(df, lookback=swing_lookback)
    merged = sorted(highs + lows, key=lambda s: s.index)
    if len(merged) < 2:
        return WaveSkeleton(
            schema_version=SCHEMA_VERSION, scale=scale, bars_used=bars_used,
            pivots=(), normalized_points=(),
            price_min=0.0, price_max=0.0,
            atr_value=atr, trend_hint="UNKNOWN",
            reason="too_few_swings", params=params,
        )

    kept = _zigzag_filter(
        merged,
        atr_value=atr,
        min_reversal_atr=min_reversal_atr,
        min_pivot_gap_bars=min_pivot_gap_bars,
    )
    # Cap to most-recent max_pivots
    if len(kept) > max_pivots:
        kept = kept[-max_pivots:]

    if len(kept) < 2:
        return WaveSkeleton(
            schema_version=SCHEMA_VERSION, scale=scale, bars_used=bars_used,
            pivots=(), normalized_points=(),
            price_min=0.0, price_max=0.0,
            atr_value=atr, trend_hint="UNKNOWN",
            reason="zigzag_filter_too_aggressive", params=params,
        )

    # Compute strength as relative reversal vs median reversal
    reversals = [
        abs(float(kept[i].price) - float(kept[i - 1].price))
        for i in range(1, len(kept))
    ]
    median_rev = float(np.median(reversals)) if reversals else 0.0
    pivots: list[WavePivot] = []
    for sw in kept:
        # last reversal magnitude vs previous kept pivot
        if pivots:
            r = abs(float(sw.price) - float(pivots[-1].price))
            strength = 1.0 if median_rev <= 0 else float(min(1.0, r / (2 * median_rev)))
        else:
            strength = 0.5
        pivots.append(WavePivot(
            index=int(sw.index),
            ts=_ts_to_str(sw.ts),
            price=float(sw.price),
            kind=str(sw.kind),
            source="zigzag",
            strength=float(strength),
        ))

    if include_developing:
        dev = _developing_pivot(
            df, kept[-1], atr,
            min_reversal_atr=min_reversal_atr,
        )
        if dev is not None and dev.kind != pivots[-1].kind:
            pivots.append(dev)

    prices = [p.price for p in pivots]
    p_min = float(min(prices))
    p_max = float(max(prices))
    if p_max <= p_min:
        # degenerate; flatten y
        normalized = []
        n = len(pivots)
        for i, p in enumerate(pivots):
            x = 0.0 if n == 1 else i / (n - 1)
            normalized.append((float(x), 0.5))
    else:
        idx_min = pivots[0].index
        idx_max = pivots[-1].index
        x_span = max(1, idx_max - idx_min)
        raw = []
        for p in pivots:
            x = (p.index - idx_min) / x_span
            y = (p.price - p_min) / (p_max - p_min)
            raw.append((float(x), float(y)))
        normalized = _resample(raw, resample_points)

    trend = _trend_hint(pivots)

    return WaveSkeleton(
        schema_version=SCHEMA_VERSION,
        scale=scale,
        bars_used=bars_used,
        pivots=tuple(pivots),
        normalized_points=tuple(normalized),
        price_min=p_min,
        price_max=p_max,
        atr_value=atr,
        trend_hint=trend,
        reason="ok",
        params=params,
    )


__all__ = [
    "SCHEMA_VERSION",
    "WavePivot",
    "WaveSkeleton",
    "empty_skeleton",
    "encode_wave_skeleton",
    "DEFAULT_MIN_REVERSAL_ATR",
    "DEFAULT_MIN_PIVOT_GAP_BARS",
    "DEFAULT_MAX_PIVOTS",
    "DEFAULT_RESAMPLE_POINTS",
]
