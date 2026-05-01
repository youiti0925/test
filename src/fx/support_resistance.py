"""Horizontal support / resistance detection via swing clustering.

Used by royal_road_decision_v2 (NOT by current_runtime, NOT by v1).
Reads `patterns.detect_swings()` output (which is future-leak-safe by
construction) and clusters confirmed swings into level objects with
touch counts, recency weights, and strength scores.

Future-leak rule (matches `patterns.py`)
----------------------------------------
A swing at index `i` is only confirmed once `i + lookback` more bars
have closed. detect_swings already enforces this. Clustering only
adds aggregation — it never reads bars to the right of `len(df) - 1`.

Strength score (heuristic, NOT validated)
-----------------------------------------
strength_score = log(1 + touch_count) * recency_weight * (1 + role_reversal_count*0.1)
recency_weight = exp(-bars_since_last_touch / half_life_bars)

This is a research signal, not a final trading rule. Adoption requires
multi-symbol × multi-period evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal

import numpy as np
import pandas as pd

from .patterns import Swing, detect_swings


# Heuristic constants — observation grade, NOT in PARAMETER_BASELINE_V1.
_CLUSTER_BUCKET_ATR: Final[float] = 0.5    # swings within 0.5 ATR cluster together
_NEAR_LEVEL_ATR: Final[float] = 0.5         # within 0.5 ATR of close = "near"
_STRONG_TOUCH_THRESHOLD: Final[int] = 3     # >=3 touches = "strong"
_RECENCY_HALF_LIFE_BARS: Final[float] = 200.0
_FALSE_BREAKOUT_LOOKBACK_BARS: Final[int] = 5
# Per-level history scan: a "side flip" is counted when the close
# crosses the level price between consecutive bars by more than this
# tolerance band (ATR-relative). This filters out tick-level noise
# at the level price.
_LEVEL_FLIP_TOLERANCE_ATR: Final[float] = 0.05
# A "false breakout" is a side flip that reverts (flips back) within
# this many bars.
_LEVEL_FALSE_BREAKOUT_RETURN_BARS: Final[int] = 5


LevelKind = Literal["support", "resistance", "both"]


@dataclass(frozen=True)
class Level:
    price: float
    kind: LevelKind
    touch_count: int
    first_touch_ts: pd.Timestamp | None
    last_touch_ts: pd.Timestamp | None
    first_touch_index: int
    last_touch_index: int
    broken_count: int
    role_reversal_count: int
    false_breakout_count: int
    strength_score: float
    recency_score: float
    distance_to_close_atr: float | None

    def to_dict(self) -> dict:
        return {
            "price": float(self.price),
            "kind": self.kind,
            "touch_count": int(self.touch_count),
            "first_touch_ts": (
                self.first_touch_ts.isoformat()
                if isinstance(self.first_touch_ts, pd.Timestamp)
                else self.first_touch_ts
            ),
            "last_touch_ts": (
                self.last_touch_ts.isoformat()
                if isinstance(self.last_touch_ts, pd.Timestamp)
                else self.last_touch_ts
            ),
            "first_touch_index": int(self.first_touch_index),
            "last_touch_index": int(self.last_touch_index),
            "broken_count": int(self.broken_count),
            "role_reversal_count": int(self.role_reversal_count),
            "false_breakout_count": int(self.false_breakout_count),
            "strength_score": float(self.strength_score),
            "recency_score": float(self.recency_score),
            "distance_to_close_atr": (
                float(self.distance_to_close_atr)
                if self.distance_to_close_atr is not None else None
            ),
        }


@dataclass(frozen=True)
class SRSnapshot:
    """Per-bar S/R observation for trace consumption."""

    levels: tuple[Level, ...]
    nearest_support: Level | None
    nearest_resistance: Level | None
    near_strong_support: bool
    near_strong_resistance: bool
    breakout: bool
    pullback: bool
    role_reversal: bool
    fake_breakout: bool
    reason: str
    schema_version: str = "support_resistance_v2"

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "levels": [lvl.to_dict() for lvl in self.levels],
            "nearest_support": (
                self.nearest_support.to_dict()
                if self.nearest_support else None
            ),
            "nearest_resistance": (
                self.nearest_resistance.to_dict()
                if self.nearest_resistance else None
            ),
            "near_strong_support": bool(self.near_strong_support),
            "near_strong_resistance": bool(self.near_strong_resistance),
            "breakout": bool(self.breakout),
            "pullback": bool(self.pullback),
            "role_reversal": bool(self.role_reversal),
            "fake_breakout": bool(self.fake_breakout),
            "reason": self.reason,
        }


def empty_snapshot(reason: str = "insufficient_data") -> SRSnapshot:
    return SRSnapshot(
        levels=(),
        nearest_support=None,
        nearest_resistance=None,
        near_strong_support=False,
        near_strong_resistance=False,
        breakout=False,
        pullback=False,
        role_reversal=False,
        fake_breakout=False,
        reason=reason,
    )


def _level_history_counts(
    *,
    closes: np.ndarray,
    level_price: float,
    atr_value: float,
    first_touch_index: int,
    last_touch_index: int,
) -> tuple[int, int, int]:
    """Walk visible history to count side flips at the level price.

    Returns (broken_count, false_breakout_count, role_reversal_count).

    Definitions:
      - side(i): "above" if closes[i] > level + tol; "below" if
        closes[i] < level - tol; "at" otherwise. Tol = `_LEVEL_FLIP_TOLERANCE_ATR
        * atr_value`.
      - broken_count: number of times consecutive non-"at" sides
        differ over the scan window.
      - false_breakout_count: a flip that returns to the original
        side within `_LEVEL_FALSE_BREAKOUT_RETURN_BARS` bars.
      - role_reversal_count: flips that did NOT revert within the
        window — the level genuinely changed role and stayed there.

    Future-leak safe: only `closes` (the visible history through the
    current bar) is consulted. The first_touch_index marker means the
    scan starts at or after that index — earlier bars cannot have
    "broken" the level since the level didn't exist yet.
    """
    if len(closes) < 2 or atr_value <= 0:
        return (0, 0, 0)
    tol = max(_LEVEL_FLIP_TOLERANCE_ATR * atr_value, 1e-12)
    start = max(0, int(first_touch_index))
    if start >= len(closes) - 1:
        return (0, 0, 0)
    # Pre-compute side labels for the scan window.
    sides: list[str] = []
    for i in range(start, len(closes)):
        c = float(closes[i])
        if c > level_price + tol:
            sides.append("above")
        elif c < level_price - tol:
            sides.append("below")
        else:
            sides.append("at")

    # Walk consecutive non-"at" sides and count flips + reversions.
    broken = 0
    false_breakouts = 0
    role_reversals = 0
    last_side: str | None = None
    flip_open_at: int | None = None  # index (within `sides`) where last flip occurred
    flip_open_from: str | None = None  # the side we left from

    for idx, s in enumerate(sides):
        if s == "at":
            continue
        if last_side is None:
            last_side = s
            continue
        if s != last_side:
            broken += 1
            # Check if this flip closes a previously open flip (i.e.
            # we're returning to flip_open_from within the window).
            if (
                flip_open_at is not None
                and flip_open_from is not None
                and s == flip_open_from
                and (idx - flip_open_at) <= _LEVEL_FALSE_BREAKOUT_RETURN_BARS
            ):
                false_breakouts += 1
                flip_open_at = None
                flip_open_from = None
            else:
                # If a previous flip was still open and exceeded the
                # return window, count it as a genuine reversal.
                if flip_open_at is not None and flip_open_from is not None:
                    if (idx - flip_open_at) > _LEVEL_FALSE_BREAKOUT_RETURN_BARS:
                        role_reversals += 1
                flip_open_at = idx
                flip_open_from = last_side
            last_side = s
    # Tail: if a flip is still open at the end of the scan window AND
    # it has been open for more than the return window, count it as
    # a confirmed role reversal.
    if (
        flip_open_at is not None
        and flip_open_from is not None
        and (len(sides) - 1 - flip_open_at) > _LEVEL_FALSE_BREAKOUT_RETURN_BARS
    ):
        role_reversals += 1
    return (broken, false_breakouts, role_reversals)


def _cluster_swings(
    swings: list[Swing],
    *,
    bucket_size: float,
    n_bars: int,
) -> list[Level]:
    """Greedy 1-D clustering by price proximity.

    Sort swings by price, walk linearly, and join any two consecutive
    swings whose price gap is <= bucket_size. Each cluster becomes one
    Level. The kind ("support" / "resistance" / "both") is derived from
    the swing kinds inside the cluster — a cluster of only swing-lows
    is "support", only swing-highs is "resistance", mixed is "both".
    """
    if not swings or bucket_size <= 0:
        return []
    sorted_swings = sorted(swings, key=lambda s: s.price)
    clusters: list[list[Swing]] = [[sorted_swings[0]]]
    for sw in sorted_swings[1:]:
        if abs(sw.price - clusters[-1][-1].price) <= bucket_size:
            clusters[-1].append(sw)
        else:
            clusters.append([sw])

    levels: list[Level] = []
    for cluster in clusters:
        prices = [s.price for s in cluster]
        kinds = {s.kind for s in cluster}
        kind: LevelKind
        if kinds == {"L"}:
            kind = "support"
        elif kinds == {"H"}:
            kind = "resistance"
        else:
            kind = "both"
        first_touch = min(cluster, key=lambda s: s.index)
        last_touch = max(cluster, key=lambda s: s.index)
        bars_since_last = max(0, n_bars - 1 - last_touch.index)
        recency_score = math.exp(-bars_since_last / _RECENCY_HALF_LIFE_BARS)
        # broken / false_breakout / role_reversal counts are computed
        # in detect_levels via _level_history_counts (needs closes
        # + atr_value not available here). Initialised to 0.
        levels.append(Level(
            price=float(np.mean(prices)),
            kind=kind,
            touch_count=len(cluster),
            first_touch_ts=getattr(first_touch, "ts", None),
            last_touch_ts=getattr(last_touch, "ts", None),
            first_touch_index=int(first_touch.index),
            last_touch_index=int(last_touch.index),
            broken_count=0,
            role_reversal_count=0,
            false_breakout_count=0,
            strength_score=0.0,            # filled in detect_levels
            recency_score=float(recency_score),
            distance_to_close_atr=None,    # filled in detect_levels
        ))
    return levels


def detect_levels(
    df: pd.DataFrame,
    *,
    atr_value: float | None,
    last_close: float | None,
    lookback: int = 3,
) -> SRSnapshot:
    """Build an `SRSnapshot` for the current bar.

    Required: `df` is the OHLC window up through the current bar
    (inclusive), `atr_value` is the current ATR, `last_close` is the
    most recent close. Both must be non-None for clustering to fire.
    """
    if (
        df is None or len(df) < 2 * lookback + 5
        or atr_value is None or atr_value <= 0
        or last_close is None
    ):
        return empty_snapshot("insufficient_data")

    highs, lows = detect_swings(df, lookback=lookback)
    swings = highs + lows
    if not swings:
        return empty_snapshot("no_swings_detected")

    bucket_size = _CLUSTER_BUCKET_ATR * atr_value
    levels = _cluster_swings(swings, bucket_size=bucket_size, n_bars=len(df))

    if not levels:
        return empty_snapshot("clustering_produced_no_levels")

    closes = df["close"].to_numpy()
    # Compute distance_to_close_atr + per-level history counts +
    # final strength_score per level (counts feed back into score).
    enriched: list[Level] = []
    for lvl in levels:
        dist_atr = abs(last_close - lvl.price) / atr_value
        broken, false_brk, role_rev = _level_history_counts(
            closes=closes,
            level_price=lvl.price,
            atr_value=atr_value,
            first_touch_index=lvl.first_touch_index,
            last_touch_index=lvl.last_touch_index,
        )
        # strength_score: log(touches) × recency
        # × (1 + role_reversal_count × 0.1)             ← genuine flips raise
        # × max(0.1, 1 − 0.15 × false_breakouts)        ← false breaks lower
        # × max(0.2, 1 − 0.05 × max(0, broken − role_rev))  ← raw breaks lower
        score = (
            math.log1p(lvl.touch_count)
            * lvl.recency_score
            * (1.0 + role_rev * 0.1)
            * max(0.1, 1.0 - 0.15 * false_brk)
            * max(0.2, 1.0 - 0.05 * max(0, broken - role_rev))
        )
        enriched.append(Level(
            price=lvl.price,
            kind=lvl.kind,
            touch_count=lvl.touch_count,
            first_touch_ts=lvl.first_touch_ts,
            last_touch_ts=lvl.last_touch_ts,
            first_touch_index=lvl.first_touch_index,
            last_touch_index=lvl.last_touch_index,
            broken_count=int(broken),
            role_reversal_count=int(role_rev),
            false_breakout_count=int(false_brk),
            strength_score=float(score),
            recency_score=lvl.recency_score,
            distance_to_close_atr=dist_atr,
        ))
    levels = enriched

    # Identify nearest support/resistance relative to current close.
    supports = [
        lvl for lvl in levels
        if lvl.kind in ("support", "both") and lvl.price < last_close
    ]
    resistances = [
        lvl for lvl in levels
        if lvl.kind in ("resistance", "both") and lvl.price > last_close
    ]
    nearest_support = (
        max(supports, key=lambda l: l.price) if supports else None
    )
    nearest_resistance = (
        min(resistances, key=lambda l: l.price) if resistances else None
    )

    near_strong_support = bool(
        nearest_support is not None
        and nearest_support.distance_to_close_atr is not None
        and nearest_support.distance_to_close_atr <= _NEAR_LEVEL_ATR
        and nearest_support.touch_count >= _STRONG_TOUCH_THRESHOLD
    )
    near_strong_resistance = bool(
        nearest_resistance is not None
        and nearest_resistance.distance_to_close_atr is not None
        and nearest_resistance.distance_to_close_atr <= _NEAR_LEVEL_ATR
        and nearest_resistance.touch_count >= _STRONG_TOUCH_THRESHOLD
    )

    # Bar-level events: breakout / pullback / role_reversal / fake_breakout
    closes = df["close"].to_numpy()
    breakout, pullback, role_reversal, fake_breakout, reason = (
        _bar_level_events(
            levels=levels, closes=closes, atr_value=atr_value,
            last_close=last_close,
        )
    )

    return SRSnapshot(
        levels=tuple(levels),
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        near_strong_support=near_strong_support,
        near_strong_resistance=near_strong_resistance,
        breakout=breakout,
        pullback=pullback,
        role_reversal=role_reversal,
        fake_breakout=fake_breakout,
        reason=reason,
    )


def _bar_level_events(
    *,
    levels: list[Level],
    closes: np.ndarray,
    atr_value: float,
    last_close: float,
) -> tuple[bool, bool, bool, bool, str]:
    if len(closes) < 2:
        return (False, False, False, False, "insufficient_window")

    prev_close = float(closes[-2])
    breakout = False
    pullback = False
    role_reversal = False
    fake_breakout = False
    reason = ""

    for lvl in levels:
        h = lvl.price
        # close-based breakout: previous close on near side, current
        # close on far side.
        if prev_close <= h < last_close or last_close <= h < prev_close:
            breakout = True
            reason = (
                f"close_crossed_level@{h:.6f}"
                if not reason else reason
            )
            break

    # Fake breakout: in last N bars we had a close beyond a level, but
    # the current close is back inside.
    n = min(_FALSE_BREAKOUT_LOOKBACK_BARS, len(closes) - 1)
    if n > 0:
        recent_closes = closes[-n - 1:-1]
        for lvl in levels:
            h = lvl.price
            if (recent_closes > h).any() and last_close <= h:
                fake_breakout = True
                reason = reason or "fake_breakout_above_level"
                break
            if (recent_closes < h).any() and last_close >= h:
                fake_breakout = True
                reason = reason or "fake_breakout_below_level"
                break

    # Pullback: previous bar broke through a level by more than 1 ATR
    # but current close is now within 1 ATR of the level on the far
    # side. Conservative: only flag if level is "both" or the kind
    # matches the new role.
    for lvl in levels:
        h = lvl.price
        if (
            prev_close > h + atr_value and last_close <= h + atr_value
            and last_close >= h
        ):
            pullback = True
            role_reversal = True
            reason = reason or f"pullback_to_role_reversed_level@{h:.6f}"
            break
        if (
            prev_close < h - atr_value and last_close >= h - atr_value
            and last_close <= h
        ):
            pullback = True
            role_reversal = True
            reason = reason or f"pullback_to_role_reversed_level@{h:.6f}"
            break

    if not reason:
        reason = "no_significant_level_interaction"
    return breakout, pullback, role_reversal, fake_breakout, reason


__all__ = [
    "Level",
    "SRSnapshot",
    "detect_levels",
    "empty_snapshot",
]
