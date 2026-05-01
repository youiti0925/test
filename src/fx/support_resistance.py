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


LevelKind = Literal["support", "resistance", "both"]


@dataclass(frozen=True)
class Level:
    price: float
    kind: LevelKind
    touch_count: int
    first_touch_ts: pd.Timestamp | None
    last_touch_ts: pd.Timestamp | None
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
        # role_reversal_count proxy: a "both" cluster has at least one
        # reversal (a high that later acted as low or vice versa).
        role_reversal_count = (
            len(cluster) - 1 if kind == "both" else 0
        )
        # broken_count and false_breakout_count are computed in the
        # bar-loop (not at clustering time) — kept at 0 here.
        strength = (
            math.log1p(len(cluster))
            * recency_score
            * (1.0 + role_reversal_count * 0.1)
        )
        levels.append(Level(
            price=float(np.mean(prices)),
            kind=kind,
            touch_count=len(cluster),
            first_touch_ts=getattr(first_touch, "ts", None),
            last_touch_ts=getattr(last_touch, "ts", None),
            last_touch_index=int(last_touch.index),
            broken_count=0,
            role_reversal_count=int(role_reversal_count),
            false_breakout_count=0,
            strength_score=float(strength),
            recency_score=float(recency_score),
            distance_to_close_atr=None,  # filled in detect_levels
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

    # Compute distance_to_close_atr per level
    enriched: list[Level] = []
    for lvl in levels:
        dist_atr = abs(last_close - lvl.price) / atr_value
        enriched.append(Level(
            price=lvl.price,
            kind=lvl.kind,
            touch_count=lvl.touch_count,
            first_touch_ts=lvl.first_touch_ts,
            last_touch_ts=lvl.last_touch_ts,
            last_touch_index=lvl.last_touch_index,
            broken_count=lvl.broken_count,
            role_reversal_count=lvl.role_reversal_count,
            false_breakout_count=lvl.false_breakout_count,
            strength_score=lvl.strength_score,
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
