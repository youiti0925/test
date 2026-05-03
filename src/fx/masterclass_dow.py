"""Masterclass Dow structure review (observation-only).

Surfaces the Masterclass Dow concepts as a single audit panel:

  - trend (UP / DOWN / RANGE / UNKNOWN)
  - last_higher_low / last_lower_high (the swings whose break
    invalidates the current trend)
  - trend_break_price: a close beyond this level invalidates
    the current trend label
  - reversal_confirmation_price: a close beyond THIS level
    confirms the OPPOSITE trend (HH→LL or LL→HH)
  - failure_swing: True when the most recent counter-trend swing
    failed to break the prior structural level
  - non_swing: True when fewer than 2 confirmed swings exist
  - status_ja: human Japanese summary

Future-leak rule
----------------
Uses `patterns.detect_swings` which is itself future-leak safe
(swings confirmed only after `lookback` post-bars). The caller
must pass a visible df already truncated to bars whose ts ≤
parent_bar_ts.

Strict invariants
-----------------
- Output carries `observation_only = True` and
  `used_in_decision = False`.
"""
from __future__ import annotations

from typing import Final

import pandas as pd

from .patterns import detect_swings


SCHEMA_VERSION: Final[str] = "dow_structure_review_v1"


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
    }


def _trend_from_swings(swings: list) -> tuple[str, list[str]]:
    """Walk the chronological swings and emit (trend, sequence).

    sequence elements are HH / HL / LH / LL.
    """
    seq: list[str] = []
    last_high: float | None = None
    last_low: float | None = None
    for sw in swings:
        if sw.kind == "H":
            if last_high is not None:
                seq.append("HH" if sw.price > last_high else "LH")
            last_high = sw.price
        else:
            if last_low is not None:
                seq.append("HL" if sw.price > last_low else "LL")
            last_low = sw.price
    tail = seq[-4:] if seq else []
    up = sum(1 for s in tail if s in ("HH", "HL"))
    down = sum(1 for s in tail if s in ("LL", "LH"))
    if up >= 3 and down <= 1:
        trend = "UP"
    elif down >= 3 and up <= 1:
        trend = "DOWN"
    elif up >= 2 and down >= 2:
        trend = "MIXED"
    elif tail:
        trend = "RANGE"
    else:
        trend = "UNKNOWN"
    return trend, seq


def _swing_dict(sw) -> dict:
    return {
        "index": int(sw.index),
        "ts": (
            sw.ts.isoformat()
            if hasattr(sw.ts, "isoformat") else str(sw.ts)
        ),
        "price": float(sw.price),
        "kind": sw.kind,
    }


def build_dow_structure_review(
    *,
    visible_df: pd.DataFrame | None,
    lookback: int = 3,
) -> dict:
    """Build the Dow structure review for the visible window.

    Returns `available=False` for too-short windows (<10 bars).
    """
    if visible_df is None or len(visible_df) < 10:
        return _empty_panel("too_few_bars")
    highs, lows = detect_swings(visible_df, lookback=lookback)
    swings = sorted(highs + lows, key=lambda s: s.index)
    if len(swings) < 2:
        out = _empty_panel("non_swing")
        out["non_swing"] = True
        return out

    trend, sequence = _trend_from_swings(swings)

    # last_higher_low: most recent low whose price > the prior low
    last_higher_low = None
    last_lower_low = None
    prev_low = None
    for sw in swings:
        if sw.kind != "L":
            continue
        if prev_low is not None:
            if sw.price > prev_low:
                last_higher_low = sw
            elif sw.price < prev_low:
                last_lower_low = sw
        prev_low = sw.price

    last_higher_high = None
    last_lower_high = None
    prev_high = None
    for sw in swings:
        if sw.kind != "H":
            continue
        if prev_high is not None:
            if sw.price > prev_high:
                last_higher_high = sw
            elif sw.price < prev_high:
                last_lower_high = sw
        prev_high = sw.price

    # Trend-break + reversal-confirmation prices.
    # UP: a close below the last_higher_low invalidates UP.
    #     a close below ANOTHER lower_high after that confirms DOWN.
    # DOWN: a close above the last_lower_high invalidates DOWN.
    #       a close above another higher_high after that confirms UP.
    trend_break_price = None
    reversal_confirmation_price = None
    if trend == "UP":
        if last_higher_low is not None:
            trend_break_price = float(last_higher_low.price)
        if last_lower_high is not None:
            reversal_confirmation_price = float(last_lower_high.price)
    elif trend == "DOWN":
        if last_lower_high is not None:
            trend_break_price = float(last_lower_high.price)
        if last_higher_high is not None:
            reversal_confirmation_price = float(last_higher_high.price)

    # Failure swing: the most recent counter-trend swing did NOT
    # exceed the prior structural extreme.
    failure_swing = False
    if trend == "UP" and len(swings) >= 4:
        last_high_swing = next(
            (sw for sw in reversed(swings) if sw.kind == "H"), None,
        )
        prior_highs = [sw for sw in swings[:-1] if sw.kind == "H"]
        if last_high_swing is not None and prior_highs:
            top_prior = max(p.price for p in prior_highs)
            if last_high_swing.price < top_prior:
                failure_swing = True
    elif trend == "DOWN" and len(swings) >= 4:
        last_low_swing = next(
            (sw for sw in reversed(swings) if sw.kind == "L"), None,
        )
        prior_lows = [sw for sw in swings[:-1] if sw.kind == "L"]
        if last_low_swing is not None and prior_lows:
            bot_prior = min(p.price for p in prior_lows)
            if last_low_swing.price > bot_prior:
                failure_swing = True

    if trend == "UP":
        status_ja = (
            "上昇トレンド継続中です。"
            + (
                f"押し安値 (last higher low) を割ると上昇構造が崩れます。"
                if trend_break_price is not None else ""
            )
            + (" Failure swing 兆候あり (高値更新失敗)。" if failure_swing else "")
        )
    elif trend == "DOWN":
        status_ja = (
            "下降トレンド継続中です。"
            + (
                f"戻り高値 (last lower high) を上抜けると下降構造が崩れます。"
                if trend_break_price is not None else ""
            )
            + (" Failure swing 兆候あり (安値更新失敗)。" if failure_swing else "")
        )
    elif trend == "RANGE":
        status_ja = (
            "レンジ相場。トレンドが明確でないため、"
            "ダウ理論的にはエントリーを見送る局面です。"
        )
    elif trend == "MIXED":
        status_ja = (
            "もみ合い (HH/HL と LH/LL が混在)。方向性が確定していません。"
        )
    else:
        status_ja = "ダウ構造を判定するスイングが不足しています。"

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "trend": trend,
        "sequence": sequence,
        "last_4_sequence": sequence[-4:] if sequence else [],
        "last_higher_low": (
            _swing_dict(last_higher_low)
            if last_higher_low is not None else None
        ),
        "last_higher_high": (
            _swing_dict(last_higher_high)
            if last_higher_high is not None else None
        ),
        "last_lower_low": (
            _swing_dict(last_lower_low)
            if last_lower_low is not None else None
        ),
        "last_lower_high": (
            _swing_dict(last_lower_high)
            if last_lower_high is not None else None
        ),
        "trend_break_price": trend_break_price,
        "reversal_confirmation_price": reversal_confirmation_price,
        "failure_swing": bool(failure_swing),
        "non_swing": False,
        "status_ja": status_ja,
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_dow_structure_review",
]
