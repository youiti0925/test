"""Shared retest detection (chart_patterns + lower_timeframe_trigger).

Why a single helper:
  - Two callers need the same semantic ("did price retest a broken
    level and continue in the breakout direction?"). v1 had two
    separate close-based implementations with slightly different
    thresholds. This module unifies them.
  - The retest definition can now be either close-based or wick-based
    via `wick_allowed`. Wick-based catches retests where the high/low
    touched the level but the close did not — common on lower TFs.

Future-leak rule
----------------
The caller passes `closes`, `highs`, `lows`, `timestamps` arrays that
correspond to bars already visible at the current parent bar. The
`parent_bar_ts` is the close timestamp of that parent bar; this helper
will refuse to consider any bar with `timestamps[i] > parent_bar_ts`.
That guard is also pinned by `test_retest_helper_no_future_leak`.

Heuristic constants (NOT validated, NOT in PARAMETER_BASELINE_V1):
  _DEFAULT_TOLERANCE_ATR
  _DEFAULT_CONTINUATION_ATR
  _DEFAULT_MAX_RETEST_BARS
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Sequence

import numpy as np
import pandas as pd


_DEFAULT_TOLERANCE_ATR: Final[float] = 0.3
_DEFAULT_CONTINUATION_ATR: Final[float] = 0.2
_DEFAULT_MAX_RETEST_BARS: Final[int] = 20


Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class RetestResult:
    schema_version: str
    retest_level: float | None
    side: Side
    breakout_index: int | None
    retest_index: int | None
    continuation_index: int | None
    breakout_ts: str | None
    retest_ts: str | None
    continuation_ts: str | None
    tolerance_atr: float
    continuation_atr: float
    max_retest_bars: int
    wick_allowed: bool
    close_confirm_required: bool
    retest_confirmed: bool
    used_bars_end_ts: str | None  # the parent bar boundary (no future leak)
    reason: str

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "retest_level": (
                float(self.retest_level)
                if self.retest_level is not None else None
            ),
            "side": self.side,
            "breakout_index": self.breakout_index,
            "retest_index": self.retest_index,
            "continuation_index": self.continuation_index,
            "breakout_ts": self.breakout_ts,
            "retest_ts": self.retest_ts,
            "continuation_ts": self.continuation_ts,
            "tolerance_atr": float(self.tolerance_atr),
            "continuation_atr": float(self.continuation_atr),
            "max_retest_bars": int(self.max_retest_bars),
            "wick_allowed": bool(self.wick_allowed),
            "close_confirm_required": bool(self.close_confirm_required),
            "retest_confirmed": bool(self.retest_confirmed),
            "used_bars_end_ts": self.used_bars_end_ts,
            "reason": self.reason,
        }


def _empty_result(
    *, level: float | None, side: Side, reason: str,
    parent_bar_ts: pd.Timestamp | None = None,
    wick_allowed: bool = True,
    close_confirm_required: bool = True,
    tolerance_atr: float = _DEFAULT_TOLERANCE_ATR,
    continuation_atr: float = _DEFAULT_CONTINUATION_ATR,
    max_retest_bars: int = _DEFAULT_MAX_RETEST_BARS,
) -> RetestResult:
    return RetestResult(
        schema_version="retest_v1",
        retest_level=level,
        side=side,
        breakout_index=None,
        retest_index=None,
        continuation_index=None,
        breakout_ts=None,
        retest_ts=None,
        continuation_ts=None,
        tolerance_atr=tolerance_atr,
        continuation_atr=continuation_atr,
        max_retest_bars=max_retest_bars,
        wick_allowed=wick_allowed,
        close_confirm_required=close_confirm_required,
        retest_confirmed=False,
        used_bars_end_ts=(
            parent_bar_ts.isoformat()
            if isinstance(parent_bar_ts, pd.Timestamp) else parent_bar_ts
        ),
        reason=reason,
    )


def detect_retest(
    *,
    closes: Sequence[float],
    highs: Sequence[float] | None,
    lows: Sequence[float] | None,
    timestamps: Sequence[pd.Timestamp] | None,
    level: float | None,
    side: Side,
    breakout_search_start: int,
    atr_value: float,
    parent_bar_ts: pd.Timestamp | None,
    tolerance_atr: float = _DEFAULT_TOLERANCE_ATR,
    continuation_atr: float = _DEFAULT_CONTINUATION_ATR,
    max_retest_bars: int = _DEFAULT_MAX_RETEST_BARS,
    wick_allowed: bool = True,
    close_confirm_required: bool = True,
) -> RetestResult:
    """Detect a confirmed retest of `level` in the `side` direction.

    Steps (executed strictly within the visible window):
      1. Walk closes from `breakout_search_start` forward and find the
         FIRST close that is on the breakout side of `level`.
         (For SELL: first close < level. For BUY: first close > level.)
      2. From that breakout bar onward (up to `max_retest_bars` later),
         find a "retest" bar where price returned within
         `tolerance_atr * atr` of `level`. The retest can be either:
            - close-based: |close - level| <= tol
            - wick-based:  any of high/low within tol of level
              (only if `wick_allowed` is True)
         If `close_confirm_required` is True, the retest also requires
         the bar's close to be on the breakout side of `level` (i.e. the
         move never closed back through the level).
      3. The bar AFTER the retest must continue in the breakout direction
         by at least `continuation_atr * atr` (close-to-close).
      4. Returns `retest_confirmed=True` only when all three steps succeed
         within `parent_bar_ts`.

    `parent_bar_ts` is honored as a strict boundary: any bar with
    `timestamps[i] > parent_bar_ts` is ignored even if it is within the
    supplied arrays. This is the no-future-leak guarantee.

    Returns a `RetestResult` shape regardless of outcome (so trace
    consumers always see the full schema, including the bar indices
    consulted).
    """
    closes_arr = np.asarray(closes, dtype=float)
    if (
        level is None or atr_value is None or atr_value <= 0
        or len(closes_arr) == 0 or side not in ("BUY", "SELL")
    ):
        return _empty_result(
            level=level, side=side, reason="invalid_inputs",
            parent_bar_ts=parent_bar_ts,
            wick_allowed=wick_allowed,
            close_confirm_required=close_confirm_required,
            tolerance_atr=tolerance_atr,
            continuation_atr=continuation_atr,
            max_retest_bars=max_retest_bars,
        )

    # Build the visible window honoring parent_bar_ts.
    end_idx = len(closes_arr)
    if timestamps is not None and parent_bar_ts is not None:
        ts_arr = list(timestamps)
        for k in range(len(ts_arr)):
            if ts_arr[k] > parent_bar_ts:
                end_idx = k
                break
    if end_idx < 2:
        return _empty_result(
            level=level, side=side, reason="visible_window_too_short",
            parent_bar_ts=parent_bar_ts,
            wick_allowed=wick_allowed,
            close_confirm_required=close_confirm_required,
            tolerance_atr=tolerance_atr,
            continuation_atr=continuation_atr,
            max_retest_bars=max_retest_bars,
        )

    visible_closes = closes_arr[:end_idx]
    visible_highs = (
        np.asarray(highs, dtype=float)[:end_idx] if highs is not None
        else None
    )
    visible_lows = (
        np.asarray(lows, dtype=float)[:end_idx] if lows is not None
        else None
    )
    visible_ts = list(timestamps[:end_idx]) if timestamps is not None else None

    def _ts_iso(i: int) -> str | None:
        if visible_ts is None or i < 0 or i >= len(visible_ts):
            return None
        t = visible_ts[i]
        return t.isoformat() if isinstance(t, pd.Timestamp) else str(t)

    parent_iso = (
        parent_bar_ts.isoformat()
        if isinstance(parent_bar_ts, pd.Timestamp) else parent_bar_ts
    )

    # Step 1: locate the breakout bar.
    start = max(0, int(breakout_search_start))
    breakout_idx: int | None = None
    for i in range(start, len(visible_closes)):
        c = float(visible_closes[i])
        if side == "SELL" and c < level:
            breakout_idx = i
            break
        if side == "BUY" and c > level:
            breakout_idx = i
            break
    if breakout_idx is None:
        return _empty_result(
            level=level, side=side, reason="no_breakout_in_visible_window",
            parent_bar_ts=parent_bar_ts,
            wick_allowed=wick_allowed,
            close_confirm_required=close_confirm_required,
            tolerance_atr=tolerance_atr,
            continuation_atr=continuation_atr,
            max_retest_bars=max_retest_bars,
        )

    tol = tolerance_atr * atr_value
    cont = continuation_atr * atr_value
    end = min(len(visible_closes), breakout_idx + 1 + max_retest_bars)

    # Step 2 + 3: walk forward looking for a retest + continuation.
    for j in range(breakout_idx + 1, end - 1):
        c_j = float(visible_closes[j])
        # Close-based retest
        close_within = abs(c_j - level) <= tol
        # Wick-based retest (uses high/low when arrays supplied)
        wick_within = False
        if wick_allowed and visible_highs is not None and visible_lows is not None:
            h_j = float(visible_highs[j])
            l_j = float(visible_lows[j])
            wick_within = (
                (l_j - tol) <= level <= (h_j + tol)
            )
        if not (close_within or wick_within):
            continue
        # close_confirm_required: retest bar's close must remain on the
        # breakout side of `level` (didn't close back through).
        if close_confirm_required:
            if side == "SELL" and c_j > level:
                continue
            if side == "BUY" and c_j < level:
                continue
        # Step 3: continuation
        c_next = float(visible_closes[j + 1])
        if side == "SELL" and c_next <= c_j - cont:
            return RetestResult(
                schema_version="retest_v1",
                retest_level=float(level),
                side=side,
                breakout_index=int(breakout_idx),
                retest_index=int(j),
                continuation_index=int(j + 1),
                breakout_ts=_ts_iso(breakout_idx),
                retest_ts=_ts_iso(j),
                continuation_ts=_ts_iso(j + 1),
                tolerance_atr=tolerance_atr,
                continuation_atr=continuation_atr,
                max_retest_bars=max_retest_bars,
                wick_allowed=wick_allowed,
                close_confirm_required=close_confirm_required,
                retest_confirmed=True,
                used_bars_end_ts=parent_iso,
                reason=(
                    "wick_retest_continuation" if (wick_within and not close_within)
                    else "close_retest_continuation"
                ),
            )
        if side == "BUY" and c_next >= c_j + cont:
            return RetestResult(
                schema_version="retest_v1",
                retest_level=float(level),
                side=side,
                breakout_index=int(breakout_idx),
                retest_index=int(j),
                continuation_index=int(j + 1),
                breakout_ts=_ts_iso(breakout_idx),
                retest_ts=_ts_iso(j),
                continuation_ts=_ts_iso(j + 1),
                tolerance_atr=tolerance_atr,
                continuation_atr=continuation_atr,
                max_retest_bars=max_retest_bars,
                wick_allowed=wick_allowed,
                close_confirm_required=close_confirm_required,
                retest_confirmed=True,
                used_bars_end_ts=parent_iso,
                reason=(
                    "wick_retest_continuation" if (wick_within and not close_within)
                    else "close_retest_continuation"
                ),
            )

    return RetestResult(
        schema_version="retest_v1",
        retest_level=float(level),
        side=side,
        breakout_index=int(breakout_idx),
        retest_index=None,
        continuation_index=None,
        breakout_ts=_ts_iso(breakout_idx),
        retest_ts=None,
        continuation_ts=None,
        tolerance_atr=tolerance_atr,
        continuation_atr=continuation_atr,
        max_retest_bars=max_retest_bars,
        wick_allowed=wick_allowed,
        close_confirm_required=close_confirm_required,
        retest_confirmed=False,
        used_bars_end_ts=parent_iso,
        reason="no_continuation_within_max_retest_bars",
    )


__all__ = [
    "RetestResult",
    "detect_retest",
]
