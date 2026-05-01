"""Lower-timeframe (15m / 5m) trigger detection for v2.

Strict no-future-leak rule
--------------------------
For a 1h base bar whose CLOSE timestamp = T, only lower-TF bars with
ts <= T are visible. The detector slices `df_lower_tf` accordingly
before running any pattern logic. If no lower-TF data is supplied (or
the slice is empty), it returns an UNKNOWN snapshot — never crashes.

Detected micro-triggers:
  - bullish_pinbar / bearish_pinbar
  - bullish_engulfing / bearish_engulfing
  - breakout       (close above N-bar high or below N-bar low)
  - retest         (after breakout, return to broken level within bound)
  - micro_double_bottom / micro_double_top  (swing-based mini patterns)

This is v2-minimal. Heuristic constants only; pending validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from .technical_confluence import _candlestick_signal


_MICRO_LOOKBACK_BARS: Final[int] = 8
_MICRO_DOUBLE_TOL_PCT: Final[float] = 0.0005   # 0.05%
_RETEST_TOL_PCT: Final[float] = 0.0008         # 0.08%


@dataclass(frozen=True)
class LowerTimeframeTrigger:
    schema_version: str
    interval: str | None
    available: bool
    bars_used: int
    last_lower_tf_ts: str | None
    bullish_pinbar: bool
    bearish_pinbar: bool
    bullish_engulfing: bool
    bearish_engulfing: bool
    breakout: bool
    retest: bool
    micro_double_bottom: bool
    micro_double_top: bool
    bullish_trigger: bool
    bearish_trigger: bool
    unavailable_reason: str | None

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "interval": self.interval,
            "available": bool(self.available),
            "bars_used": int(self.bars_used),
            "last_lower_tf_ts": self.last_lower_tf_ts,
            "bullish_pinbar": bool(self.bullish_pinbar),
            "bearish_pinbar": bool(self.bearish_pinbar),
            "bullish_engulfing": bool(self.bullish_engulfing),
            "bearish_engulfing": bool(self.bearish_engulfing),
            "breakout": bool(self.breakout),
            "retest": bool(self.retest),
            "micro_double_bottom": bool(self.micro_double_bottom),
            "micro_double_top": bool(self.micro_double_top),
            "bullish_trigger": bool(self.bullish_trigger),
            "bearish_trigger": bool(self.bearish_trigger),
            "unavailable_reason": self.unavailable_reason,
        }


def empty_trigger(
    interval: str | None,
    reason: str,
) -> LowerTimeframeTrigger:
    return LowerTimeframeTrigger(
        schema_version="lower_tf_trigger_v2",
        interval=interval,
        available=False,
        bars_used=0,
        last_lower_tf_ts=None,
        bullish_pinbar=False,
        bearish_pinbar=False,
        bullish_engulfing=False,
        bearish_engulfing=False,
        breakout=False,
        retest=False,
        micro_double_bottom=False,
        micro_double_top=False,
        bullish_trigger=False,
        bearish_trigger=False,
        unavailable_reason=reason,
    )


def detect_lower_tf_trigger(
    *,
    df_lower_tf: pd.DataFrame | None,
    lower_tf_interval: str | None,
    base_bar_close_ts: pd.Timestamp,
) -> LowerTimeframeTrigger:
    """Compute the lower-TF trigger snapshot for the current 1h bar.

    `df_lower_tf` may be None (no lower-TF data attached). In that case
    a `unavailable_reason="not_attached"` snapshot is returned so the
    rest of the pipeline isn't disturbed.

    Future-leak guarantee: all lower-TF bars with timestamp > base_bar
    are dropped before any computation.
    """
    if df_lower_tf is None or len(df_lower_tf) == 0:
        return empty_trigger(lower_tf_interval, "not_attached")
    # Strict-leq filter to enforce no future leak.
    visible = df_lower_tf[df_lower_tf.index <= base_bar_close_ts]
    if len(visible) < 3:
        return empty_trigger(lower_tf_interval, "insufficient_visible_bars")

    cs = _candlestick_signal(visible.tail(2))
    bull_pin = bool(cs.get("bullish_pinbar"))
    bear_pin = bool(cs.get("bearish_pinbar"))
    bull_eng = bool(cs.get("bullish_engulfing"))
    bear_eng = bool(cs.get("bearish_engulfing"))

    closes = visible["close"].to_numpy()
    highs = visible["high"].to_numpy()
    lows = visible["low"].to_numpy()
    last_close = float(closes[-1])
    last_ts = visible.index[-1].isoformat() if len(visible) > 0 else None

    n = min(_MICRO_LOOKBACK_BARS, len(closes) - 1)
    breakout = False
    retest = False
    if n > 0:
        prior_high = float(np.max(highs[-n - 1:-1]))
        prior_low = float(np.min(lows[-n - 1:-1]))
        # breakout = current close beyond prior N-bar high/low
        if last_close > prior_high:
            breakout = True
            # retest if the most recent low touches close to broken level
            recent_low = float(np.min(lows[-3:]))
            retest = bool(
                abs(recent_low - prior_high) / prior_high
                <= _RETEST_TOL_PCT
            )
        elif last_close < prior_low:
            breakout = True
            recent_high = float(np.max(highs[-3:]))
            retest = bool(
                abs(recent_high - prior_low) / prior_low
                <= _RETEST_TOL_PCT
            )

    # Micro double bottom: two recent lows close to each other with a
    # peak between them, and current price has rebounded.
    micro_dbottom = False
    micro_dtop = False
    if len(lows) >= 5:
        lows_window = lows[-_MICRO_LOOKBACK_BARS:]
        l_idx = np.argmin(lows_window)
        # Find a second low not equal to the global min
        rest = np.delete(lows_window, l_idx)
        if len(rest) > 0:
            second_min = float(np.min(rest))
            global_min = float(lows_window[l_idx])
            if (
                global_min > 0
                and abs(second_min - global_min) / global_min
                <= _MICRO_DOUBLE_TOL_PCT
                and last_close > second_min
            ):
                micro_dbottom = True
        highs_window = highs[-_MICRO_LOOKBACK_BARS:]
        h_idx = np.argmax(highs_window)
        rest_h = np.delete(highs_window, h_idx)
        if len(rest_h) > 0:
            second_max = float(np.max(rest_h))
            global_max = float(highs_window[h_idx])
            if (
                global_max > 0
                and abs(second_max - global_max) / global_max
                <= _MICRO_DOUBLE_TOL_PCT
                and last_close < second_max
            ):
                micro_dtop = True

    bullish_trigger = bool(
        bull_pin or bull_eng or micro_dbottom
        or (breakout and last_close > closes[-2])
    )
    bearish_trigger = bool(
        bear_pin or bear_eng or micro_dtop
        or (breakout and last_close < closes[-2])
    )

    return LowerTimeframeTrigger(
        schema_version="lower_tf_trigger_v2",
        interval=lower_tf_interval,
        available=True,
        bars_used=int(len(visible)),
        last_lower_tf_ts=last_ts,
        bullish_pinbar=bull_pin,
        bearish_pinbar=bear_pin,
        bullish_engulfing=bull_eng,
        bearish_engulfing=bear_eng,
        breakout=breakout,
        retest=retest,
        micro_double_bottom=micro_dbottom,
        micro_double_top=micro_dtop,
        bullish_trigger=bullish_trigger,
        bearish_trigger=bearish_trigger,
        unavailable_reason=None,
    )


__all__ = [
    "LowerTimeframeTrigger",
    "detect_lower_tf_trigger",
    "empty_trigger",
]
