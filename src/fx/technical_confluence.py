"""Technical confluence v1 — observation-only royal-road feature trace.

This module computes the royal-road technical confluence summary for a
single bar and returns it as a JSON-serializable dict. The output is
threaded into `BarDecisionTrace.technical_confluence` (a new optional
slice) so post-hoc analysis can correlate royal-road signals with
WIN/LOSS outcomes.

Observation-only contract
-------------------------
- `decide_action` / `risk_gate` / `backtest_engine` decide() pipeline
  do NOT read any field produced here.
- Adding/removing fields here MUST NOT change `result.trades`,
  `result.metrics`, or `BarDecisionTrace.decision.final_action`.
  This is pinned by `tests/test_technical_confluence_trace_observation_only.py`.
- All thresholds are local constants in this module — they do NOT live
  in `parameter_defaults.PARAMETER_BASELINE_V1` (would change baseline
  hash) and they do NOT flow through `runtime_parameters.py`.

Royal-road sections covered
---------------------------
1. Dow / market structure  (reuses `patterns.PatternResult`)
2. Support / resistance    (uses confirmed swings as level proxy)
3. Candlestick signals     (pinbar / engulfing / harami / strong body)
4. Indicator context       (RSI regime / BB lifecycle / MACD momentum)
5. Risk plan observation   (ATR stop vs structure stop, distances)
6. Final confluence label  (STRONG/WEAK BUY/SELL, NO_TRADE, AVOID_TRADE)

What this module does NOT do
----------------------------
- Detect S/R from horizontal level clustering (uses confirmed swings only).
- Detect trendlines / fake breakouts beyond a one-bar simple revert check.
- Detect chart patterns beyond what `patterns.analyse` already produced.
- Score head_and_shoulders / flag / wedge / triangle (out of scope for v1).
- Connect anything to decide_action.

A future PR (and only after multi-symbol × multi-period evidence) may
gate this output behind an opt-in flag and feed it to decision logic.
v1 is observation-only by design.
"""
from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd

from .indicators import Snapshot
from .patterns import PatternResult, TrendState


POLICY_VERSION: Final[str] = "technical_confluence_v1"


# ---------------------------------------------------------------------------
# Tunable constants (observation thresholds, NOT in PARAMETER_BASELINE_V1)
# ---------------------------------------------------------------------------

# Candlestick
_PINBAR_WICK_BODY_RATIO: Final[float] = 2.0  # wick must be >= 2x body
_PINBAR_BODY_RANGE_RATIO: Final[float] = 0.35  # body is small vs range
_STRONG_BODY_RATIO: Final[float] = 0.65  # body >= 65% of range
_REJECTION_WICK_RATIO: Final[float] = 0.5   # rejected wick >= 50% of range

# Support / resistance
_NEAR_LEVEL_ATR: Final[float] = 0.5         # within 0.5 ATR counts as "near"
_FAKE_BREAKOUT_LOOKBACK: Final[int] = 3     # bars to look back for breakout

# Bollinger lifecycle
_BB_HISTORY_BARS: Final[int] = 60           # rolling window for percentile
_BB_SQUEEZE_PCTL: Final[float] = 0.20       # bottom 20% of band-width history
_BB_EXPANSION_PCTL: Final[float] = 0.80     # top 20% of band-width history
_BB_BAND_WALK_BARS: Final[int] = 3          # bars touching band counts as walk
_BB_BAND_TOUCH_TOLERANCE_PCT: Final[float] = 0.001  # 0.1% of close

# Risk plan defaults (observation only)
_DEFAULT_STOP_ATR_MULT_OBS: Final[float] = 2.0
_DEFAULT_TP_ATR_MULT_OBS: Final[float] = 3.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_technical_confluence(
    *,
    df_window: pd.DataFrame,
    snapshot: Snapshot | None,
    pattern: PatternResult | None,
    atr_value: float | None,
    technical_only_action: str,
    stop_atr_mult: float = _DEFAULT_STOP_ATR_MULT_OBS,
    tp_atr_mult: float = _DEFAULT_TP_ATR_MULT_OBS,
) -> dict:
    """Compute the technical_confluence_v1 dict for one bar.

    Parameters
    ----------
    df_window:
        OHLC(V) up through the current bar inclusive (`df.iloc[: i + 1]`).
        Must have columns `open`, `high`, `low`, `close`. No future bars.
    snapshot:
        Indicator snapshot at the current bar. May be None when the
        engine bailed before computing indicators (ATR unavailable).
    pattern:
        `patterns.analyse(df_window)` result. May be None for the same
        reason as snapshot.
    atr_value:
        Current ATR(14) value. May be None on warmup bars.
    technical_only_action:
        `"BUY" | "SELL" | "HOLD"` from `technical_signal()`. Used solely
        to pick which side's structure stop to surface; never gates.
    stop_atr_mult / tp_atr_mult:
        Observation values reported in `risk_plan_obs`. Defaults match
        the historical engine defaults (2.0 / 3.0). Pass the runtime
        values when available so the trace shows the actual multiplier.

    Returns
    -------
    A JSON-serializable dict shaped like:

        {
          "policy_version": "technical_confluence_v1",
          "market_regime": ...,
          "dow_structure": {...},
          "support_resistance": {...},
          "candlestick_signal": {...},
          "chart_pattern": {...},
          "indicator_context": {...},
          "risk_plan_obs": {...},
          "vote_breakdown": {...},
          "final_confluence": {...},
        }

    All numeric values are float/int/None (no numpy scalars). All
    boolean values are bool (no np.bool_). Strings are plain str.
    """
    last_close = (
        float(df_window["close"].iloc[-1])
        if len(df_window) > 0 else None
    )

    market_regime = _market_regime(pattern)
    dow_structure = _dow_structure(pattern, last_close)
    support_resistance = _support_resistance(
        df_window, pattern, atr_value, last_close,
    )
    candlestick_signal = _candlestick_signal(df_window)
    chart_pattern = _chart_pattern(pattern)
    indicator_context = _indicator_context(df_window, snapshot, pattern)
    risk_plan_obs = _risk_plan_obs(
        last_close, pattern, atr_value, technical_only_action,
        stop_atr_mult, tp_atr_mult,
    )
    vote_breakdown = _vote_breakdown(
        snapshot, technical_only_action, market_regime,
        dow_structure, support_resistance, candlestick_signal,
    )
    final_confluence = _final_confluence_label(
        technical_only_action,
        market_regime, dow_structure, support_resistance,
        candlestick_signal, chart_pattern, indicator_context,
        risk_plan_obs, vote_breakdown,
    )

    return {
        "policy_version": POLICY_VERSION,
        "market_regime": market_regime,
        "dow_structure": dow_structure,
        "support_resistance": support_resistance,
        "candlestick_signal": candlestick_signal,
        "chart_pattern": chart_pattern,
        "indicator_context": indicator_context,
        "risk_plan_obs": risk_plan_obs,
        "vote_breakdown": vote_breakdown,
        "final_confluence": final_confluence,
    }


def empty_technical_confluence() -> dict:
    """Return a fully-populated UNKNOWN dict for bars where computation
    is not possible (warmup, missing data). Same schema as the live
    output so consumers can rely on a stable shape."""
    return {
        "policy_version": POLICY_VERSION,
        "market_regime": "UNKNOWN",
        "dow_structure": {
            "structure_code": "UNKNOWN",
            "last_swing_high": None,
            "last_swing_low": None,
            "bos_up": False,
            "bos_down": False,
        },
        "support_resistance": {
            "nearest_support": None,
            "nearest_resistance": None,
            "distance_to_support_atr": None,
            "distance_to_resistance_atr": None,
            "near_support": False,
            "near_resistance": False,
            "breakout": False,
            "pullback": False,
            "role_reversal": False,
            "fake_breakout": False,
            "reason": "insufficient_data",
        },
        "candlestick_signal": {
            "bullish_pinbar": False,
            "bearish_pinbar": False,
            "bullish_engulfing": False,
            "bearish_engulfing": False,
            "harami": False,
            "strong_bull_body": False,
            "strong_bear_body": False,
            "rejection_wick": False,
        },
        "chart_pattern": {
            "double_top": False,
            "double_bottom": False,
            "triple_top": False,
            "triple_bottom": False,
            "head_and_shoulders": False,
            "inverse_head_and_shoulders": False,
            "flag": False,
            "wedge": False,
            "triangle": False,
            "neckline_broken": False,
            "retested": False,
        },
        "indicator_context": {
            "rsi_value": None,
            "rsi_range_valid": None,
            "rsi_trend_danger": None,
            "macd_momentum_up": None,
            "macd_momentum_down": None,
            "bb_squeeze": None,
            "bb_expansion": None,
            "bb_band_walk": None,
            "ma_trend_support": None,
        },
        "risk_plan_obs": {
            "atr_stop_distance_atr": None,
            "structure_stop_price": None,
            "structure_stop_distance_atr": None,
            "rr_atr_based": None,
            "rr_structure_based": None,
            "invalidation_clear": False,
        },
        "vote_breakdown": {
            "indicator_buy_votes": 0,
            "indicator_sell_votes": 0,
            "voted_action": "HOLD",
            "macro_alignment": "UNKNOWN",
            "htf_alignment": "UNKNOWN",
            "structure_alignment": "UNKNOWN",
            "sr_alignment": "UNKNOWN",
            "candle_alignment": "UNKNOWN",
        },
        "final_confluence": {
            "label": "UNKNOWN",
            "score": 0.0,
            "bullish_reasons": [],
            "bearish_reasons": [],
            "avoid_reasons": ["insufficient_data"],
        },
    }


# ---------------------------------------------------------------------------
# Section 1: market regime / Dow structure
# ---------------------------------------------------------------------------


def _market_regime(pattern: PatternResult | None) -> str:
    if pattern is None:
        return "UNKNOWN"
    state = pattern.trend_state
    if state == TrendState.UPTREND:
        return "TREND_UP"
    if state == TrendState.DOWNTREND:
        return "TREND_DOWN"
    if state == TrendState.RANGE:
        return "RANGE"
    if state == TrendState.VOLATILE:
        return "VOLATILE"
    return "UNKNOWN"


def _dow_structure(
    pattern: PatternResult | None, last_close: float | None,
) -> dict:
    if pattern is None or last_close is None:
        return {
            "structure_code": "UNKNOWN",
            "last_swing_high": None,
            "last_swing_low": None,
            "bos_up": False,
            "bos_down": False,
        }
    last_high = (
        float(pattern.swing_highs[-1].price)
        if pattern.swing_highs else None
    )
    last_low = (
        float(pattern.swing_lows[-1].price)
        if pattern.swing_lows else None
    )
    code = pattern.market_structure[-1] if pattern.market_structure else "UNKNOWN"
    bos_up = bool(last_high is not None and last_close > last_high)
    bos_down = bool(last_low is not None and last_close < last_low)
    return {
        "structure_code": code,
        "last_swing_high": last_high,
        "last_swing_low": last_low,
        "bos_up": bos_up,
        "bos_down": bos_down,
    }


# ---------------------------------------------------------------------------
# Section 2: support / resistance observation
# ---------------------------------------------------------------------------


def _support_resistance(
    df_window: pd.DataFrame,
    pattern: PatternResult | None,
    atr_value: float | None,
    last_close: float | None,
) -> dict:
    base = {
        "nearest_support": None,
        "nearest_resistance": None,
        "distance_to_support_atr": None,
        "distance_to_resistance_atr": None,
        "near_support": False,
        "near_resistance": False,
        "breakout": False,
        "pullback": False,
        "role_reversal": False,
        "fake_breakout": False,
        "reason": "",
    }
    if (
        pattern is None or last_close is None
        or atr_value is None or atr_value <= 0
    ):
        base["reason"] = "missing_pattern_or_atr"
        return base

    highs = [s.price for s in pattern.swing_highs]
    lows = [s.price for s in pattern.swing_lows]

    resistances = [h for h in highs if h > last_close]
    supports = [l for l in lows if l < last_close]
    nearest_resistance = min(resistances) if resistances else None
    nearest_support = max(supports) if supports else None

    dist_r_atr = (
        (nearest_resistance - last_close) / atr_value
        if nearest_resistance is not None else None
    )
    dist_s_atr = (
        (last_close - nearest_support) / atr_value
        if nearest_support is not None else None
    )

    near_resistance = bool(
        dist_r_atr is not None and dist_r_atr <= _NEAR_LEVEL_ATR
    )
    near_support = bool(
        dist_s_atr is not None and dist_s_atr <= _NEAR_LEVEL_ATR
    )

    breakout, fake_breakout, pullback, role_reversal, reason = (
        _breakout_pullback(df_window, highs, lows, atr_value, last_close)
    )

    return {
        "nearest_support": (
            float(nearest_support) if nearest_support is not None else None
        ),
        "nearest_resistance": (
            float(nearest_resistance) if nearest_resistance is not None else None
        ),
        "distance_to_support_atr": (
            float(dist_s_atr) if dist_s_atr is not None else None
        ),
        "distance_to_resistance_atr": (
            float(dist_r_atr) if dist_r_atr is not None else None
        ),
        "near_support": near_support,
        "near_resistance": near_resistance,
        "breakout": breakout,
        "pullback": pullback,
        "role_reversal": role_reversal,
        "fake_breakout": fake_breakout,
        "reason": reason,
    }


def _breakout_pullback(
    df_window: pd.DataFrame,
    highs: list[float],
    lows: list[float],
    atr_value: float,
    last_close: float,
) -> tuple[bool, bool, bool, bool, str]:
    """Simple close-based breakout / fake-breakout / pullback / role-reversal.

    Definitions (intentionally conservative):
      breakout      = current close on the far side of a swing level
                      that the previous close was on the near side of.
      fake_breakout = within `_FAKE_BREAKOUT_LOOKBACK` bars before the
                      current bar, a close broke above (or below) a
                      level that the current close is back inside.
      pullback      = previous close was on the far side of the level
                      AND current close has retreated toward but not
                      crossed back through it (within 1 ATR).
      role_reversal = a level previously acted as resistance is now
                      acting as support (or vice versa) — proxied by
                      previous-close-above-low and current-close-still-above.
    """
    if len(df_window) < 2:
        return (False, False, False, False, "insufficient_window")

    closes = df_window["close"].to_numpy()
    prev_close = float(closes[-2])
    breakout = False
    fake_breakout = False
    pullback = False
    role_reversal = False
    reason = ""

    # Levels strictly: highs are resistance candidates, lows are support
    for h in highs:
        if prev_close <= h < last_close:
            breakout = True
            reason = "close_above_resistance"
            break
        if last_close <= h < prev_close:
            breakout = True
            reason = "close_below_resistance_after_above"
            break
    for l in lows:
        if last_close < l <= prev_close:
            breakout = True
            reason = "close_below_support"
            break

    # Fake breakout: any of the last N bars closed beyond a level, but
    # the current close is back inside.
    n = min(_FAKE_BREAKOUT_LOOKBACK, len(closes) - 1)
    if n > 0:
        recent_closes = closes[-n - 1 : -1]  # excluding current bar
        for h in highs:
            if (recent_closes > h).any() and last_close <= h:
                fake_breakout = True
                reason = "fake_breakout_above_then_back"
                break
        if not fake_breakout:
            for l in lows:
                if (recent_closes < l).any() and last_close >= l:
                    fake_breakout = True
                    reason = "fake_breakout_below_then_back"
                    break

    # Pullback: previous bar broke a resistance, current close is just
    # under the level (within 1 ATR) but still above it.
    for h in highs:
        if prev_close > h >= last_close - atr_value and last_close >= h:
            pullback = True
            reason = reason or "pullback_to_resistance_now_support"
            role_reversal = True
            break
    for l in lows:
        if prev_close < l <= last_close + atr_value and last_close <= l:
            pullback = True
            reason = reason or "pullback_to_support_now_resistance"
            role_reversal = True
            break

    if not reason:
        reason = "no_significant_level_interaction"
    return breakout, fake_breakout, pullback, role_reversal, reason


# ---------------------------------------------------------------------------
# Section 3: candlestick signals
# ---------------------------------------------------------------------------


def _candlestick_signal(df_window: pd.DataFrame) -> dict:
    out = {
        "bullish_pinbar": False,
        "bearish_pinbar": False,
        "bullish_engulfing": False,
        "bearish_engulfing": False,
        "harami": False,
        "strong_bull_body": False,
        "strong_bear_body": False,
        "rejection_wick": False,
    }
    if len(df_window) < 1:
        return out

    o = float(df_window["open"].iloc[-1])
    h = float(df_window["high"].iloc[-1])
    l = float(df_window["low"].iloc[-1])
    c = float(df_window["close"].iloc[-1])
    body = abs(c - o)
    bar_range = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if bar_range <= 0:
        return out

    body_ratio = body / bar_range

    # Pinbar: small body, one wick dominates
    if (
        body_ratio <= _PINBAR_BODY_RANGE_RATIO
        and body > 0
        and lower_wick >= _PINBAR_WICK_BODY_RATIO * body
        and lower_wick > upper_wick
    ):
        out["bullish_pinbar"] = True
    if (
        body_ratio <= _PINBAR_BODY_RANGE_RATIO
        and body > 0
        and upper_wick >= _PINBAR_WICK_BODY_RATIO * body
        and upper_wick > lower_wick
    ):
        out["bearish_pinbar"] = True

    # Strong body
    if body_ratio >= _STRONG_BODY_RATIO and c > o:
        out["strong_bull_body"] = True
    if body_ratio >= _STRONG_BODY_RATIO and c < o:
        out["strong_bear_body"] = True

    # Rejection wick: a wick >= 50% of the range that opposes the body
    if c > o and upper_wick / bar_range >= _REJECTION_WICK_RATIO:
        out["rejection_wick"] = True
    if c < o and lower_wick / bar_range >= _REJECTION_WICK_RATIO:
        out["rejection_wick"] = True

    if len(df_window) >= 2:
        po = float(df_window["open"].iloc[-2])
        pc = float(df_window["close"].iloc[-2])
        prev_top = max(po, pc)
        prev_bot = min(po, pc)
        cur_top = max(o, c)
        cur_bot = min(o, c)
        # Bullish engulfing: previous bearish, current bullish, current
        # body engulfs previous body.
        if (
            pc < po and c > o
            and cur_bot < prev_bot and cur_top > prev_top
        ):
            out["bullish_engulfing"] = True
        if (
            pc > po and c < o
            and cur_top > prev_top and cur_bot < prev_bot
        ):
            out["bearish_engulfing"] = True
        # Harami: current body inside previous body, regardless of color
        if cur_top <= prev_top and cur_bot >= prev_bot and (cur_top - cur_bot) < (prev_top - prev_bot):
            out["harami"] = True

    return out


# ---------------------------------------------------------------------------
# Section 4: chart pattern (mirror of pattern.detected_pattern)
# ---------------------------------------------------------------------------


def _chart_pattern(pattern: PatternResult | None) -> dict:
    out = {
        "double_top": False,
        "double_bottom": False,
        "triple_top": False,
        "triple_bottom": False,
        "head_and_shoulders": False,
        "inverse_head_and_shoulders": False,
        "flag": False,
        "wedge": False,
        "triangle": False,
        "neckline_broken": False,
        "retested": False,
    }
    if pattern is None or pattern.detected_pattern is None:
        return out
    name = pattern.detected_pattern
    if name == "DOUBLE_TOP_CANDIDATE":
        out["double_top"] = True
    elif name == "DOUBLE_BOTTOM_CANDIDATE":
        out["double_bottom"] = True
    elif name == "TRIPLE_TOP_CANDIDATE":
        out["triple_top"] = True
    elif name == "TRIPLE_BOTTOM_CANDIDATE":
        out["triple_bottom"] = True
    out["neckline_broken"] = bool(pattern.neckline_broken)
    return out


# ---------------------------------------------------------------------------
# Section 5: indicator context (RSI regime / BB lifecycle / MACD)
# ---------------------------------------------------------------------------


def _indicator_context(
    df_window: pd.DataFrame,
    snapshot: Snapshot | None,
    pattern: PatternResult | None,
) -> dict:
    out: dict[str, Any] = {
        "rsi_value": None,
        "rsi_range_valid": None,
        "rsi_trend_danger": None,
        "macd_momentum_up": None,
        "macd_momentum_down": None,
        "bb_squeeze": None,
        "bb_expansion": None,
        "bb_band_walk": None,
        "ma_trend_support": None,
    }
    if snapshot is None:
        return out

    rsi_value = float(snapshot.rsi_14)
    out["rsi_value"] = rsi_value

    trend = pattern.trend_state if pattern is not None else TrendState.RANGE
    out["rsi_range_valid"] = bool(trend == TrendState.RANGE)
    out["rsi_trend_danger"] = bool(
        (trend == TrendState.UPTREND and rsi_value > 70)
        or (trend == TrendState.DOWNTREND and rsi_value < 30)
    )

    out["macd_momentum_up"] = bool(
        snapshot.macd_hist > 0 and snapshot.macd > snapshot.macd_signal
    )
    out["macd_momentum_down"] = bool(
        snapshot.macd_hist < 0 and snapshot.macd < snapshot.macd_signal
    )

    bb_obs = _bollinger_lifecycle(df_window, snapshot)
    out.update(bb_obs)

    if snapshot.sma_20 > snapshot.sma_50:
        out["ma_trend_support"] = "BUY"
    elif snapshot.sma_20 < snapshot.sma_50:
        out["ma_trend_support"] = "SELL"
    else:
        out["ma_trend_support"] = "NEUTRAL"

    return out


def _bollinger_lifecycle(
    df_window: pd.DataFrame, snapshot: Snapshot,
) -> dict:
    """squeeze / expansion / band_walk based on rolling band-width history."""
    n = min(_BB_HISTORY_BARS, len(df_window))
    if n < 20:
        return {
            "bb_squeeze": None,
            "bb_expansion": None,
            "bb_band_walk": None,
        }
    closes = df_window["close"].iloc[-n:]
    rolling_mean = closes.rolling(window=20, min_periods=20).mean()
    rolling_std = closes.rolling(window=20, min_periods=20).std()
    upper = rolling_mean + 2.0 * rolling_std
    lower = rolling_mean - 2.0 * rolling_std
    width = (upper - lower).dropna()
    if len(width) < 10:
        return {
            "bb_squeeze": None,
            "bb_expansion": None,
            "bb_band_walk": None,
        }
    cur_width = float(width.iloc[-1])
    pctl = float((width <= cur_width).mean())
    bb_squeeze = bool(pctl <= _BB_SQUEEZE_PCTL)
    bb_expansion = bool(pctl >= _BB_EXPANSION_PCTL)

    # band_walk: last K bars' close near upper band (for bull walk) or
    # near lower band (for bear walk).
    k = min(_BB_BAND_WALK_BARS, len(width))
    last_closes = closes.iloc[-k:].to_numpy()
    last_uppers = upper.iloc[-k:].to_numpy()
    last_lowers = lower.iloc[-k:].to_numpy()
    tol_upper = last_uppers * _BB_BAND_TOUCH_TOLERANCE_PCT
    tol_lower = last_lowers * _BB_BAND_TOUCH_TOLERANCE_PCT
    walking_up = bool(
        np.all(last_closes >= (last_uppers - np.abs(tol_upper)))
    )
    walking_down = bool(
        np.all(last_closes <= (last_lowers + np.abs(tol_lower)))
    )
    return {
        "bb_squeeze": bb_squeeze,
        "bb_expansion": bb_expansion,
        "bb_band_walk": bool(walking_up or walking_down),
    }


# ---------------------------------------------------------------------------
# Section 6: risk plan observation
# ---------------------------------------------------------------------------


def _risk_plan_obs(
    last_close: float | None,
    pattern: PatternResult | None,
    atr_value: float | None,
    technical_only_action: str,
    stop_atr_mult: float,
    tp_atr_mult: float,
) -> dict:
    out: dict[str, Any] = {
        "atr_stop_distance_atr": float(stop_atr_mult),
        "structure_stop_price": None,
        "structure_stop_distance_atr": None,
        "rr_atr_based": (
            float(tp_atr_mult / stop_atr_mult)
            if stop_atr_mult > 0 else None
        ),
        "rr_structure_based": None,
        "invalidation_clear": False,
    }
    if (
        last_close is None or pattern is None
        or atr_value is None or atr_value <= 0
    ):
        return out

    structure_stop: float | None = None
    if technical_only_action == "BUY" and pattern.swing_lows:
        structure_stop = float(pattern.swing_lows[-1].price)
    elif technical_only_action == "SELL" and pattern.swing_highs:
        structure_stop = float(pattern.swing_highs[-1].price)

    if structure_stop is None:
        return out

    distance = abs(last_close - structure_stop)
    distance_atr = distance / atr_value
    out["structure_stop_price"] = float(structure_stop)
    out["structure_stop_distance_atr"] = float(distance_atr)
    # RR_structure: 3 ATR target / structure stop distance (ATR units)
    if distance_atr > 0:
        out["rr_structure_based"] = float(tp_atr_mult / distance_atr)
    out["invalidation_clear"] = bool(distance_atr >= 0.5)
    return out


# ---------------------------------------------------------------------------
# Section 7: vote breakdown — expose the indicator-vote internals
# ---------------------------------------------------------------------------


def _vote_breakdown(
    snapshot: Snapshot | None,
    technical_only_action: str,
    market_regime: str,
    dow_structure: dict,
    support_resistance: dict,
    candlestick_signal: dict,
) -> dict:
    """Reconstruct the 4-way indicator vote and surface alignment labels.

    The vote rules MUST match `indicators.technical_signal` exactly (RSI
    70/30, MACD line+hist, SMA20 vs SMA50, BB position 0.2/0.8). This is
    not a re-implementation of the decision — it's an audit copy so
    consumers can see *which* indicators voted which way.
    """
    out = {
        "indicator_buy_votes": 0,
        "indicator_sell_votes": 0,
        "voted_action": technical_only_action,
        "macro_alignment": "UNKNOWN",
        "htf_alignment": "UNKNOWN",
        "structure_alignment": "UNKNOWN",
        "sr_alignment": "UNKNOWN",
        "candle_alignment": "UNKNOWN",
    }
    if snapshot is None:
        return out

    buy = 0
    sell = 0
    if snapshot.rsi_14 < 30:
        buy += 1
    elif snapshot.rsi_14 > 70:
        sell += 1
    if snapshot.macd_hist > 0 and snapshot.macd > snapshot.macd_signal:
        buy += 1
    elif snapshot.macd_hist < 0 and snapshot.macd < snapshot.macd_signal:
        sell += 1
    if snapshot.sma_20 > snapshot.sma_50:
        buy += 1
    elif snapshot.sma_20 < snapshot.sma_50:
        sell += 1
    if snapshot.bb_position < 0.2:
        buy += 1
    elif snapshot.bb_position > 0.8:
        sell += 1

    out["indicator_buy_votes"] = int(buy)
    out["indicator_sell_votes"] = int(sell)

    out["structure_alignment"] = _alignment_from_regime(market_regime)
    out["sr_alignment"] = _alignment_from_sr(support_resistance)
    out["candle_alignment"] = _alignment_from_candle(candlestick_signal)
    # macro / htf are not exposed here — we keep them UNKNOWN since this
    # module is constrained to OHLCV+indicators+pattern. Confluence
    # consumers can fill macro/htf alignment from the corresponding
    # MacroContextSlice / HigherTimeframeSlice if needed.
    return out


def _alignment_from_regime(regime: str) -> str:
    if regime == "TREND_UP":
        return "BUY"
    if regime == "TREND_DOWN":
        return "SELL"
    return "NEUTRAL"


def _alignment_from_sr(sr: dict) -> str:
    if sr.get("near_resistance"):
        return "SELL"
    if sr.get("near_support"):
        return "BUY"
    return "NEUTRAL"


def _alignment_from_candle(c: dict) -> str:
    bull = (
        c.get("bullish_pinbar")
        or c.get("bullish_engulfing")
        or c.get("strong_bull_body")
    )
    bear = (
        c.get("bearish_pinbar")
        or c.get("bearish_engulfing")
        or c.get("strong_bear_body")
    )
    if bull and not bear:
        return "BUY"
    if bear and not bull:
        return "SELL"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Section 8: final confluence label
# ---------------------------------------------------------------------------


def _final_confluence_label(
    technical_only_action: str,
    market_regime: str,
    dow_structure: dict,
    support_resistance: dict,
    candlestick_signal: dict,
    chart_pattern: dict,
    indicator_context: dict,
    risk_plan_obs: dict,
    vote_breakdown: dict,
) -> dict:
    bullish_reasons: list[str] = []
    bearish_reasons: list[str] = []
    avoid_reasons: list[str] = []

    if market_regime == "TREND_UP":
        bullish_reasons.append("regime_trend_up")
    elif market_regime == "TREND_DOWN":
        bearish_reasons.append("regime_trend_down")
    elif market_regime == "VOLATILE":
        avoid_reasons.append("regime_volatile")

    if dow_structure.get("bos_up"):
        bullish_reasons.append("bos_up")
    if dow_structure.get("bos_down"):
        bearish_reasons.append("bos_down")

    if support_resistance.get("near_support"):
        bullish_reasons.append("near_support")
    if support_resistance.get("near_resistance"):
        bearish_reasons.append("near_resistance")
    if support_resistance.get("fake_breakout"):
        avoid_reasons.append("fake_breakout")

    if candlestick_signal.get("bullish_pinbar"):
        bullish_reasons.append("bullish_pinbar")
    if candlestick_signal.get("bearish_pinbar"):
        bearish_reasons.append("bearish_pinbar")
    if candlestick_signal.get("bullish_engulfing"):
        bullish_reasons.append("bullish_engulfing")
    if candlestick_signal.get("bearish_engulfing"):
        bearish_reasons.append("bearish_engulfing")
    if candlestick_signal.get("strong_bull_body"):
        bullish_reasons.append("strong_bull_body")
    if candlestick_signal.get("strong_bear_body"):
        bearish_reasons.append("strong_bear_body")

    if chart_pattern.get("double_bottom") or chart_pattern.get("triple_bottom"):
        bullish_reasons.append("bottom_pattern")
    if chart_pattern.get("double_top") or chart_pattern.get("triple_top"):
        bearish_reasons.append("top_pattern")

    if indicator_context.get("rsi_trend_danger"):
        avoid_reasons.append("rsi_trend_danger")
    if indicator_context.get("bb_squeeze"):
        # squeeze is a setup for breakout — not a directional reason
        bullish_reasons.append("bb_squeeze_setup")
        bearish_reasons.append("bb_squeeze_setup")
    if indicator_context.get("ma_trend_support") == "BUY":
        bullish_reasons.append("ma_trend_buy")
    elif indicator_context.get("ma_trend_support") == "SELL":
        bearish_reasons.append("ma_trend_sell")
    if indicator_context.get("macd_momentum_up"):
        bullish_reasons.append("macd_momentum_up")
    if indicator_context.get("macd_momentum_down"):
        bearish_reasons.append("macd_momentum_down")

    if not risk_plan_obs.get("invalidation_clear"):
        avoid_reasons.append("invalidation_unclear")

    n_bull = len(bullish_reasons)
    n_bear = len(bearish_reasons)
    n_avoid = len(avoid_reasons)
    score = float((n_bull - n_bear) / max(n_bull + n_bear + 1, 1))

    if n_avoid >= 2:
        label = "AVOID_TRADE"
    elif n_bull >= 4 and n_bear <= 1:
        label = "STRONG_BUY_SETUP"
    elif n_bear >= 4 and n_bull <= 1:
        label = "STRONG_SELL_SETUP"
    elif n_bull >= 2 and n_bear <= 1:
        label = "WEAK_BUY_SETUP"
    elif n_bear >= 2 and n_bull <= 1:
        label = "WEAK_SELL_SETUP"
    elif n_bull == 0 and n_bear == 0:
        label = "UNKNOWN"
    else:
        label = "NO_TRADE"

    return {
        "label": label,
        "score": score,
        "bullish_reasons": list(bullish_reasons),
        "bearish_reasons": list(bearish_reasons),
        "avoid_reasons": list(avoid_reasons),
    }


__all__ = [
    "POLICY_VERSION",
    "build_technical_confluence",
    "empty_technical_confluence",
]
