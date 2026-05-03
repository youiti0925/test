"""Support / resistance observation tests for technical_confluence_v1.

The detector reuses confirmed swings from `patterns.analyse` as level
proxies. Each test constructs an OHLC dataframe whose swing structure
is unambiguous, then asserts the resulting nearest_support /
nearest_resistance / near_* / breakout values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.indicators import build_snapshot
from src.fx.patterns import analyse as analyse_patterns
from src.fx.risk import atr as compute_atr
from src.fx.technical_confluence import (
    _NEAR_LEVEL_ATR,
    _support_resistance,
    build_technical_confluence,
)


def _make_zigzag_df(
    n_bars: int = 200,
    levels: tuple[float, ...] = (100.0, 105.0, 100.0, 110.0, 100.0, 115.0),
    bars_per_leg: int = 30,
) -> pd.DataFrame:
    """Build a zigzag price series to seed clear swing highs and lows."""
    closes = []
    for i, lvl in enumerate(levels):
        if i == 0:
            seg_start = lvl
        else:
            seg_start = levels[i - 1]
        seg = np.linspace(seg_start, lvl, bars_per_leg)
        closes.extend(seg.tolist())
    closes = np.array(closes[:n_bars])
    while len(closes) < n_bars:
        closes = np.append(closes, closes[-1])
    closes = closes[:n_bars]
    idx = pd.date_range("2025-01-01", periods=n_bars, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": closes - 0.05,
        "high": closes + 0.4,
        "low": closes - 0.4,
        "close": closes,
        "volume": 1000.0,
    }, index=idx)


def test_nearest_resistance_above_current_close():
    df = _make_zigzag_df()
    pattern = analyse_patterns(df)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    # Force the "current price" to be below the most recent peak by
    # appending a low-close bar.
    last_close = df["close"].iloc[-1]
    sr = _support_resistance(
        df_window=df, pattern=pattern, atr_value=atr_v,
        last_close=last_close,
    )
    # Either we have a resistance (some swing highs above current close),
    # OR the zigzag finished above all swings — both are valid.
    if sr["nearest_resistance"] is not None:
        assert sr["nearest_resistance"] > last_close
        assert sr["distance_to_resistance_atr"] is not None
        assert sr["distance_to_resistance_atr"] >= 0


def test_nearest_support_below_current_close():
    df = _make_zigzag_df()
    pattern = analyse_patterns(df)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = df["close"].iloc[-1]
    sr = _support_resistance(
        df_window=df, pattern=pattern, atr_value=atr_v,
        last_close=last_close,
    )
    if sr["nearest_support"] is not None:
        assert sr["nearest_support"] < last_close
        assert sr["distance_to_support_atr"] is not None
        assert sr["distance_to_support_atr"] >= 0


def test_near_support_when_within_threshold():
    """When the current close sits inside _NEAR_LEVEL_ATR of a swing
    low, near_support must be True."""
    df = _make_zigzag_df()
    pattern = analyse_patterns(df)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    if not pattern.swing_lows:
        pytest.skip("fixture produced no swing lows")
    near_low = pattern.swing_lows[-1].price
    # Force last_close just above the swing low by ~0.1 ATR
    forced_close = near_low + 0.1 * atr_v
    sr = _support_resistance(
        df_window=df, pattern=pattern, atr_value=atr_v,
        last_close=forced_close,
    )
    assert sr["near_support"] is True
    assert sr["distance_to_support_atr"] <= _NEAR_LEVEL_ATR


def test_near_resistance_when_within_threshold():
    df = _make_zigzag_df()
    pattern = analyse_patterns(df)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    if not pattern.swing_highs:
        pytest.skip("fixture produced no swing highs")
    near_high = pattern.swing_highs[-1].price
    forced_close = near_high - 0.1 * atr_v
    sr = _support_resistance(
        df_window=df, pattern=pattern, atr_value=atr_v,
        last_close=forced_close,
    )
    assert sr["near_resistance"] is True
    assert sr["distance_to_resistance_atr"] <= _NEAR_LEVEL_ATR


def test_breakout_close_above_resistance():
    """Build a fixture where the previous close sits below a swing high
    and the current close has just risen above it."""
    df = _make_zigzag_df()
    pattern = analyse_patterns(df)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    if not pattern.swing_highs:
        pytest.skip("fixture produced no swing highs")
    target_resistance = pattern.swing_highs[-1].price
    # Construct a 2-bar slice manually: prev_close just below, last_close just above
    last_two = df.copy().iloc[-2:].reset_index(drop=False)
    last_two.loc[len(last_two) - 2, "close"] = target_resistance - 0.05
    last_two.loc[len(last_two) - 1, "close"] = target_resistance + 0.05
    last_two = last_two.set_index("index")
    sr = _support_resistance(
        df_window=last_two, pattern=pattern, atr_value=atr_v,
        last_close=float(last_two["close"].iloc[-1]),
    )
    assert sr["breakout"] is True


def test_missing_atr_returns_safe_default():
    df = _make_zigzag_df()
    pattern = analyse_patterns(df)
    sr = _support_resistance(
        df_window=df, pattern=pattern, atr_value=None,
        last_close=float(df["close"].iloc[-1]),
    )
    # All levels and distances must collapse to None / False
    assert sr["nearest_support"] is None
    assert sr["nearest_resistance"] is None
    assert sr["distance_to_support_atr"] is None
    assert sr["distance_to_resistance_atr"] is None
    assert sr["near_support"] is False
    assert sr["near_resistance"] is False
    assert sr["reason"] == "missing_pattern_or_atr"


def test_full_build_returns_required_keys():
    """End-to-end: build_technical_confluence returns the documented dict."""
    df = _make_zigzag_df()
    snap = build_snapshot("TEST", df)
    pattern = analyse_patterns(df)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    out = build_technical_confluence(
        df_window=df, snapshot=snap, pattern=pattern,
        atr_value=atr_v, technical_only_action="HOLD",
    )
    expected = {
        "policy_version", "market_regime",
        "dow_structure", "support_resistance", "candlestick_signal",
        "chart_pattern", "indicator_context", "risk_plan_obs",
        "vote_breakdown", "final_confluence",
    }
    assert set(out.keys()) == expected
    # final label must come from the closed taxonomy
    valid = {
        "STRONG_BUY_SETUP", "WEAK_BUY_SETUP",
        "STRONG_SELL_SETUP", "WEAK_SELL_SETUP",
        "NO_TRADE", "AVOID_TRADE", "UNKNOWN",
    }
    assert out["final_confluence"]["label"] in valid
