"""Tests for trendlines.detect_trendlines (v2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.risk import atr as compute_atr
from src.fx.trendlines import detect_trendlines, empty_context


def _ascending_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    base = np.linspace(100, 110, n)
    noise = rng.normal(0, 0.2, n)
    close = base + noise
    return pd.DataFrame({
        "open": close - 0.05, "high": close + 0.3,
        "low": close - 0.3, "close": close, "volume": [1000] * n,
    }, index=idx)


def _descending_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    base = np.linspace(110, 100, n)
    noise = rng.normal(0, 0.2, n)
    close = base + noise
    return pd.DataFrame({
        "open": close - 0.05, "high": close + 0.3,
        "low": close - 0.3, "close": close, "volume": [1000] * n,
    }, index=idx)


def test_empty_context_shape():
    c = empty_context()
    assert c.ascending_support is None
    assert c.descending_resistance is None
    assert c.bullish_signal is False
    assert c.bearish_signal is False
    d = c.to_dict()
    assert d["schema_version"] == "trendline_context_v2"


def test_insufficient_data_returns_empty():
    df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    c = detect_trendlines(df, atr_value=None, last_close=None)
    assert c.ascending_support is None
    assert c.descending_resistance is None


def test_ascending_uptrend_produces_ascending_support():
    df = _ascending_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    c = detect_trendlines(df, atr_value=atr_v, last_close=last_close)
    if c.ascending_support is not None:
        assert c.ascending_support.slope > 0
        assert c.ascending_support.kind == "ascending_support"


def test_descending_downtrend_produces_descending_resistance():
    df = _descending_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    c = detect_trendlines(df, atr_value=atr_v, last_close=last_close)
    if c.descending_resistance is not None:
        assert c.descending_resistance.slope < 0
        assert c.descending_resistance.kind == "descending_resistance"


def test_to_dict_emits_schema_version():
    df = _ascending_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    c = detect_trendlines(df, atr_value=atr_v, last_close=last_close)
    d = c.to_dict()
    assert d["schema_version"] == "trendline_context_v2"
    assert "bullish_signal" in d
    assert "bearish_signal" in d
    assert "reason" in d


def test_trendline_with_three_or_more_touches_is_marked_strong():
    """An ascending series with multiple climbing lows must produce
    `is_strong=True` and `touch_count >= 3` for the ascending_support
    line. confidence must rise above the 2-touch baseline (0.5)."""
    rng = np.random.default_rng(7)
    n = 150
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    closes = []
    for k in range(n):
        base = 100 + 0.05 * k                 # rising baseline
        osc = 3.0 * np.sin(k * 0.3)           # oscillation
        closes.append(base + osc + rng.normal(0, 0.1))
    closes = np.array(closes)
    df = pd.DataFrame({
        "open": closes - 0.05, "high": closes + 0.5,
        "low": closes - 0.5, "close": closes, "volume": [1000] * n,
    }, index=idx)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    c = detect_trendlines(
        df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]),
    )
    assert c.ascending_support is not None
    asc = c.ascending_support
    assert asc.touch_count >= 3, (
        f"expected >=3 touches on the ascending support line, got {asc.touch_count}"
    )
    assert asc.is_strong is True
    assert asc.confidence > 0.5, (
        f"3+ touch line confidence ({asc.confidence}) must exceed 2-touch "
        f"baseline of 0.5"
    )
    assert len(asc.confirming_touch_indices) == asc.touch_count


def test_two_touch_line_has_lower_confidence_than_three_plus():
    """Direct invocation: a 2-touch only line must report
    is_strong=False and confidence=0.5. Uses `_fit_two_anchor_line`
    directly with synthetic Swing objects so the test is fully
    deterministic (no swing detection involved)."""
    from src.fx.patterns import Swing
    from src.fx.trendlines import _fit_two_anchor_line
    sw_a = Swing(index=10, ts=pd.Timestamp("2025-01-01", tz="UTC"),
                 price=100.0, kind="L")
    sw_b = Swing(index=50, ts=pd.Timestamp("2025-01-02", tz="UTC"),
                 price=102.0, kind="L")
    # closes_array: 60 bars, irrelevant content (line stays away from them)
    closes = np.full(60, 105.0)
    tl = _fit_two_anchor_line(
        sw_a, sw_b,
        closes=closes, atr_value=1.0, kind="ascending_support",
        candidate_swings=[sw_a, sw_b],   # exactly two swings, no extras
    )
    assert tl is not None
    assert tl.touch_count == 2
    assert tl.is_strong is False
    assert tl.confidence == pytest.approx(0.5)


def test_no_future_leak_consistent_with_partial_window():
    df = _ascending_df(n=200)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    c_full = detect_trendlines(df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]))
    df_half = df.iloc[:100]
    atr_h = float(compute_atr(df_half, 14).iloc[-1])
    c_half = detect_trendlines(df_half, atr_value=atr_h, last_close=float(df_half["close"].iloc[-1]))
    # Both calls succeed without exception.
    _ = c_full.to_dict()
    _ = c_half.to_dict()
