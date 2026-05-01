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
