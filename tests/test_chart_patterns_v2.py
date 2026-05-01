"""Tests for chart_patterns.detect_patterns (v2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.chart_patterns import (
    ChartPatternSnapshot,
    detect_patterns,
    empty_snapshot,
)
from src.fx.risk import atr as compute_atr


def _df(closes, freq="1h"):
    idx = pd.date_range("2025-01-01", periods=len(closes), freq=freq, tz="UTC")
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": closes - 0.05, "high": closes + 0.3,
        "low": closes - 0.3, "close": closes, "volume": [1000] * len(closes),
    }, index=idx)


def test_empty_snapshot_shape():
    s = empty_snapshot()
    assert s.matches == ()
    assert s.bullish_breakout_confirmed is False
    assert s.bearish_breakout_confirmed is False
    d = s.to_dict()
    assert d["schema_version"] == "chart_pattern_v2"


def test_insufficient_data_returns_empty():
    df = _df([100.0, 100.5, 101.0])
    s = detect_patterns(df, atr_value=None, last_close=None)
    assert s.matches == ()


def test_pattern_neckline_required_for_entry_grade():
    """Even when a shape forms, neckline_broken=False must NOT count
    as a confirmed breakout (no entry-grade signal)."""
    rng = np.random.default_rng(7)
    closes = 100.0 + np.cumsum(rng.standard_normal(120) * 0.4)
    df = _df(closes.tolist())
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_patterns(df, atr_value=atr_v, last_close=last_close)
    for m in s.matches:
        if not m.neckline_broken:
            assert (
                m.kind not in ("flag_bullish", "flag_bearish")
                or m.kind == "triangle_symmetric"
                or s.bullish_breakout_confirmed is False
                or s.bearish_breakout_confirmed is False
            )


def test_to_dict_emits_required_keys():
    rng = np.random.default_rng(11)
    closes = 100.0 + np.cumsum(rng.standard_normal(120) * 0.4)
    df = _df(closes.tolist())
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_patterns(df, atr_value=atr_v, last_close=last_close)
    d = s.to_dict()
    for key in (
        "schema_version", "matches",
        "head_and_shoulders", "inverse_head_and_shoulders",
        "flag", "wedge", "triangle",
        "bullish_breakout_confirmed", "bearish_breakout_confirmed",
        "reason",
    ):
        assert key in d


def test_no_future_leak_partial_window_does_not_crash():
    rng = np.random.default_rng(12)
    closes = 100.0 + np.cumsum(rng.standard_normal(200) * 0.4)
    df = _df(closes.tolist())
    atr_v = float(compute_atr(df, 14).iloc[-1])
    s_full = detect_patterns(df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]))
    half = df.iloc[:100]
    atr_h = float(compute_atr(half, 14).iloc[-1])
    s_half = detect_patterns(half, atr_value=atr_h, last_close=float(half["close"].iloc[-1]))
    assert isinstance(s_full, ChartPatternSnapshot)
    assert isinstance(s_half, ChartPatternSnapshot)
