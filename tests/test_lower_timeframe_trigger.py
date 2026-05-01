"""Tests for lower_timeframe_trigger.detect_lower_tf_trigger."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.lower_timeframe_trigger import (
    LowerTimeframeTrigger,
    detect_lower_tf_trigger,
    empty_trigger,
)


def _df_lower(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    return pd.DataFrame({
        "open": close - 0.02, "high": close + 0.1,
        "low": close - 0.1, "close": close, "volume": [100] * n,
    }, index=idx)


def test_no_data_returns_unavailable_not_attached():
    out = detect_lower_tf_trigger(
        df_lower_tf=None, lower_tf_interval=None,
        base_bar_close_ts=pd.Timestamp("2025-01-01", tz="UTC"),
    )
    assert out.available is False
    assert out.unavailable_reason == "not_attached"


def test_empty_df_returns_unavailable():
    df = pd.DataFrame(
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    out = detect_lower_tf_trigger(
        df_lower_tf=df, lower_tf_interval="15m",
        base_bar_close_ts=pd.Timestamp("2025-01-01", tz="UTC"),
    )
    assert out.available is False


def test_no_future_leak_strict_leq_filter():
    """Bars with timestamp > base_bar must NEVER influence the result.

    Build two scenarios with the same prefix but different futures;
    the trigger must produce identical output for both."""
    df = _df_lower(n=300)
    cutoff = df.index[100]
    # Build df_no_future = bars whose ts <= cutoff (i.e. bars 0..100 inclusive).
    df_no_future = df[df.index <= cutoff]
    # Build df_with_future = same prefix + a wildly different future tail.
    extra = df.iloc[200:300].copy()
    extra["close"] = 1e6
    extra["high"] = 1e6
    extra["low"] = 1e6
    df_with_future = pd.concat([df_no_future, extra])
    out_with_future = detect_lower_tf_trigger(
        df_lower_tf=df_with_future, lower_tf_interval="15m",
        base_bar_close_ts=cutoff,
    )
    out_no_future = detect_lower_tf_trigger(
        df_lower_tf=df_no_future, lower_tf_interval="15m",
        base_bar_close_ts=cutoff,
    )
    # Same outputs: filtering must drop the future spike.
    d_a = out_with_future.to_dict()
    d_b = out_no_future.to_dict()
    for key in (
        "bullish_pinbar", "bearish_pinbar", "bullish_engulfing",
        "bearish_engulfing", "breakout", "retest",
        "micro_double_bottom", "micro_double_top",
        "bullish_trigger", "bearish_trigger", "bars_used",
        "last_lower_tf_ts",
    ):
        assert d_a[key] == d_b[key], f"future leaked at key {key}"


def test_to_dict_includes_required_keys():
    df = _df_lower()
    cutoff = df.index[-1]
    out = detect_lower_tf_trigger(
        df_lower_tf=df, lower_tf_interval="15m",
        base_bar_close_ts=cutoff,
    )
    d = out.to_dict()
    for key in (
        "schema_version", "interval", "available", "bars_used",
        "last_lower_tf_ts",
        "bullish_pinbar", "bearish_pinbar",
        "bullish_engulfing", "bearish_engulfing",
        "breakout", "retest",
        "micro_double_bottom", "micro_double_top",
        "bullish_trigger", "bearish_trigger",
        "unavailable_reason",
    ):
        assert key in d


def test_visible_bars_trim_at_base_ts():
    df = _df_lower(n=200)
    base_idx = 100
    cutoff = df.index[base_idx]
    out = detect_lower_tf_trigger(
        df_lower_tf=df, lower_tf_interval="15m",
        base_bar_close_ts=cutoff,
    )
    assert out.bars_used <= base_idx + 1
