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


# ──────── deterministic real-data fixtures ────────────────────────


def _df_with_bullish_engulfing_at_end(n: int = 120) -> pd.DataFrame:
    """Lower-TF df whose last 2 bars are a clear bullish engulfing:
    bar -2 is bearish (open=100.5, close=100.0), bar -1 is bullish
    that engulfs (open=99.95, close=101.0)."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    closes = 100.0 + np.cumsum(rng.standard_normal(n - 2) * 0.05)
    opens = closes - 0.02
    highs = closes + 0.1
    lows = closes - 0.1
    # last two bars
    closes = np.append(closes, [100.0, 101.0])
    opens = np.append(opens, [100.5, 99.95])
    highs = np.append(highs, [100.6, 101.05])
    lows = np.append(lows, [99.9, 99.9])
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [100] * n,
    }, index=idx)


def _df_with_bearish_engulfing_at_end(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    closes = 100.0 + np.cumsum(rng.standard_normal(n - 2) * 0.05)
    opens = closes - 0.02
    highs = closes + 0.1
    lows = closes - 0.1
    closes = np.append(closes, [100.0, 99.0])
    opens = np.append(opens, [99.5, 100.05])
    highs = np.append(highs, [100.1, 100.1])
    lows = np.append(lows, [99.4, 98.95])
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [100] * n,
    }, index=idx)


def test_bullish_engulfing_lower_tf_fires_bullish_trigger():
    df = _df_with_bullish_engulfing_at_end()
    cutoff = df.index[-1]
    out = detect_lower_tf_trigger(
        df_lower_tf=df, lower_tf_interval="15m",
        base_bar_close_ts=cutoff,
    )
    assert out.available is True
    assert out.bullish_engulfing is True
    assert out.bullish_trigger is True
    assert out.bearish_trigger is False


def test_bearish_engulfing_lower_tf_fires_bearish_trigger():
    df = _df_with_bearish_engulfing_at_end()
    cutoff = df.index[-1]
    out = detect_lower_tf_trigger(
        df_lower_tf=df, lower_tf_interval="15m",
        base_bar_close_ts=cutoff,
    )
    assert out.available is True
    assert out.bearish_engulfing is True
    assert out.bearish_trigger is True
    assert out.bullish_trigger is False


def test_lower_tf_trigger_not_attached_when_df_lower_is_none():
    out = detect_lower_tf_trigger(
        df_lower_tf=None, lower_tf_interval="15m",
        base_bar_close_ts=pd.Timestamp("2025-01-01", tz="UTC"),
    )
    assert out.available is False
    assert out.bullish_trigger is False
    assert out.bearish_trigger is False


def test_lower_tf_trigger_filters_future_bars_strictly():
    """parent_bar_ts truncates: bars beyond it must NOT influence the
    trigger output. The "future" portion can be anything."""
    df_visible = _df_with_bullish_engulfing_at_end()
    cutoff = df_visible.index[-1]
    # Build a df with a wild "future" tail that would otherwise produce
    # a different signal.
    idx_future = pd.date_range(
        cutoff + pd.Timedelta("15min"), periods=20, freq="15min",
    )
    future = pd.DataFrame({
        "open": [200.0] * 20, "high": [200.5] * 20,
        "low": [199.5] * 20, "close": [200.0] * 20,
        "volume": [100] * 20,
    }, index=idx_future)
    df_with_future = pd.concat([df_visible, future])
    out_with_future = detect_lower_tf_trigger(
        df_lower_tf=df_with_future, lower_tf_interval="15m",
        base_bar_close_ts=cutoff,
    )
    # Same outputs as the visible-only case.
    expected = detect_lower_tf_trigger(
        df_lower_tf=df_visible, lower_tf_interval="15m",
        base_bar_close_ts=cutoff,
    )
    for key in (
        "bullish_pinbar", "bearish_pinbar",
        "bullish_engulfing", "bearish_engulfing",
        "breakout", "retest",
        "bullish_trigger", "bearish_trigger",
        "bars_used",
    ):
        assert getattr(out_with_future, key) == getattr(expected, key), (
            f"future leaked at key {key}"
        )


def test_v2_evidence_axes_count_includes_lower_tf_trigger():
    """End-to-end: when df_lower_tf produces a bullish_engulfing at the
    final bar, the v2 decide function must list
    `lower_tf_bullish_trigger=True` in its evidence_axes.bullish dict."""
    from datetime import datetime, timezone
    from src.fx.risk import atr as compute_atr
    from src.fx.risk_gate import RiskState
    from src.fx.royal_road_decision_v2 import decide_royal_road_v2

    # Base 1h df (just enough for ATR + swings).
    n = 120
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    closes = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    df = pd.DataFrame({
        "open": closes, "high": closes + 0.3, "low": closes - 0.3,
        "close": closes, "volume": [1000] * n,
    }, index=idx)

    # Lower-TF df ending exactly at the last 1h bar's timestamp.
    df_lower = _df_with_bullish_engulfing_at_end(n=140)
    # Re-anchor so the lower-TF index ends at the 1h cutoff.
    last_1h = df.index[-1]
    new_index = pd.date_range(
        end=last_1h, periods=len(df_lower), freq="15min",
    )
    df_lower = df_lower.copy()
    df_lower.index = new_index

    rs = RiskState(
        df=df, events=(),
        spread_pct=None, sentiment_snapshot=None,
        now=datetime(2025, 1, 5, 12, 0, tzinfo=timezone.utc),
    )

    confluence = {
        "policy_version": "technical_confluence_v1",
        "market_regime": "TREND_UP",
        "dow_structure": {"structure_code": "HL"},
        "support_resistance": {
            "near_support": True, "near_resistance": False,
            "role_reversal": False, "fake_breakout": False,
        },
        "candlestick_signal": {},
        "chart_pattern": {},
        "indicator_context": {},
        "risk_plan_obs": {
            "atr_stop_distance_atr": 2.0,
            "structure_stop_price": float(df["low"].min()),
            "structure_stop_distance_atr": 1.2,
            "rr_atr_based": 1.5,
            "rr_structure_based": 2.5,
            "invalidation_clear": True,
        },
        "vote_breakdown": {},
        "final_confluence": {
            "label": "STRONG_BUY_SETUP", "score": 0.6,
            "bullish_reasons": [], "bearish_reasons": [], "avoid_reasons": [],
        },
    }

    d = decide_royal_road_v2(
        df_window=df,
        technical_confluence=confluence,
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=1.5,
        risk_state=rs,
        atr_value=float(compute_atr(df, 14).iloc[-1] or 0.5),
        last_close=float(df["close"].iloc[-1]),
        symbol="EURUSD=X",
        macro_context=None,
        df_lower_tf=df_lower,
        lower_tf_interval="15m",
        base_bar_close_ts=last_1h,
        mode="balanced",
    )
    adv = d.advisory
    bullish_axes = adv["evidence_axes"]["bullish"]
    assert "lower_tf_bullish_trigger" in bullish_axes
    assert bullish_axes["lower_tf_bullish_trigger"] is True
    # The lower_tf payload in advisory is also populated.
    assert adv["lower_tf_trigger"]["available"] is True
    assert adv["lower_tf_trigger"]["bullish_trigger"] is True
