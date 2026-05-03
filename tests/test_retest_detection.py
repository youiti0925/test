"""Tests for the shared retest_detection helper."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.retest_detection import RetestResult, detect_retest


def _ts_seq(n: int) -> list[pd.Timestamp]:
    return list(pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC"))


# ─────────── close-based retest ───────────────────────────────────


def test_close_retest_buy_confirmed():
    closes = np.array([
        98.0, 97.0, 96.0, 97.0, 98.0,
        101.0,           # break above 100
        100.5, 100.05,   # retest within tol
        100.4,           # continuation up
    ])
    res = detect_retest(
        closes=closes, highs=None, lows=None, timestamps=None,
        level=100.0, side="BUY",
        breakout_search_start=0, atr_value=1.0, parent_bar_ts=None,
        wick_allowed=False,
    )
    assert isinstance(res, RetestResult)
    assert res.retest_confirmed is True
    assert res.reason == "close_retest_continuation"


def test_close_retest_sell_confirmed():
    closes = np.array([
        102.0, 103.0, 102.0, 99.0,
        99.5, 99.95, 99.6,
    ])
    res = detect_retest(
        closes=closes, highs=None, lows=None, timestamps=None,
        level=100.0, side="SELL",
        breakout_search_start=0, atr_value=1.0, parent_bar_ts=None,
        wick_allowed=False,
    )
    assert res.retest_confirmed is True


def test_close_retest_no_continuation():
    closes = np.array([
        102.0, 99.0, 99.95, 99.95, 99.95, 99.95,
    ])
    res = detect_retest(
        closes=closes, highs=None, lows=None, timestamps=None,
        level=100.0, side="SELL",
        breakout_search_start=0, atr_value=1.0, parent_bar_ts=None,
        wick_allowed=False,
    )
    assert res.retest_confirmed is False
    assert res.reason == "no_continuation_within_max_retest_bars"
    assert res.breakout_index == 1


# ─────────── wick-based retest ────────────────────────────────────


def test_wick_retest_buy_confirmed_when_close_outside_tolerance():
    """The bar's CLOSE is well above the level (so the close-only
    detector would NOT match), but the LOW dipped to within tolerance.
    With wick_allowed=True the retest should fire."""
    closes = np.array([
        99.0, 101.5,        # break above 100 (close=101.5)
        102.0,              # retest bar: close=102 (out of tol),
                            # but low=99.95 (within tol). Continuation: ?
        102.5,              # continuation up (>= 0.2 ATR over 102)
    ])
    highs = np.array([99.5, 102.0, 102.5, 102.6])
    lows = np.array([98.5, 100.5, 99.95, 102.0])
    res = detect_retest(
        closes=closes, highs=highs, lows=lows, timestamps=None,
        level=100.0, side="BUY",
        breakout_search_start=0, atr_value=1.0, parent_bar_ts=None,
        wick_allowed=True,
        close_confirm_required=False,
    )
    assert res.retest_confirmed is True
    assert res.reason == "wick_retest_continuation"


def test_wick_retest_off_when_wick_allowed_false():
    """Same fixture as above, but wick_allowed=False → close-only
    detector cannot find a close within tolerance → False."""
    closes = np.array([99.0, 101.5, 102.0, 102.5])
    highs = np.array([99.5, 102.0, 102.5, 102.6])
    lows = np.array([98.5, 100.5, 99.95, 102.0])
    res = detect_retest(
        closes=closes, highs=highs, lows=lows, timestamps=None,
        level=100.0, side="BUY",
        breakout_search_start=0, atr_value=1.0, parent_bar_ts=None,
        wick_allowed=False,
    )
    assert res.retest_confirmed is False


# ─────────── close_confirm_required ───────────────────────────────


def test_close_confirm_required_blocks_when_retest_close_crosses_back():
    """Wick comes within tol but the close itself crossed BACK through
    the level — close_confirm_required=True must block."""
    closes = np.array([
        99.0, 101.5,
        99.5,         # retest close BACK below 100 → close_confirm fails
        99.0,
    ])
    highs = np.array([99.5, 102.0, 100.05, 99.5])
    lows = np.array([98.5, 100.5, 99.0, 98.8])
    res = detect_retest(
        closes=closes, highs=highs, lows=lows, timestamps=None,
        level=100.0, side="BUY",
        breakout_search_start=0, atr_value=1.0, parent_bar_ts=None,
        wick_allowed=True,
        close_confirm_required=True,
    )
    assert res.retest_confirmed is False


# ─────────── future-leak guard ────────────────────────────────────


def test_retest_helper_no_future_leak():
    """Bars whose timestamp > parent_bar_ts must be ignored. Same
    closes are passed twice; with parent_bar_ts truncating early, the
    later 'future spike' must not cause a confirmation."""
    closes = np.array([
        99.0, 101.5,
        102.0, 102.5,
        # Future bars (must be invisible if parent_bar_ts < their ts)
        99.0, 200.0,
    ])
    ts = _ts_seq(len(closes))
    parent_bar_ts = ts[3]    # cut off at bar index 3 (close=102.5)
    res = detect_retest(
        closes=closes, highs=None, lows=None,
        timestamps=ts,
        level=100.0, side="BUY",
        breakout_search_start=0, atr_value=1.0,
        parent_bar_ts=parent_bar_ts,
        wick_allowed=False,
        close_confirm_required=False,
    )
    # Visible window stops at index 3. Visible_closes = [99, 101.5, 102, 102.5].
    # Breakout at index 1 (101.5 > 100). Looking for a retest within tol
    # in indices 2..3-1=2. closes[2]=102, |102-100|=2.0 > 0.3 tol → no
    # close-based retest within visible window. Result: False, NOT True.
    assert res.retest_confirmed is False
    # And the "200" future spike in closes[5] must not appear anywhere
    # in the result (no breakout_index pointing past visible_window end).
    assert res.breakout_index is not None
    assert res.breakout_index <= 3


def test_retest_helper_no_breakout_returns_false():
    closes = np.array([102.0, 101.5, 101.0, 100.5, 100.2, 100.5])
    res = detect_retest(
        closes=closes, highs=None, lows=None, timestamps=None,
        level=100.0, side="SELL",
        breakout_search_start=0, atr_value=1.0, parent_bar_ts=None,
    )
    assert res.retest_confirmed is False
    assert res.breakout_index is None
    assert res.reason == "no_breakout_in_visible_window"


def test_retest_helper_to_dict_emits_full_schema():
    res = detect_retest(
        closes=[101.0, 99.0, 99.5, 99.95, 99.6],
        highs=None, lows=None, timestamps=None,
        level=100.0, side="SELL",
        breakout_search_start=0, atr_value=1.0, parent_bar_ts=None,
    )
    d = res.to_dict()
    for key in (
        "schema_version", "retest_level", "side",
        "breakout_index", "retest_index", "continuation_index",
        "breakout_ts", "retest_ts", "continuation_ts",
        "tolerance_atr", "continuation_atr", "max_retest_bars",
        "wick_allowed", "close_confirm_required",
        "retest_confirmed", "used_bars_end_ts", "reason",
    ):
        assert key in d


# ─────────── integration: chart_patterns + lower_tf both use it ───


def test_chart_patterns_uses_shared_helper(monkeypatch):
    """Pin: chart_patterns imports from retest_detection."""
    src = open("src/fx/chart_patterns.py").read()
    assert "from .retest_detection import detect_retest" in src


def test_lower_timeframe_trigger_uses_shared_helper():
    src = open("src/fx/lower_timeframe_trigger.py").read()
    assert "from .retest_detection import detect_retest" in src
