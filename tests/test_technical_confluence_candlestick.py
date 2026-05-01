"""Candlestick detector tests for technical_confluence_v1.

Each test builds the smallest 1- or 2-bar fixture sufficient to drive
the detector and asserts the relevant boolean. The detector itself is
called in isolation (no engine, no trace, no risk gate).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.fx.technical_confluence import _candlestick_signal


def _bar(o: float, h: float, l: float, c: float) -> dict:
    return {"open": o, "high": h, "low": l, "close": c}


def _df(bars: list[dict]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(bars), freq="1h", tz="UTC")
    return pd.DataFrame(bars, index=idx)


def test_bullish_pinbar_detected():
    # Long lower wick, small body near top of range, bullish-ish close
    df = _df([_bar(o=100.0, h=100.2, l=98.0, c=100.05)])
    out = _candlestick_signal(df)
    assert out["bullish_pinbar"] is True
    assert out["bearish_pinbar"] is False


def test_bearish_pinbar_detected():
    # Long upper wick, small body near bottom of range
    df = _df([_bar(o=100.0, h=102.0, l=99.85, c=99.95)])
    out = _candlestick_signal(df)
    assert out["bearish_pinbar"] is True
    assert out["bullish_pinbar"] is False


def test_no_pinbar_when_body_dominates():
    # Big body, small wicks → strong bull, not a pinbar
    df = _df([_bar(o=100.0, h=101.6, l=99.95, c=101.5)])
    out = _candlestick_signal(df)
    assert out["bullish_pinbar"] is False
    assert out["bearish_pinbar"] is False
    assert out["strong_bull_body"] is True


def test_bullish_engulfing_detected():
    # Bar 1: bearish (o=100, c=99). Bar 2: bullish opening below prev close,
    # closing above prev open and below prev high.
    df = _df([
        _bar(o=100.0, h=100.2, l=98.9, c=99.0),
        _bar(o=98.8, h=100.4, l=98.7, c=100.3),
    ])
    out = _candlestick_signal(df)
    assert out["bullish_engulfing"] is True
    assert out["bearish_engulfing"] is False


def test_bearish_engulfing_detected():
    # Bar 1: bullish (o=99, c=100). Bar 2: bearish that engulfs.
    df = _df([
        _bar(o=99.0, h=100.2, l=98.9, c=100.0),
        _bar(o=100.2, h=100.3, l=98.7, c=98.8),
    ])
    out = _candlestick_signal(df)
    assert out["bearish_engulfing"] is True
    assert out["bullish_engulfing"] is False


def test_harami_detected():
    # Bar 1: large body o=100..c=98. Bar 2: small body fully inside.
    df = _df([
        _bar(o=100.0, h=100.5, l=97.5, c=98.0),
        _bar(o=99.0, h=99.4, l=98.6, c=99.2),
    ])
    out = _candlestick_signal(df)
    assert out["harami"] is True


def test_strong_bear_body_detected():
    df = _df([_bar(o=101.5, h=101.6, l=99.9, c=100.0)])
    out = _candlestick_signal(df)
    assert out["strong_bear_body"] is True
    assert out["strong_bull_body"] is False


def test_rejection_wick_on_close():
    # Bull close but with a big upper wick (rejection of higher prices)
    df = _df([_bar(o=100.0, h=102.0, l=99.95, c=100.4)])
    out = _candlestick_signal(df)
    assert out["rejection_wick"] is True


def test_zero_range_bar_returns_all_false():
    # Doji with literally no range — should not crash, all False
    df = _df([_bar(o=100.0, h=100.0, l=100.0, c=100.0)])
    out = _candlestick_signal(df)
    assert all(v is False for v in out.values())


def test_empty_df_returns_all_false():
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    out = _candlestick_signal(df)
    assert all(v is False for v in out.values())
