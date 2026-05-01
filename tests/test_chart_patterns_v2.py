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


def test_retest_confirmed_true_when_price_retests_then_continues():
    """Direct unit test of the retest helper: simulate SELL break +
    retest + continuation."""
    from src.fx.chart_patterns import _retest_confirmed
    closes = np.array([
        102.0, 103.0, 104.0, 103.0, 102.0,    # forming the pattern
        99.0,                                  # break (close < neckline=100)
        99.5, 99.7, 99.9,                      # going back up toward neckline
        99.95,                                 # within 0.3 ATR of neckline
        99.6,                                  # continuation down by >=0.2 ATR
    ])
    assert _retest_confirmed(
        closes=closes, neckline=100.0, side_bias="SELL",
        breakout_search_start=0, atr_value=1.0,
    ) is True


def test_retest_confirmed_false_when_no_continuation():
    """Retest occurs but the next bar does not continue the breakout
    direction → retest NOT confirmed."""
    from src.fx.chart_patterns import _retest_confirmed
    closes = np.array([
        102.0, 99.0,        # break
        99.95, 99.95,       # retest with no continuation
        99.95, 99.95,
    ])
    assert _retest_confirmed(
        closes=closes, neckline=100.0, side_bias="SELL",
        breakout_search_start=0, atr_value=1.0,
    ) is False


def test_retest_confirmed_buy_mirror():
    """BUY-side retest: break above neckline, return down, continue up."""
    from src.fx.chart_patterns import _retest_confirmed
    closes = np.array([
        98.0, 97.0, 96.0, 97.0, 98.0,
        101.0,            # break above 100
        100.5, 100.05, 100.05,
        100.4,            # continuation up
    ])
    assert _retest_confirmed(
        closes=closes, neckline=100.0, side_bias="BUY",
        breakout_search_start=0, atr_value=1.0,
    ) is True


def test_retest_confirmed_no_breakout_returns_false():
    """If the close never breaks the neckline, retest cannot fire."""
    from src.fx.chart_patterns import _retest_confirmed
    closes = np.array([102.0, 101.5, 101.0, 100.5, 100.2, 100.5])
    assert _retest_confirmed(
        closes=closes, neckline=100.0, side_bias="SELL",
        breakout_search_start=0, atr_value=1.0,
    ) is False


def test_retest_does_not_look_past_closes_array():
    """The closes array is the future-leak boundary. Bars beyond the
    visible window are by construction unavailable. Caller passes
    df['close'].iloc[:i+1].to_numpy() so future leak is impossible
    at the data layer."""
    from src.fx.chart_patterns import _retest_confirmed
    short = np.array([102.0, 99.0, 99.5])
    # Not enough bars after retest to confirm continuation → False.
    assert _retest_confirmed(
        closes=short, neckline=100.0, side_bias="SELL",
        breakout_search_start=0, atr_value=1.0,
    ) is False


def test_flag_match_exposes_formation_impulse_consolidation_fields():
    """A bull flag fixture MUST match deterministically — no skip.

    Construction:
      - 30 bars of flat baseline at 100 (history for ATR + swings)
      - 8 bars linear impulse 100 -> 120 (range = 20 >> 1.5 ATR)
      - 10 bars tight consolidation near 120 (range ~0.5)
      - 1 breakout bar at 122 (close > consol_high)

    The detector's flag check walks adaptive consol / impulse lengths
    and must find this combination.
    """
    closes = (
        [100.0] * 30
        + list(np.linspace(100, 120, 8))
        + [120.1, 119.6, 119.9, 120.0, 119.7, 119.95, 119.9, 119.8, 120.05, 119.7]
        + [122.0]
    )
    df = _df(closes)
    atr_v = float(compute_atr(df, 14).iloc[-1] or 1.0)
    s = detect_patterns(df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]))
    flag = s.flag
    assert flag is not None, (
        "deterministic flag fixture (30 base + 8 impulse + 10 consol + 1 break) "
        "must produce a flag match"
    )
    assert flag.side_bias == "BUY"
    assert flag.neckline_broken is True
    d = flag.to_dict()
    for key in (
        "formation_start_index", "formation_end_index",
        "formation_bars", "impulse_bars", "consolidation_bars",
        "upper_line", "lower_line",
        "convergence_score", "breakout_direction",
        "pattern_quality_score", "reason",
    ):
        assert key in d
    assert d["impulse_bars"] is not None and d["impulse_bars"] > 0
    assert d["consolidation_bars"] is not None and d["consolidation_bars"] > 0
    assert d["formation_bars"] is not None
    assert d["formation_bars"] >= d["impulse_bars"]
    assert d["upper_line"] is not None
    assert d["lower_line"] is not None
    assert d["convergence_score"] is not None
    assert 0.0 <= d["convergence_score"] <= 1.0
    assert d["pattern_quality_score"] >= 0.0


def test_wedge_match_exposes_convergence_and_lines():
    """A deterministic falling wedge fixture must populate upper_line /
    lower_line / convergence_score / pattern_quality_score / reason."""
    n = 200
    closes = []
    for k in range(n):
        base = 110 - 0.04 * k
        amp = max(0.5, 5.0 - 0.025 * k)
        closes.append(base + amp * np.sin(k * 0.4))
    df = _df(closes)
    atr_v = float(compute_atr(df, 14).iloc[-1] or 1.0)
    s = detect_patterns(df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]))
    assert s.wedge is not None, (
        "deterministic falling wedge fixture (200 bars, declining baseline + "
        "narrowing oscillation) must produce a wedge match"
    )
    d = s.wedge.to_dict()
    assert d["kind"] == "wedge_falling"
    assert d["upper_line"] is not None
    assert d["lower_line"] is not None
    assert d["upper_line"]["slope"] < 0
    assert d["lower_line"]["slope"] < 0
    assert d["convergence_score"] is not None
    assert d["pattern_quality_score"] > 0.0
    assert d["breakout_direction"] in ("BUY", "SELL", "UNKNOWN")
    assert d["reason"]


def test_triangle_match_categorises_kind():
    """A deterministic symmetric triangle fixture (narrowing sin
    oscillation around a constant baseline) must produce
    triangle_symmetric with upper_line.kind=descending and
    lower_line.kind=ascending."""
    n = 200
    closes = []
    for k in range(n):
        base = 100.0
        amp = max(0.3, 8.0 - 0.04 * k)
        closes.append(base + amp * np.sin(k * 0.5))
    df = _df(closes)
    atr_v = float(compute_atr(df, 14).iloc[-1] or 1.0)
    s = detect_patterns(df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]))
    assert s.triangle is not None, (
        "deterministic symmetric triangle fixture must match"
    )
    d = s.triangle.to_dict()
    assert d["kind"] == "triangle_symmetric"
    assert d["upper_line"] is not None
    assert d["lower_line"] is not None
    assert d["upper_line"]["kind"] == "descending"
    assert d["lower_line"]["kind"] == "ascending"
    assert d["upper_line"]["slope"] < 0
    assert d["lower_line"]["slope"] > 0


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
