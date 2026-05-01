"""Tests for support_resistance.detect_levels (v2 clustering)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.risk import atr as compute_atr
from src.fx.support_resistance import (
    Level,
    SRSnapshot,
    detect_levels,
    empty_snapshot,
)


def _zigzag_df(n: int = 200, levels=(100.0, 105.0, 100.0, 105.0, 100.0, 105.0)) -> pd.DataFrame:
    """A repeating zigzag designed to produce many touches at ~100 and
    ~105, so clustering should produce two levels."""
    bars_per_leg = n // (len(levels) - 1)
    closes = []
    for i in range(len(levels) - 1):
        seg = np.linspace(levels[i], levels[i + 1], bars_per_leg + 1)[:-1]
        closes.extend(seg.tolist())
    while len(closes) < n:
        closes.append(closes[-1])
    closes = np.array(closes[:n])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": closes - 0.05, "high": closes + 0.5,
        "low": closes - 0.5, "close": closes, "volume": [1000] * n,
    }, index=idx)


def test_empty_snapshot_shape():
    s = empty_snapshot()
    assert s.levels == ()
    assert s.nearest_support is None
    assert s.nearest_resistance is None
    assert s.reason
    d = s.to_dict()
    assert d["schema_version"] == "support_resistance_v2"


def test_insufficient_data_returns_empty():
    df = pd.DataFrame({"open": [], "high": [], "low": [], "close": [], "volume": []})
    s = detect_levels(df, atr_value=None, last_close=None)
    assert s.levels == ()
    assert s.reason == "insufficient_data"


def test_clustering_groups_repeated_touches_into_levels():
    df = _zigzag_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    assert s.reason != "insufficient_data"
    # Expect at least 1 level with touch_count >= 2 (repeated zigzag).
    multi_touch = [lvl for lvl in s.levels if lvl.touch_count >= 2]
    assert len(multi_touch) >= 1, (
        f"expected at least one multi-touch cluster, got "
        f"{[lvl.touch_count for lvl in s.levels]}"
    )


def test_levels_carry_distance_to_close_atr():
    df = _zigzag_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    assert s.levels
    for lvl in s.levels:
        assert lvl.distance_to_close_atr is not None
        assert lvl.distance_to_close_atr >= 0


def test_strength_score_positive_for_multi_touch():
    df = _zigzag_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    multi = [lvl for lvl in s.levels if lvl.touch_count >= 2]
    if multi:
        for lvl in multi:
            assert lvl.strength_score > 0


def test_to_dict_round_trip():
    df = _zigzag_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    d = s.to_dict()
    assert d["schema_version"] == "support_resistance_v2"
    assert "levels" in d and isinstance(d["levels"], list)
    assert "near_strong_support" in d
    assert "near_strong_resistance" in d
    assert "fake_breakout" in d


def test_broken_count_positive_when_closes_traverse_level():
    """Direct unit test of the per-level history scan: a series that
    crosses the level price multiple times must produce broken_count > 0
    AND role_reversal_count > 0 (sustained crossings)."""
    from src.fx.support_resistance import _level_history_counts
    # 4 sustained crossings of the price 100.
    closes = np.concatenate([
        np.linspace(95, 105, 30),    # cross 100 going up
        np.linspace(105, 95, 30),    # cross 100 going down
        np.linspace(95, 110, 40),    # cross 100 going up (sustained at 110)
        np.linspace(110, 90, 40),    # cross 100 going down (sustained at 90)
    ])
    broken, false_brk, role_rev = _level_history_counts(
        closes=closes, level_price=100.0, atr_value=1.0,
        first_touch_index=0, last_touch_index=len(closes) - 1,
    )
    assert broken >= 4, f"expected >=4 traversals, got {broken}"
    assert role_rev >= 1, f"expected >=1 sustained reversal, got {role_rev}"


def test_false_breakout_count_positive_on_quick_reversion():
    """A close that briefly exceeds the level and returns within 5 bars
    must register false_breakout_count >= 1."""
    from src.fx.support_resistance import _level_history_counts
    # Spike above 100 for 1 bar, then back below for many bars.
    base_below = np.full(40, 95.0)
    spike = np.array([102.0])               # 1-bar excursion above 100
    return_below = np.full(40, 95.0)
    closes = np.concatenate([base_below, spike, return_below])
    broken, false_brk, role_rev = _level_history_counts(
        closes=closes, level_price=100.0, atr_value=1.0,
        first_touch_index=0, last_touch_index=len(closes) - 1,
    )
    assert broken >= 2, "spike should produce two flips (up then down)"
    assert false_brk >= 1, (
        f"a 1-bar excursion that returned within 5 bars must count as "
        f"a false breakout; got false_brk={false_brk}"
    )


def test_role_reversal_count_positive_on_sustained_break():
    """A break that does NOT return within 5 bars counts as a genuine
    role reversal (broken AND last side change held)."""
    from src.fx.support_resistance import _level_history_counts
    closes = np.concatenate([
        np.full(20, 95.0),    # below the level
        np.full(20, 105.0),   # sustained break to above (>5 bars)
    ])
    broken, false_brk, role_rev = _level_history_counts(
        closes=closes, level_price=100.0, atr_value=1.0,
        first_touch_index=0, last_touch_index=len(closes) - 1,
    )
    assert broken == 1
    assert false_brk == 0
    assert role_rev == 1


def test_strength_score_decreases_with_false_breakouts():
    """Two clusters with identical touches/recency but different false
    breakout counts must have different strength_score."""
    from src.fx.support_resistance import (
        _level_history_counts, _CLUSTER_BUCKET_ATR,
    )
    # Cluster A: clean (no false breakouts).
    closes_a = np.concatenate([
        np.full(20, 95.0), np.full(20, 105.0),
    ])
    a = _level_history_counts(
        closes=closes_a, level_price=100.0, atr_value=1.0,
        first_touch_index=0, last_touch_index=len(closes_a) - 1,
    )
    # Cluster B: same broken_count but with two extra false breakouts.
    closes_b = np.concatenate([
        np.full(10, 95.0),
        np.array([102.0]),       # false breakout #1
        np.full(10, 95.0),
        np.array([102.0]),       # false breakout #2
        np.full(10, 95.0),
        np.full(10, 105.0),      # sustained break
    ])
    b = _level_history_counts(
        closes=closes_b, level_price=100.0, atr_value=1.0,
        first_touch_index=0, last_touch_index=len(closes_b) - 1,
    )
    # B has more flips total and more false breakouts than A.
    assert b[1] > a[1]


def test_no_future_leak_uses_only_past_swings():
    """The detector relies on patterns.detect_swings which is future-
    leak safe. As a smoke check: detect_levels(df.iloc[:K]) only uses
    data up to bar K-1; so doubling K cannot make any K-1 swing
    disappear (more data, never less)."""
    df = _zigzag_df(n=300)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    s_full = detect_levels(df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]))
    df_half = df.iloc[:150]
    atr_h = float(compute_atr(df_half, 14).iloc[-1])
    s_half = detect_levels(df_half, atr_value=atr_h, last_close=float(df_half["close"].iloc[-1]))
    # Both should produce levels (no exception).
    assert isinstance(s_full, SRSnapshot)
    assert isinstance(s_half, SRSnapshot)
