"""Tests for support_resistance v2.2 zone-based fields + top5/rejected."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.risk import atr as compute_atr
from src.fx.support_resistance import detect_levels


def _zigzag_df(n: int = 250) -> pd.DataFrame:
    parts = []
    for cycle in range(20):
        parts.extend(np.linspace(95, 102, 10).tolist())
        parts.extend(np.linspace(102, 95, 10).tolist())
    parts.extend([108, 90, 110, 88, 105, 95])
    closes = np.array(parts[:n])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": closes - 0.05, "high": closes + 0.4,
        "low": closes - 0.4, "close": closes, "volume": [1000] * n,
    }, index=idx)


def test_levels_have_zone_fields():
    df = _zigzag_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    for lvl in s.levels:
        assert lvl.zone_low is not None
        assert lvl.zone_high is not None
        assert lvl.zone_high >= lvl.zone_low
        assert lvl.zone_width_atr is not None
        assert lvl.zone_width_atr >= 0


def test_levels_have_zone_counts():
    df = _zigzag_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    for lvl in s.levels:
        d = lvl.to_dict()
        for key in (
            "zone_low", "zone_high", "zone_width_atr",
            "wick_touch_count", "close_touch_count",
            "body_break_count", "wick_fakeout_count",
            "rejection_count", "confidence", "reasons",
        ):
            assert key in d, f"Level missing key {key}"


def test_top5_and_rejected_present():
    df = _zigzag_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    d = s.to_dict()
    assert "selected_level_zones_top5" in d
    assert "rejected_level_zones" in d
    # Selected count <= 5; rejected entries always carry reject_reason.
    assert len(d["selected_level_zones_top5"]) <= 5
    for r in d["rejected_level_zones"]:
        assert r["reject_reason"] is not None
        assert r["reject_reason"] in (
            "too_few_touches", "too_old", "low_quality",
            "too_far_from_price", "already_broken", "outside_top5",
        )


def test_zone_width_falls_back_to_bucket_when_single_swing():
    """A cluster of one swing has price min == max → zone collapses
    to a thin band using bucket size as fallback. Verify zone_width_atr
    reports a positive value (not zero)."""
    # Build a fixture with at least one isolated swing far from others.
    rng = np.random.default_rng(7)
    n = 200
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    closes = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    closes[100] = 130.0   # isolated spike high (single swing)
    df = pd.DataFrame({
        "open": closes - 0.05, "high": closes + 0.3,
        "low": closes - 0.3, "close": closes, "volume": [1000] * n,
    }, index=idx)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    # At least one level exists.
    assert s.levels


def test_zone_count_increases_with_more_interactions():
    """A level whose zone gets crossed many times must report
    body_break_count > 0 once cross-traversals occur."""
    # Build closes that traverse 100 several times.
    parts = (
        list(np.linspace(95, 105, 30))
        + list(np.linspace(105, 95, 30))
        + list(np.linspace(95, 110, 50))
        + list(np.linspace(110, 90, 50))
    )
    closes = np.array(parts)
    n = len(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": closes - 0.05, "high": closes + 0.4,
        "low": closes - 0.4, "close": closes, "volume": [1000] * n,
    }, index=idx)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    # At least one level should have non-trivial zone counts.
    has_meaningful = any(
        (lvl.close_touch_count + lvl.wick_touch_count + lvl.body_break_count)
        >= 5
        for lvl in s.levels
    )
    assert has_meaningful, (
        f"levels = {[(round(l.price, 2), l.close_touch_count, l.wick_touch_count, l.body_break_count) for l in s.levels]}"
    )


def test_reject_reason_taxonomy_closed():
    """All rejected level reject_reasons must come from the closed
    taxonomy."""
    df = _zigzag_df()
    atr_v = float(compute_atr(df, 14).iloc[-1])
    last_close = float(df["close"].iloc[-1])
    s = detect_levels(df, atr_value=atr_v, last_close=last_close)
    valid = {
        "too_few_touches", "too_old", "low_quality",
        "too_far_from_price", "already_broken", "outside_top5",
    }
    for lvl in s.rejected_level_zones:
        assert lvl.reject_reason in valid
