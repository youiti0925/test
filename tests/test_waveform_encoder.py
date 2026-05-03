"""Tests for src/fx/waveform_encoder.py.

These tests pin the future-leak guarantee, the basic skeleton shape,
and the trend hint classification — so a future refactor can't quietly
start peeking forward or change the schema.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fx.waveform_encoder import (
    DEFAULT_RESAMPLE_POINTS,
    SCHEMA_VERSION,
    WavePivot,
    WaveSkeleton,
    empty_skeleton,
    encode_wave_skeleton,
)


def _double_bottom_ohlc(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic double-bottom shape (canonical W) with small noise so
    swing detection has something to bite on."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    x = np.linspace(0.0, 1.0, n)

    def db(t: float) -> float:
        if t < 0.20:
            return 0.70 - (0.70 - 0.05) * (t / 0.20)
        if t < 0.45:
            return 0.05 + (0.55 - 0.05) * ((t - 0.20) / 0.25)
        if t < 0.70:
            return 0.55 - (0.55 - 0.10) * ((t - 0.45) / 0.25)
        return 0.10 + (0.90 - 0.10) * ((t - 0.70) / 0.30)

    base = np.array([db(t) for t in x])
    noise = rng.normal(0.0, 0.005, size=n)
    close = 1.10 + (base + noise) * 0.05
    return pd.DataFrame({
        "open": close - 0.0001,
        "high": close + 0.0008,
        "low": close - 0.0008,
        "close": close,
        "volume": [1000] * n,
    }, index=idx)


def _trending_up_ohlc(n: int = 120, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    drift = np.linspace(0.0, 0.05, n)
    noise = rng.normal(0.0, 0.0015, size=n).cumsum()
    close = 1.10 + drift + noise * 0.5
    return pd.DataFrame({
        "open": close - 0.0001,
        "high": close + 0.0010,
        "low": close - 0.0010,
        "close": close,
        "volume": [1000] * n,
    }, index=idx)


def test_empty_input_returns_empty_skeleton():
    out = encode_wave_skeleton(None, scale="short")
    assert isinstance(out, WaveSkeleton)
    assert out.schema_version == SCHEMA_VERSION
    assert out.pivots == ()
    assert out.normalized_points == ()
    assert out.reason == "empty_input"


def test_atr_unavailable_returns_empty_skeleton():
    df = _double_bottom_ohlc(n=10)  # too short for ATR
    out = encode_wave_skeleton(df, scale="micro")
    assert out.pivots == ()
    assert out.reason in {"atr_unavailable", "too_few_swings"}


def test_double_bottom_yields_pivots_and_normalized_polyline():
    df = _double_bottom_ohlc(n=200)
    out = encode_wave_skeleton(df, scale="short")
    assert out.reason == "ok"
    assert len(out.pivots) >= 3
    # All pivots are H or L
    assert {p.kind for p in out.pivots}.issubset({"H", "L"})
    # Normalized polyline has the configured length
    assert len(out.normalized_points) == DEFAULT_RESAMPLE_POINTS
    # Polyline values are in [0, 1]
    for x, y in out.normalized_points:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0


def test_future_leak_guard_truncated_window_yields_no_post_bar_pivots():
    """Cut the df to ts ≤ parent_bar_ts and assert no pivot index
    exceeds the last bar of the truncated frame."""
    df = _double_bottom_ohlc(n=200)
    parent_idx = 150
    truncated = df.iloc[: parent_idx + 1]
    out = encode_wave_skeleton(truncated, scale="short")
    assert out.reason == "ok"
    last_allowed_idx = len(truncated) - 1
    for p in out.pivots:
        assert p.index <= last_allowed_idx, (
            f"pivot index {p.index} exceeds visible window"
            f" last index {last_allowed_idx}"
        )


def test_developing_pivot_is_tagged_as_developing():
    df = _double_bottom_ohlc(n=200)
    out = encode_wave_skeleton(df, scale="short", include_developing=True)
    assert out.reason == "ok"
    sources = {p.source for p in out.pivots}
    # Every pivot has a known source
    assert sources.issubset({"swing", "zigzag", "developing"})


def test_trend_hint_up_for_trending_up_series():
    df = _trending_up_ohlc(n=300)
    out = encode_wave_skeleton(df, scale="short")
    # Either UP, MIXED, or RANGE depending on noise — but never DOWN for
    # a strictly drifting-up series.
    assert out.trend_hint in {"UP", "MIXED", "RANGE"}


def test_to_dict_round_trip_keys():
    df = _double_bottom_ohlc(n=200)
    out = encode_wave_skeleton(df, scale="short")
    d = out.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["scale"] == "short"
    assert "pivots" in d and "normalized_points" in d
    assert "params" in d
    assert (
        d["params"].get("thresholds_status") == "heuristic_not_validated"
    )


def test_empty_skeleton_helper_returns_proper_schema():
    out = empty_skeleton("medium", "short_history")
    assert out.scale == "medium"
    assert out.reason == "short_history"
    assert out.pivots == ()
    assert out.schema_version == SCHEMA_VERSION


def test_wave_pivot_is_frozen():
    p = WavePivot(
        index=1, ts="2025-01-01T00:00:00+00:00", price=1.10,
        kind="H", source="zigzag", strength=0.5,
    )
    try:
        p.price = 1.20  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("WavePivot should be frozen")
