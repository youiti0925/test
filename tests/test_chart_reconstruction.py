"""Tests for chart_reconstruction multi-scale aggregator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.chart_reconstruction import (
    SCALES,
    SCHEMA_VERSION,
    reconstruct_chart_multi_scale,
)
from src.fx.risk import atr as compute_atr


def _df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def test_short_history_yields_long_unavailable():
    df = _df(150)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    out = reconstruct_chart_multi_scale(
        df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]),
    )
    assert out["schema_version"] == SCHEMA_VERSION
    scales = out["scales"]
    # micro / short available; medium / long unavailable.
    assert scales["micro"]["available"] is True
    assert scales["short"]["available"] is True
    assert scales["medium"]["available"] is False
    assert scales["medium"]["unavailable_reason"] == "short_history"
    assert scales["long"]["available"] is False
    assert scales["long"]["unavailable_reason"] == "short_history"


def test_long_history_makes_all_scales_available():
    df = _df(1100)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    out = reconstruct_chart_multi_scale(
        df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]),
    )
    for sc in ("micro", "short", "medium", "long"):
        assert out["scales"][sc]["available"] is True, (
            f"scale {sc} should be available with 1100 bars"
        )
        assert out["scales"][sc]["bars_used"] == SCALES[sc]


def test_each_scale_emits_required_keys():
    df = _df(400)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    out = reconstruct_chart_multi_scale(
        df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]),
    )
    for sc in ("micro", "short", "medium", "long"):
        s = out["scales"][sc]
        for key in (
            "scale", "bars_required", "bars_used", "available",
            "unavailable_reason", "swing_points",
            "level_zone_candidates", "trendline_candidates",
            "pattern_candidates", "wave_structure", "visual_quality",
            # waveform / shape additions (observation-only)
            "wave_skeleton", "pattern_shape_review",
        ):
            assert key in s, f"scale {sc} missing key {key}"
    assert "wave_shape_review" in out, "cross-scale wave_shape_review missing"
    assert (
        out["wave_shape_review"]["schema_version"]
        == "pattern_shape_review_v1"
    )


def test_no_data_returns_all_unavailable():
    out = reconstruct_chart_multi_scale(
        None, atr_value=None, last_close=None,
    )
    for sc in ("micro", "short", "medium", "long"):
        assert out["scales"][sc]["available"] is False
        assert out["scales"][sc]["unavailable_reason"] == "missing_inputs"


def test_visual_quality_in_unit_interval():
    df = _df(1100)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    out = reconstruct_chart_multi_scale(
        df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]),
    )
    for sc in ("micro", "short", "medium", "long"):
        v = out["scales"][sc]["visual_quality"]
        assert 0.0 <= v <= 1.0


def test_wave_skeleton_attached_per_scale():
    df = _df(1100)
    atr_v = float(compute_atr(df, 14).iloc[-1])
    out = reconstruct_chart_multi_scale(
        df, atr_value=atr_v, last_close=float(df["close"].iloc[-1]),
    )
    for sc in ("short", "medium", "long"):
        skel = out["scales"][sc]["wave_skeleton"]
        assert skel["schema_version"] == "wave_skeleton_v1"
        assert skel["scale"] == sc
        # bars_used positive when scale is available
        assert skel["bars_used"] > 0


def test_no_data_includes_empty_wave_shape_review():
    out = reconstruct_chart_multi_scale(
        None, atr_value=None, last_close=None,
    )
    assert "wave_shape_review" in out
    review = out["wave_shape_review"]
    assert review["schema_version"] == "pattern_shape_review_v1"
    assert review["best_pattern"] is None
