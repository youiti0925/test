"""Tests for waveform similarity primitives (spec §7 / §19).

These pin two critical guarantees:
  * Same shape at different price levels is highly similar.
  * Small time-axis warps stay similar under DTW but degrade under
    correlation — exactly the contrast that motivates DTW.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.waveform_matcher import (
    compute_signature,
    correlation_similarity,
    cosine_similarity,
    dtw_distance,
    dtw_similarity,
    normalize_window,
    similarity,
    structure_similarity,
)


def _df_from_closes(closes: np.ndarray) -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    high = closes * 1.001
    low = closes * 0.999
    return pd.DataFrame(
        {"open": closes, "high": high, "low": low, "close": closes,
         "volume": [1000] * n},
        index=idx,
    )


def test_normalize_zscore_invariant_to_level_and_scale():
    a = np.linspace(100, 110, 50)
    b = a * 100  # different price level + scale
    za = normalize_window(a, method="z_score")
    zb = normalize_window(b, method="z_score")
    np.testing.assert_allclose(za, zb, atol=1e-10)


def test_normalize_start_price_centers_at_zero():
    a = np.array([100.0, 101.0, 102.0, 101.0])
    out = normalize_window(a, method="start_price")
    assert out[0] == 0.0
    assert out[-1] == pytest.approx(0.01, abs=1e-6)


def test_cosine_identical_vectors_returns_one():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-9)


def test_cosine_opposite_vectors_returns_zero():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    assert cosine_similarity(v, -v) == pytest.approx(0.0, abs=1e-9)


def test_correlation_constant_vector_returns_neutral():
    """A flat series has zero variance; corr is undefined → neutral."""
    a = np.ones(10)
    b = np.linspace(0, 1, 10)
    assert correlation_similarity(a, b) == 0.5


def test_dtw_identical_distance_zero():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert dtw_distance(v, v) == 0.0
    assert dtw_similarity(v, v) == pytest.approx(1.0, abs=1e-9)


def test_dtw_more_tolerant_than_correlation_to_small_warp():
    """A 2-bar lag should remain very similar under DTW but drop a bit
    under correlation (which compares position-by-position)."""
    base = np.sin(np.linspace(0, 4 * np.pi, 80))
    warped = np.roll(base, 2)
    # DTW with default 10% band easily handles a 2-of-80 (2.5%) shift
    assert dtw_similarity(base, warped) > 0.85
    # Correlation drops noticeably (still > 0.5 because the rolled array
    # contains the same shape, just offset).
    assert correlation_similarity(base, warped) < dtw_similarity(base, warped)


def test_signature_same_shape_different_level_is_similar():
    """Two double-tops at very different price levels should match."""
    # Idealised double top: rise, peak, dip, peak, fall
    shape = np.concatenate([
        np.linspace(0, 1, 15),
        np.linspace(1, 0.7, 5),
        np.linspace(0.7, 1, 5),
        np.linspace(1, 0, 15),
    ])
    cheap = 1.0 + shape * 0.01     # ~$1 priced
    expensive = 150.0 + shape * 1.5  # ~150 yen priced

    a = compute_signature(_df_from_closes(cheap), method="z_score")
    b = compute_signature(_df_from_closes(expensive), method="z_score")
    score = similarity(a, b, method="dtw")
    assert score > 0.85


def test_structure_similarity_perfect_overlap():
    s = ["HH", "HL", "HH", "HL"]
    assert structure_similarity(s, s) == 1.0


def test_structure_similarity_disjoint():
    a = ["HH", "HL", "HH"]
    b = ["LL", "LH", "LL"]
    assert structure_similarity(a, b) == 0.0


def test_signature_round_trip_via_dict():
    closes = np.linspace(100, 110, 60)
    sig = compute_signature(_df_from_closes(closes), method="z_score")
    restored = type(sig).from_dict(sig.to_dict())
    np.testing.assert_allclose(restored.vector, sig.vector, atol=1e-12)
    assert restored.trend_state == sig.trend_state
    assert restored.length == sig.length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
