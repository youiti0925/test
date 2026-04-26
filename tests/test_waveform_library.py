"""Tests for the waveform library (spec §7.2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.waveform_library import (
    WaveformSample,
    build_library,
    read_library,
    write_library,
)


def _ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.005, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": [1000] * n},
        index=idx,
    )


def test_build_library_emits_expected_count():
    df = _ohlcv(200)
    samples = build_library(
        df, symbol="X", timeframe="1h",
        window_bars=60, step_bars=10, forward_horizons=(4, 12),
    )
    # End positions: 60, 70, 80, ..., 200  → 15 windows
    assert len(samples) == 15


def test_build_library_signature_window_excludes_future_labels():
    """The signature uses window bars only; forward labels look beyond."""
    df = _ohlcv(200, seed=2)
    samples = build_library(
        df, symbol="X", timeframe="1h",
        window_bars=60, step_bars=20, forward_horizons=(4, 24),
    )
    # Pick a sample firmly in the middle so all labels exist.
    s = samples[3]
    # The signature vector length matches window_bars.
    assert s.signature.length == 60
    # Future labels are nonzero (synthetic data is noisy).
    assert s.forward_returns_pct[4] is not None
    assert s.forward_returns_pct[24] is not None


def test_build_library_too_close_to_right_edge_yields_none_labels():
    df = _ohlcv(70)
    samples = build_library(
        df, symbol="X", timeframe="1h",
        window_bars=60, step_bars=1, forward_horizons=(4, 24),
    )
    # The very last sample ends right at the edge → both labels None
    assert samples[-1].forward_returns_pct[24] is None


def test_round_trip_jsonl(tmp_path: Path):
    df = _ohlcv(200, seed=3)
    samples = build_library(
        df, symbol="X", timeframe="1h",
        window_bars=60, step_bars=20, forward_horizons=(4, 12, 24),
    )
    p = tmp_path / "lib.jsonl"
    n = write_library(p, samples)
    assert n == len(samples)
    loaded = read_library(p)
    assert len(loaded) == len(samples)
    # Spot-check a sample
    a, b = samples[0], loaded[0]
    np.testing.assert_allclose(a.signature.vector, b.signature.vector, atol=1e-12)
    assert a.symbol == b.symbol
    assert a.start_ts == b.start_ts
    assert a.forward_returns_pct == b.forward_returns_pct


def test_append_extends_library(tmp_path: Path):
    df = _ohlcv(200, seed=4)
    samples = build_library(
        df, symbol="X", timeframe="1h",
        window_bars=60, step_bars=20, forward_horizons=(4,),
    )
    p = tmp_path / "lib.jsonl"
    write_library(p, samples)
    write_library(p, samples, append=True)
    loaded = read_library(p)
    assert len(loaded) == 2 * len(samples)


def test_build_library_rejects_tiny_window():
    df = _ohlcv(200)
    with pytest.raises(ValueError):
        build_library(df, symbol="X", timeframe="1h", window_bars=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
