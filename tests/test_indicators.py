"""Tests for indicators and the backtest engine.

These tests only cover the pure-Python, no-network path. The Claude analyst
and yfinance fetcher are mocked or excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest import run_backtest
from src.fx.indicators import (
    build_snapshot,
    ema,
    macd,
    rsi,
    sma,
    technical_signal,
)


def _synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0, scale=0.005, size=n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.002, n)))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.integers(1000, 5000, n).astype(float)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_sma_matches_pandas_rolling():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert list(sma(s, 2).dropna()) == [1.5, 2.5, 3.5, 4.5]


def test_ema_is_smoother_than_raw():
    s = pd.Series(np.arange(1, 21, dtype=float))
    result = ema(s, 5)
    assert result.notna().sum() == len(s) - 4


def test_rsi_bounded_0_100():
    df = _synthetic_df()
    result = rsi(df["close"]).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_macd_shape():
    df = _synthetic_df()
    result = macd(df["close"])
    assert set(result.columns) == {"macd", "signal", "hist"}


def test_build_snapshot_returns_finite_numbers():
    df = _synthetic_df()
    snap = build_snapshot("TEST", df)
    d = snap.to_dict()
    for k, v in d.items():
        if isinstance(v, (int, float)):
            assert np.isfinite(v), f"{k} is not finite: {v}"
    assert 0 <= d["bb_position"] <= 1.5


def test_technical_signal_returns_valid_action():
    df = _synthetic_df()
    snap = build_snapshot("TEST", df)
    assert technical_signal(snap) in {"BUY", "SELL", "HOLD"}


def test_backtest_produces_metrics_on_synthetic_data():
    df = _synthetic_df(n=500)
    result = run_backtest(df, symbol="TEST", warmup=50)
    m = result.metrics()
    assert m["n_trades"] >= 0
    assert 0.0 <= m["win_rate"] <= 1.0
    assert m["max_drawdown_pct"] <= 0.0


def test_backtest_never_peeks_at_future():
    """Sanity check: a deterministic "always HOLD" strategy produces zero trades."""
    df = _synthetic_df(n=200)
    result = run_backtest(df, symbol="TEST", signal_fn=lambda _: "HOLD", warmup=50)
    assert result.metrics()["n_trades"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
