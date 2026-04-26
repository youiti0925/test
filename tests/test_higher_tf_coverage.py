"""Pin: backtest_engine + market_timeline cover all live higher-TF base intervals.

The live `HIGHER_INTERVAL_MAP` (higher_timeframe.HIGHER_INTERVAL_MAP) covers
1m/5m/15m/30m/1h/2h/4h/1d. The offline resample maps inside
backtest_engine and market_timeline must cover the same set, otherwise
the higher-TF gate silently degrades to UNKNOWN for those bases and the
gate becomes a no-op.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import _resample_higher_tf as bte_resample
from src.fx.higher_timeframe import HIGHER_INTERVAL_MAP
from src.fx.market_timeline import _resample_higher_tf as mt_resample


def _ohlcv(n: int, freq: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.001, "low": closes * 0.999,
         "close": closes, "volume": [1000] * n},
        index=idx,
    )


@pytest.mark.parametrize(
    "base_interval, freq, n",
    [
        ("30m", "30min", 600),   # 30m → 4h, need ≥10 higher bars
        ("2h", "2h", 300),       # 2h → 1d
        ("4h", "4h", 700),       # 4h → 1W (need ≥10 weekly bars)
    ],
)
def test_offline_resample_covers_intermediate_intervals_in_engine(
    base_interval, freq, n,
):
    df = _ohlcv(n, freq)
    trend = bte_resample(df, base_interval)
    assert trend != "UNKNOWN", (
        f"backtest_engine higher-TF returned UNKNOWN for {base_interval} — "
        "map likely missing entry"
    )


@pytest.mark.parametrize(
    "base_interval, freq, n",
    [
        # n is sized so the resampled higher TF has >= 10 bars:
        # 30m→4h needs 80; 2h→1d needs 240; 4h→1W needs 1680 hours-equivalent
        ("30m", "30min", 600),
        ("2h", "2h", 300),
        ("4h", "4h", 700),
    ],
)
def test_offline_resample_covers_intermediate_intervals_in_timeline(
    base_interval, freq, n,
):
    df = _ohlcv(n, freq)
    trend = mt_resample(df, base_interval)
    assert trend != "UNKNOWN"


def test_live_higher_map_contract_unchanged():
    """If a maintainer drops a key from HIGHER_INTERVAL_MAP, this fails
    so they can decide whether the offline maps need to follow."""
    expected = {"1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"}
    assert set(HIGHER_INTERVAL_MAP.keys()) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
