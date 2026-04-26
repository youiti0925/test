"""Tests for the parameter sweep / stability scoring (spec §10).

Pinned guarantees:
  * `stability_score` returns 0 for empty/no-trade metrics.
  * Disastrous drawdowns reduce the score below a flat run.
  * `calibrate` produces one run per grid combination.
  * `best()` is the highest stability_score, NOT highest raw return.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.calibration import calibrate, stability_score
from src.fx.strategy_config import (
    HAVE_YAML,
    StrategyConfig,
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


def test_stability_score_empty_metrics_returns_zero():
    assert stability_score({}) == 0.0
    assert stability_score({"n_trades": 0}) == 0.0


def test_stability_score_penalises_drawdown():
    safe = {
        "n_trades": 30, "total_return_pct": 5.0,
        "max_drawdown_pct": -3.0, "profit_factor": 1.5,
        "max_consecutive_losses": 3,
    }
    risky = {
        "n_trades": 30, "total_return_pct": 5.0,
        "max_drawdown_pct": -25.0, "profit_factor": 1.5,
        "max_consecutive_losses": 3,
    }
    assert stability_score(safe) > stability_score(risky)


def test_stability_score_rewards_higher_profit_factor():
    a = {
        "n_trades": 20, "total_return_pct": 3.0,
        "max_drawdown_pct": -5.0, "profit_factor": 1.2,
        "max_consecutive_losses": 2,
    }
    b = {**a, "profit_factor": 2.5}
    assert stability_score(b) > stability_score(a)


def test_calibrate_runs_one_per_grid_combo():
    df = _ohlcv(200, seed=1)
    report = calibrate(
        df, symbol="X", interval="1h",
        ranges={"stop_atr": [1.5, 2.0], "tp_atr": [2.5, 3.0]},
    )
    assert len(report.runs) == 4
    overrides = {tuple(sorted(r.overrides.items())) for r in report.runs}
    assert len(overrides) == 4


def test_calibrate_baseline_when_no_ranges():
    df = _ohlcv(200, seed=2)
    report = calibrate(df, symbol="X", interval="1h", ranges=None)
    assert len(report.runs) == 1


def test_calibrate_rejects_unknown_sweep_key():
    df = _ohlcv(200, seed=3)
    with pytest.raises(ValueError):
        calibrate(
            df, symbol="X", interval="1h",
            ranges={"NOT_A_REAL_KEY": [1.0, 2.0]},
        )


def test_calibrate_best_is_highest_stability_score():
    df = _ohlcv(200, seed=4)
    report = calibrate(
        df, symbol="X", interval="1h",
        ranges={"stop_atr": [1.5, 2.0, 2.5]},
    )
    best = report.best()
    assert best is not None
    for r in report.runs:
        assert best.stability_score >= r.stability_score


def test_strategy_config_round_trip(tmp_path: Path):
    cfg = StrategyConfig(symbol="EURUSD=X", interval="15m")
    p = tmp_path / "cfg.json"
    cfg.write(p)
    loaded = StrategyConfig.load(p)
    assert loaded.symbol == "EURUSD=X"
    assert loaded.interval == "15m"


@pytest.mark.skipif(not HAVE_YAML, reason="PyYAML not installed")
def test_strategy_config_yaml_round_trip(tmp_path: Path):
    cfg = StrategyConfig(symbol="USDJPY=X")
    p = tmp_path / "cfg.yaml"
    cfg.write(p)
    loaded = StrategyConfig.load(p)
    assert loaded.symbol == "USDJPY=X"


def test_strategy_config_load_missing_returns_defaults(tmp_path: Path):
    cfg = StrategyConfig.load(tmp_path / "nope.json")
    assert cfg.symbol == "USDJPY=X"
    assert cfg.risk.stop_atr == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
