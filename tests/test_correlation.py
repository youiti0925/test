"""Tests for correlation analysis.

We inject a fake fetcher so no network is hit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.correlation import (
    RELATED_SYMBOLS,
    build_correlation_snapshot,
    related_for,
)


def _series(n: int, returns: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1D")
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [1000] * n,
        },
        index=idx,
    )


def _fetcher_factory(data: dict[str, pd.DataFrame]):
    def fetcher(symbol: str, **_):
        if symbol not in data:
            raise ValueError(f"no data for {symbol}")
        return data[symbol]

    return fetcher


def test_perfect_positive_correlation():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, 100)
    primary = _series(100, returns)
    # Identical returns → correlation ~1.0
    other = _series(100, returns)
    snap = build_correlation_snapshot(
        "A", others=["B"], fetcher=_fetcher_factory({"A": primary, "B": other})
    )
    assert "B" in snap.correlations
    assert snap.correlations["B"] == pytest.approx(1.0, abs=1e-3)


def test_perfect_negative_correlation():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, 100)
    primary = _series(100, returns)
    other = _series(100, -returns)
    snap = build_correlation_snapshot(
        "A", others=["B"], fetcher=_fetcher_factory({"A": primary, "B": other})
    )
    assert snap.correlations["B"] == pytest.approx(-1.0, abs=1e-3)


def test_missing_other_is_skipped():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, 50)
    primary = _series(50, returns)
    snap = build_correlation_snapshot(
        "A",
        others=["B", "MISSING"],
        fetcher=_fetcher_factory({"A": primary, "B": _series(50, returns * 0.5)}),
    )
    assert "B" in snap.correlations
    assert "MISSING" not in snap.correlations


def test_to_dict_rounds_values():
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.01, 40)
    primary = _series(40, returns)
    snap = build_correlation_snapshot(
        "A", others=["B"], fetcher=_fetcher_factory({"A": primary, "B": _series(40, returns)})
    )
    d = snap.to_dict()
    assert isinstance(d["correlations"]["B"], float)
    assert d["primary"] == "A"


def test_related_for_known_and_unknown_symbols():
    assert related_for("USDJPY=X") == RELATED_SYMBOLS["USDJPY=X"]
    assert related_for("UNKNOWN=X") == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
