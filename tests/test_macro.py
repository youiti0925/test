"""Tests for the macro-context fetcher (spec §5)."""
from __future__ import annotations

import pandas as pd
import pytest

from src.fx.macro import MACRO_SYMBOLS, MacroSnapshot, fetch_macro_snapshot


def test_macro_snapshot_value_at_uses_asof():
    idx = pd.date_range("2025-01-01", periods=10, freq="1D", tz="UTC")
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                  index=idx, name="us10y")
    snap = MacroSnapshot(base_index=idx, series={"us10y": s}, fetch_errors={})
    # Exact match
    assert snap.value_at("us10y", idx[4]) == 5.0
    # Between bars → most recent prior value
    between = idx[4] + pd.Timedelta(hours=12)
    assert snap.value_at("us10y", between) == 5.0
    # Before any data → None
    early = idx[0] - pd.Timedelta(days=1)
    assert snap.value_at("us10y", early) is None


def test_macro_snapshot_yield_spread():
    idx = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")
    snap = MacroSnapshot(
        base_index=idx,
        series={
            "us10y": pd.Series([4.5, 4.6, 4.4, 4.5, 4.7], index=idx),
            "us_short_yield_proxy": pd.Series([4.0, 4.1, 4.0, 4.1, 4.2], index=idx),
        },
        fetch_errors={},
    )
    spread = snap.yield_spread_long_short(idx[0])
    assert spread == pytest.approx(0.5)


def test_macro_snapshot_missing_slot_returns_none():
    idx = pd.date_range("2025-01-01", periods=3, freq="1D", tz="UTC")
    snap = MacroSnapshot(base_index=idx, series={}, fetch_errors={})
    assert snap.value_at("us10y", idx[1]) is None


def test_fetch_macro_snapshot_with_mock_fetcher():
    """fetch_macro_snapshot calls our fetcher per slot and adapts ^TNX scale."""
    idx = pd.date_range("2025-01-01", periods=10, freq="1D", tz="UTC")

    def fake_fetcher(symbol, **kwargs):
        # ^TNX returns yield * 10 (Yahoo convention); we expect that to be
        # divided back to "real percent" by the fetcher.
        if symbol == "^TNX":
            return pd.DataFrame(
                {"open": [42.5] * 10, "high": [42.5] * 10, "low": [42.5] * 10,
                 "close": [42.5] * 10, "volume": [1] * 10},
                index=idx,
            )
        return pd.DataFrame(
            {"open": [100.0] * 10, "high": [100.0] * 10, "low": [100.0] * 10,
             "close": [100.0] * 10, "volume": [1] * 10},
            index=idx,
        )

    snap = fetch_macro_snapshot(idx, slots=["us10y", "vix"], fetcher=fake_fetcher)
    # 42.5 / 10 = 4.25%
    assert snap.value_at("us10y", idx[2]) == pytest.approx(4.25, abs=0.01)
    assert snap.value_at("vix", idx[2]) == 100.0
    assert snap.fetch_errors == {}


def test_fetch_macro_snapshot_records_errors():
    idx = pd.date_range("2025-01-01", periods=5, freq="1D", tz="UTC")

    def boom(symbol, **kwargs):
        raise ValueError(f"yfinance is angry about {symbol}")

    snap = fetch_macro_snapshot(idx, slots=["us10y", "dxy"], fetcher=boom)
    assert snap.series == {}
    assert "us10y" in snap.fetch_errors
    assert "dxy" in snap.fetch_errors


def test_macro_symbols_contains_expected_slots():
    """If a slot is renamed, the timeline + tests need to follow."""
    expected = {"us10y", "us_short_yield_proxy", "dxy",
                "nikkei", "sp500", "nasdaq", "vix"}
    assert set(MACRO_SYMBOLS.keys()) == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
