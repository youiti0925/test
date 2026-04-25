"""Tests for the event-overlay aggregator (spec §6)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx.calendar import Event
from src.fx.event_overlay import (
    KEYWORD_MAP,
    event_keyword,
    overlay_events,
)


def _ohlcv(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """600 hourly bars ≈ 25 days. Big enough for ±24h windows."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.003, n)
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": [1000] * n},
        index=idx,
    )


def test_event_keyword_maps_known_titles():
    assert event_keyword(Event(
        when=datetime.now(timezone.utc), currency="USD",
        title="FOMC Statement", impact="high",
    )) == "FOMC"
    assert event_keyword(Event(
        when=datetime.now(timezone.utc), currency="JPY",
        title="BOJ Press Conference", impact="high",
    )) == "BOJ"
    assert event_keyword(Event(
        when=datetime.now(timezone.utc), currency="USD",
        title="CPI YoY", impact="high",
    )) == "CPI"
    assert event_keyword(Event(
        when=datetime.now(timezone.utc), currency="USD",
        title="Random other event", impact="medium",
    )) is None


def test_overlay_returns_one_row_per_matching_event():
    df = _ohlcv()
    base = df.index[100].to_pydatetime()
    events = [
        Event(when=base, currency="USD", title="CPI YoY", impact="high"),
        Event(when=base + timedelta(hours=200),
              currency="USD", title="FOMC", impact="high"),
        Event(when=base + timedelta(hours=300),
              currency="USD", title="Low impact thing", impact="low"),
    ]
    res = overlay_events(df, events, impact="high")
    assert len(res.rows) == 2
    keywords = {r.event_keyword for r in res.rows}
    assert keywords == {"CPI", "FOMC"}


def test_overlay_filters_by_keyword():
    df = _ohlcv()
    base = df.index[100].to_pydatetime()
    events = [
        Event(when=base, currency="USD", title="CPI YoY", impact="high"),
        Event(when=base + timedelta(hours=200),
              currency="USD", title="FOMC", impact="high"),
    ]
    res = overlay_events(df, events, keyword="CPI")
    assert len(res.rows) == 1
    assert res.rows[0].event_keyword == "CPI"


def test_overlay_no_future_leak_in_returns():
    """`after_24h` must use bars >= event time only; `before_30m` < event time."""
    df = _ohlcv(seed=2)
    anchor = df.index[200]

    # Construct a known step at `anchor + 24h` so we can prove the
    # after_24h column tracks the right bar.
    step_idx = anchor + timedelta(hours=24)
    df = df.copy()
    df.loc[df.index >= step_idx, "close"] *= 1.10  # +10% step at 24h
    df.loc[df.index >= step_idx, "high"] *= 1.10
    df.loc[df.index >= step_idx, "low"] *= 1.10

    ev = Event(when=anchor.to_pydatetime(), currency="USD",
               title="CPI YoY", impact="high")
    res = overlay_events(df, [ev])
    assert len(res.rows) == 1
    row = res.rows[0]
    # The 24h-after return must reflect that +10% step (not exactly 10%
    # because of in-bar noise in the synthetic series, but should be
    # large and positive — at least +5%).
    assert row.returns["after_24h"] is not None
    assert row.returns["after_24h"] > 5.0


def test_overlay_aggregate_handles_empty():
    res = overlay_events(_ohlcv(), [])
    agg = res.aggregate()
    assert agg["n_events"] == 0


def test_overlay_aggregate_basic():
    df = _ohlcv(seed=3)
    base = df.index[120].to_pydatetime()
    events = [
        Event(when=base, currency="USD", title="CPI", impact="high"),
        Event(when=base + timedelta(hours=80),
              currency="USD", title="CPI", impact="high"),
        Event(when=base + timedelta(hours=160),
              currency="USD", title="CPI", impact="high"),
    ]
    res = overlay_events(df, events, keyword="CPI")
    agg = res.aggregate()
    assert agg["n_events"] == 3
    # Most window stats should exist (all 3 events have surrounding bars)
    assert "return_after_24h" in agg["windows"]
    win = agg["windows"]["return_after_24h"]
    assert win["n"] == 3
    assert "share_up" in win and "share_down" in win


def test_overlay_excludes_events_outside_dataframe():
    df = _ohlcv()
    too_early = Event(
        when=df.index[0].to_pydatetime() - timedelta(days=10),
        currency="USD", title="CPI", impact="high",
    )
    too_late = Event(
        when=df.index[-1].to_pydatetime() + timedelta(days=10),
        currency="USD", title="CPI", impact="high",
    )
    res = overlay_events(df, [too_early, too_late])
    assert res.rows == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
