"""Tests for the economic calendar module."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.fx.calendar import (
    Event,
    currencies_for,
    load_events,
    save_events,
    seed_defaults,
    upcoming_for_symbol,
)


def _now() -> datetime:
    return datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_currencies_for_known_symbol():
    assert "USD" in currencies_for("USDJPY=X")
    assert "JPY" in currencies_for("USDJPY=X")


def test_currencies_for_unknown_symbol_empty():
    assert currencies_for("UNKNOWN") == []


def test_save_load_round_trip(tmp_path: Path):
    path = tmp_path / "events.json"
    events = [
        Event(
            when=_now() + timedelta(hours=1),
            currency="USD",
            title="FOMC",
            impact="high",
            forecast="5.25%",
            previous="5.00%",
        )
    ]
    save_events(path, events)
    loaded = load_events(path)
    assert len(loaded) == 1
    assert loaded[0].title == "FOMC"
    assert loaded[0].currency == "USD"


def test_load_missing_file_returns_empty(tmp_path: Path):
    assert load_events(tmp_path / "nope.json") == []


def test_upcoming_filters_by_currency():
    now = _now()
    events = [
        Event(when=now + timedelta(hours=1), currency="USD", title="A", impact="high"),
        Event(when=now + timedelta(hours=2), currency="GBP", title="B", impact="high"),
    ]
    result = upcoming_for_symbol(events, "USDJPY=X", within_hours=24, now=now)
    assert [e.title for e in result] == ["A"]


def test_upcoming_filters_by_horizon():
    now = _now()
    events = [
        Event(when=now + timedelta(hours=1), currency="USD", title="soon", impact="high"),
        Event(when=now + timedelta(hours=50), currency="USD", title="later", impact="high"),
    ]
    result = upcoming_for_symbol(events, "USDJPY=X", within_hours=24, now=now)
    assert [e.title for e in result] == ["soon"]


def test_upcoming_filters_by_impact():
    now = _now()
    events = [
        Event(when=now + timedelta(hours=1), currency="USD", title="low", impact="low"),
        Event(when=now + timedelta(hours=2), currency="USD", title="high", impact="high"),
    ]
    result = upcoming_for_symbol(
        events, "USDJPY=X", within_hours=24, min_impact="medium", now=now
    )
    assert [e.title for e in result] == ["high"]


def test_upcoming_excludes_past_events():
    now = _now()
    events = [
        Event(when=now - timedelta(hours=1), currency="USD", title="past", impact="high"),
        Event(when=now + timedelta(hours=1), currency="USD", title="future", impact="high"),
    ]
    result = upcoming_for_symbol(events, "USDJPY=X", now=now)
    assert [e.title for e in result] == ["future"]


def test_seed_defaults_writes_parseable_events(tmp_path: Path):
    path = tmp_path / "events.json"
    seeded = seed_defaults(path, now=_now())
    assert len(seeded) >= 2
    reloaded = load_events(path)
    assert len(reloaded) == len(seeded)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
