"""Tests for the events.json freshness gate (spec §11)."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.fx.calendar import (
    Event,
    calendar_freshness,
    save_events,
    seed_defaults,
)
from src.fx.risk_gate import RiskState, evaluate as gate_evaluate


def _now() -> datetime:
    return datetime(2025, 4, 25, 12, 0, tzinfo=timezone.utc)


def test_freshness_missing(tmp_path: Path):
    h = calendar_freshness(tmp_path / "nope.json", now=_now())
    assert h.status == "missing"
    assert h.is_fresh is False
    assert h.event_count == 0


def test_freshness_empty(tmp_path: Path):
    p = tmp_path / "events.json"
    p.write_text("[]", encoding="utf-8")
    h = calendar_freshness(p, now=_now())
    assert h.status == "empty"
    assert h.is_fresh is False


def test_freshness_unreadable(tmp_path: Path):
    p = tmp_path / "events.json"
    p.write_text("not-valid-json", encoding="utf-8")
    h = calendar_freshness(p, now=_now())
    assert h.status == "unreadable"
    assert h.is_fresh is False


def test_freshness_stale(tmp_path: Path):
    p = tmp_path / "events.json"
    seed_defaults(p, now=_now())
    # Make the file appear 48h old (well past the 24h default threshold)
    old = time.time() - 48 * 3600
    os.utime(p, (old, old))
    h = calendar_freshness(p, max_age_hours=24, now=datetime.now(timezone.utc))
    assert h.status == "stale"
    assert h.is_fresh is False
    assert h.age_hours is not None and h.age_hours > 24


def test_freshness_fresh(tmp_path: Path):
    p = tmp_path / "events.json"
    seed_defaults(p, now=_now())
    h = calendar_freshness(p, max_age_hours=24, now=datetime.now(timezone.utc))
    assert h.status == "fresh"
    assert h.is_fresh is True
    assert h.event_count > 0


def test_gate_blocks_when_calendar_required_and_stale(tmp_path: Path):
    """Live broker mode (require_calendar_fresh=True) → HOLD on stale feed."""
    p = tmp_path / "events.json"
    seed_defaults(p, now=_now())
    old = time.time() - 48 * 3600
    os.utime(p, (old, old))
    h = calendar_freshness(p, max_age_hours=24, now=datetime.now(timezone.utc))

    df = pd.DataFrame(
        {"open": [1.0] * 60, "high": [1.0] * 60, "low": [1.0] * 60,
         "close": [1.0] * 60, "volume": [1] * 60},
        index=pd.date_range("2025-01-01", periods=60, freq="1h", tz="UTC"),
    )
    state = RiskState(df=df, calendar_freshness=h, require_calendar_fresh=True)
    res = gate_evaluate(state)
    assert res.allow_trade is False
    assert res.block.code == "calendar_stale"


def test_gate_does_not_block_when_calendar_not_required(tmp_path: Path):
    """Analyst / research mode → stale calendar surfaces in object only."""
    p = tmp_path / "events.json"
    seed_defaults(p, now=_now())
    old = time.time() - 48 * 3600
    os.utime(p, (old, old))
    h = calendar_freshness(p, max_age_hours=24, now=datetime.now(timezone.utc))

    df = pd.DataFrame(
        {"open": [1.0] * 60, "high": [1.0] * 60, "low": [1.0] * 60,
         "close": [1.0] * 60, "volume": [1] * 60},
        index=pd.date_range("2025-01-01", periods=60, freq="1h", tz="UTC"),
    )
    state = RiskState(df=df, calendar_freshness=h, require_calendar_fresh=False)
    res = gate_evaluate(state)
    assert res.allow_trade is True


def test_gate_blocks_when_calendar_missing_and_required(tmp_path: Path):
    h = calendar_freshness(tmp_path / "nope.json", now=_now())
    df = pd.DataFrame(
        {"open": [1.0] * 60, "high": [1.0] * 60, "low": [1.0] * 60,
         "close": [1.0] * 60, "volume": [1] * 60},
        index=pd.date_range("2025-01-01", periods=60, freq="1h", tz="UTC"),
    )
    state = RiskState(df=df, calendar_freshness=h, require_calendar_fresh=True)
    res = gate_evaluate(state)
    assert res.allow_trade is False
    assert res.block.code == "calendar_stale"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
