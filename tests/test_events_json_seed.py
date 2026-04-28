"""Tests for the seeded data/events.json and Risk Gate event_high firing.

Backstop the contract that PR #9 promises:
- A real `data/events.json` exists in the repo.
- It loads cleanly through `load_events()` (existing loader, not changed).
- Each event matches the schema `risk_gate.check_high_impact_event` reads.
- When an event window contains the bar timestamp, the gate returns
  a BlockReason("event_high", ...).
- When no event is in window, the gate returns None (no-block).
- A malformed entry inside events.json is dropped silently (existing
  loader behaviour) — the rest of the file still loads.
- run_engine_backtest with these seeded events surfaces event_high
  in `blocked_by` / hold_reasons / decision_trace rule_checks.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.backtest_engine import run_engine_backtest
from src.fx.calendar import (
    Event,
    calendar_freshness,
    load_events,
)
from src.fx.risk_gate import check_high_impact_event


REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS_JSON = REPO_ROOT / "data" / "events.json"


# ─── Static schema / file presence ─────────────────────────────────────────


def test_events_json_exists_in_repo():
    assert EVENTS_JSON.exists(), (
        f"data/events.json must exist so the Risk Gate can fire on real "
        f"backtests; expected at {EVENTS_JSON}"
    )


def test_events_json_loads_cleanly():
    events = load_events(EVENTS_JSON)
    assert len(events) >= 20, (
        f"events.json should carry at least 20 seed events; got {len(events)}"
    )


def test_events_json_has_required_event_kinds():
    """Smoke that the seed covers FOMC / BOJ / ECB / BoE / CPI / NFP — these
    are the kinds that map to the longest HOLD windows in HIGH_IMPACT_WINDOWS_HOURS
    and the ones we want to actually fire on USDJPY/EURUSD/GBPUSD smokes."""
    events = load_events(EVENTS_JSON)
    titles_upper = " ".join(e.title.upper() for e in events)
    for kw in ("FOMC", "BOJ", "ECB", "BOE", "CPI",
               "PAYROLL"):  # NFP often spelled "Non-Farm Payrolls"
        assert kw in titles_upper, (
            f"events.json must include at least one {kw} event; "
            f"none found in titles."
        )


def test_every_seed_event_is_high_impact():
    """The PR #9 seed is intentionally `impact=high` only — these are the
    bars we want the gate to actively block. Lower-impact events can be
    layered later in a separate PR if useful."""
    events = load_events(EVENTS_JSON)
    for e in events:
        assert e.impact == "high", (
            f"every seed event should be impact=high; "
            f"{e.title} ({e.when}) is {e.impact!r}"
        )


def test_every_seed_event_uses_known_currency():
    events = load_events(EVENTS_JSON)
    allowed = {"USD", "JPY", "EUR", "GBP"}
    for e in events:
        assert e.currency in allowed, (
            f"every seed event should use a known major currency; "
            f"{e.title} uses {e.currency!r}"
        )


def test_every_seed_event_has_timezone_aware_when():
    """`risk_gate.check_high_impact_event` compares to a tz-aware now;
    naive datetimes would crash with 'can't subtract offset-naive'."""
    events = load_events(EVENTS_JSON)
    for e in events:
        assert e.when.tzinfo is not None, (
            f"event {e.title} ({e.when}) has a timezone-naive when; "
            f"all seed events must include an explicit offset"
        )


def test_events_json_freshness_is_fresh_on_clean_checkout(tmp_path):
    """File mtime is reset to checkout time, so the freshness check should
    return 'fresh' on a clean repo checkout. (The 24h staleness logic in
    calendar_freshness is unchanged by PR #9 — this is just a sanity check.)
    """
    health = calendar_freshness(EVENTS_JSON)
    # Either fresh, or stale only because the file is older than 24h on
    # this CI/dev machine. Both states are valid behaviour of the existing
    # logic — what we care about is that the file is parseable and not empty.
    assert health.status in {"fresh", "stale"}, (
        f"events.json should be parseable and non-empty; got {health.status} "
        f"({health.detail})"
    )
    assert health.event_count >= 20


# ─── Risk Gate: event_high fires when in window ────────────────────────────


def test_event_high_fires_when_now_inside_window():
    """A FOMC at 18:00 with a 6h window must block trading from 12:00 to
    24:00. Use the seeded title 'FOMC Statement & Rate Decision' so the
    HIGH_IMPACT_WINDOWS_HOURS lookup picks the FOMC=6h window."""
    when = datetime(2025, 9, 17, 18, 0, tzinfo=timezone.utc)
    fomc = Event(
        when=when,
        currency="USD",
        title="FOMC Statement & Rate Decision",
        impact="high",
    )
    # 1 hour after the event — well inside ±6h window.
    block = check_high_impact_event(
        [fomc],
        now=when + timedelta(hours=1),
    )
    assert block is not None
    assert block.code == "event_high"
    assert "FOMC" in block.message


def test_event_high_does_not_fire_outside_window():
    when = datetime(2025, 9, 17, 18, 0, tzinfo=timezone.utc)
    fomc = Event(
        when=when,
        currency="USD",
        title="FOMC Statement & Rate Decision",
        impact="high",
    )
    # 12 hours after — way outside the 6h FOMC window.
    block = check_high_impact_event(
        [fomc],
        now=when + timedelta(hours=12),
    )
    assert block is None


def test_event_high_skips_low_impact_events():
    """The gate is documented to only block on high-impact events."""
    when = datetime(2025, 9, 17, 18, 0, tzinfo=timezone.utc)
    low = Event(
        when=when,
        currency="USD",
        title="FOMC Statement & Rate Decision",
        impact="low",
    )
    block = check_high_impact_event([low], now=when)
    assert block is None


def test_no_block_when_events_list_empty():
    """Pre-PR #9 behaviour: zero events in window → None."""
    block = check_high_impact_event([], now=datetime.now(timezone.utc))
    assert block is None


# ─── Loader is robust to malformed entries ────────────────────────────────


def test_loader_drops_malformed_entries_silently(tmp_path):
    """The existing loader silently skips entries missing required keys.
    Pin that contract so future PRs do not start raising on a typo."""
    p = tmp_path / "events.json"
    p.write_text(json.dumps([
        # Good entry
        {"when": "2025-09-17T18:00:00+00:00",
         "currency": "USD",
         "title": "FOMC Statement",
         "impact": "high"},
        # Missing 'when' — must be skipped, not raise.
        {"currency": "USD",
         "title": "Broken event",
         "impact": "high"},
        # Bad ISO — must be skipped.
        {"when": "not-a-date",
         "currency": "USD",
         "title": "Also broken",
         "impact": "high"},
        # Missing 'title' — must be skipped.
        {"when": "2025-09-17T18:00:00+00:00",
         "currency": "USD",
         "impact": "high"},
    ]), encoding="utf-8")

    events = load_events(p)
    assert len(events) == 1
    assert events[0].title == "FOMC Statement"


# ─── Backtest engine integration ──────────────────────────────────────────


def _ohlcv_around(event_when: datetime, n_before: int = 80, n_after: int = 40,
                  freq: str = "1h", seed: int = 1) -> pd.DataFrame:
    """Generate OHLCV bars centred on `event_when` with a known seed."""
    start = event_when - timedelta(hours=n_before)
    rng = np.random.default_rng(seed)
    n = n_before + n_after
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3,
         "close": close, "volume": [1000] * n},
        index=idx,
    )


def test_backtest_engine_event_high_fires_with_seeded_event():
    """End-to-end: a synthetic OHLCV window containing one in-window FOMC
    bar should produce at least one `event_high` block in
    `result.hold_reasons` and on the corresponding decision_trace bars."""
    event_when = datetime(2025, 9, 17, 18, 0, tzinfo=timezone.utc)
    fomc = Event(
        when=event_when,
        currency="USD",
        title="FOMC Statement & Rate Decision",
        impact="high",
    )
    df = _ohlcv_around(event_when, n_before=80, n_after=40)
    res = run_engine_backtest(
        df, "USDJPY=X",
        interval="1h", warmup=60,
        events=(fomc,),
        capture_traces=True,
    )
    assert res.hold_reasons.get("event_high", 0) >= 1, (
        f"event_high must appear in hold_reasons; got {dict(res.hold_reasons)}"
    )

    # Same event must show up as `event_high` BLOCK on at least one trace.
    blocked_by_event_high = [
        t for t in res.decision_traces
        if "event_high" in t.decision.blocked_by
    ]
    assert blocked_by_event_high, (
        "no decision_trace bar carries event_high in decision.blocked_by; "
        "the gate did not fire on any bar"
    )

    # The corresponding rule_check.event_high.result should be BLOCK on
    # those bars (and PASS on bars outside the window).
    saw_block = saw_pass = False
    for t in res.decision_traces:
        rc = next(rc for rc in t.rule_checks
                  if rc.canonical_rule_id == "event_high")
        if rc.result == "BLOCK":
            saw_block = True
        elif rc.result == "PASS":
            saw_pass = True
    assert saw_block, "event_high never reached BLOCK on any trace"
    assert saw_pass, "event_high never reached PASS — fixture should have both"


def test_backtest_engine_no_block_when_events_empty():
    """Pre-PR #9 baseline: with no events at all, the engine must NOT
    produce event_high anywhere — safety check that this PR did not
    accidentally add unconditional firing."""
    event_when = datetime(2025, 9, 17, 18, 0, tzinfo=timezone.utc)
    df = _ohlcv_around(event_when, n_before=80, n_after=40, seed=2)
    res = run_engine_backtest(
        df, "USDJPY=X",
        interval="1h", warmup=60,
        events=(),  # explicit zero events
        capture_traces=True,
    )
    assert res.hold_reasons.get("event_high", 0) == 0
    for t in res.decision_traces:
        assert "event_high" not in t.decision.blocked_by
        rc = next(rc for rc in t.rule_checks
                  if rc.canonical_rule_id == "event_high")
        # PASS (no events in window) is the only acceptable result here.
        assert rc.result == "PASS"


def test_backtest_engine_event_high_routes_to_blocked_by_distribution():
    """trace-stats (PR #6) folds blocked_by into a distribution. After PR #9
    the same path must produce a positive `event_high` count when events
    are loaded — confirming the seed is actually wired into the existing
    aggregator without changes to it."""
    from src.fx.decision_trace_stats import aggregate_stats
    from src.fx.decision_trace_io import export_run

    event_when = datetime(2025, 9, 17, 18, 0, tzinfo=timezone.utc)
    fomc = Event(
        when=event_when,
        currency="USD",
        title="FOMC Statement & Rate Decision",
        impact="high",
    )
    df = _ohlcv_around(event_when, n_before=80, n_after=40, seed=3)
    res = run_engine_backtest(
        df, "USDJPY=X",
        interval="1h", warmup=60,
        events=(fomc,),
        capture_traces=True,
    )
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        paths = export_run(res, out_dir=Path(td))
        stats = aggregate_stats(paths["decision_traces"])
    bb = stats["blocked_by_distribution"]
    assert bb.get("event_high", 0) >= 1, (
        f"trace-stats blocked_by_distribution must report event_high; "
        f"got {bb}"
    )
    # rule_result_distribution.event_high must show BOTH PASS and BLOCK
    rrd = stats["rule_result_distribution"]["event_high"]
    assert rrd["BLOCK"] >= 1
    assert rrd["PASS"] >= 1
