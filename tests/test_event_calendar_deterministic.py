"""PR #18 tests: deterministic event calendar / window policy / audit.

Covers user-mandated checks T1–T13 plus add'l:
  T1  stale calendar still loads events for backtest path
  T2  CORE: same events.json content + different mtime → same trades
  T3  stale status surfaces in run_metadata.calendar.warnings
  T4  events.json missing → policy=disabled_missing + warning
  T5  --no-events → policy=disabled_explicit
  T6  malformed entry → dropped_count + dropped_examples
  T7  events_coverage_start/end in metadata
  T8  per-kind window unit tests (FOMC / BOJ / ECB / BoE / CPI / NFP)
  T9  "Non-Farm Payrolls" title → NFP / 2h
  T10 "BoE Rate Decision (MPC)" title → BOE / 6h
  T11 "Rate Decision" generic spelling → RATE_DECISION / 6h
  T12 live policy: stale → events disabled (engine receives empty tuple)
  T13 audit summary across the live data/events.json (currently 40 events)
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fx.calendar import (
    Event,
    LoadEventsDiagnostics,
    calendar_freshness,
    events_file_sha8,
    load_events,
    load_events_with_diagnostics,
)
from src.fx.risk_gate import (
    HIGH_IMPACT_WINDOWS_HOURS,
    EVENT_WINDOW_POLICY_VERSION,
    _window_hours_for,
    infer_event_kind,
)
from src.fx.backtest_engine import run_engine_backtest
from src.fx.cli import (
    _CALENDAR_POLICY_BACKTEST,
    _CALENDAR_POLICY_DISABLED_EXPLICIT,
    _CALENDAR_POLICY_DISABLED_MISSING,
    _CALENDAR_POLICY_DISABLED_EMPTY,
    _CALENDAR_POLICY_LIVE,
    _load_events_for_run,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_EVENTS_JSON = REPO_ROOT / "data" / "events.json"


def _ohlcv(n: int, *, start="2026-02-04", seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3,
         "close": close, "volume": [1000] * n},
        index=idx,
    )


def _make_calendar(tmp_path, events=None):
    """Fixture: write a small events.json copy and return its path."""
    p = tmp_path / "events.json"
    payload = events or [
        {
            "when": "2026-02-11T13:30:00+00:00", "currency": "USD",
            "title": "Non-Farm Payrolls (Employment Situation)",
            "impact": "high",
        },
        {
            "when": "2026-02-13T13:30:00+00:00", "currency": "USD",
            "title": "CPI (Consumer Price Index)", "impact": "high",
        },
        {
            "when": "2026-03-19T17:00:00+00:00", "currency": "GBP",
            "title": "BoE Rate Decision (MPC)", "impact": "high",
        },
    ]
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def _set_mtime_age(path: Path, age_hours: float) -> None:
    """Backdate file mtime by `age_hours` so calendar_freshness sees stale."""
    target = time.time() - age_hours * 3600
    os.utime(path, (target, target))


# ─── T1 ───────────────────────────────────────────────────────────────────────


def test_stale_calendar_backtest_loads_events(tmp_path):
    """T1: with backtest_warn_but_use, stale events.json STILL loads
    events. The warning surfaces in metadata, but the gate sees the
    calendar."""
    p = _make_calendar(tmp_path)
    _set_mtime_age(p, age_hours=72.0)  # well past 24h threshold
    events, meta, warnings = _load_events_for_run(
        events_path=p, no_events=False,
        policy=_CALENDAR_POLICY_BACKTEST,
    )
    assert len(events) == 3, "stale calendar must still yield events"
    assert meta["calendar_freshness_status"] == "stale"
    assert meta["calendar_policy"] == _CALENDAR_POLICY_BACKTEST
    assert meta["events_loaded"] is True
    assert meta["event_high_enabled"] is True
    assert any("stale" in w for w in warnings)


# ─── T2: CORE deterministic-backtest invariant ─────────────────────────────


def _trade_summary(res):
    return [
        (t.entry_ts, t.exit_ts, t.side, t.exit_reason,
         round(t.pnl, 8), round(t.return_pct, 8), t.bars_held)
        for t in res.trades
    ]


def test_fresh_and_stale_same_events_yield_same_trades(tmp_path):
    """T2 (CORE): same events.json content; only mtime differs (fresh
    vs stale). The backtest must produce identical trades, identical
    event_high counts, identical equity curves. This is the central
    deterministic-backtest invariant of PR #18."""
    p = _make_calendar(tmp_path)
    df = _ohlcv(200, start="2026-02-04", seed=3)

    # fresh run
    _set_mtime_age(p, age_hours=2.0)
    events_fresh, meta_fresh, _ = _load_events_for_run(
        events_path=p, no_events=False,
        policy=_CALENDAR_POLICY_BACKTEST,
    )
    res_fresh = run_engine_backtest(
        df, "USDJPY=X", interval="1h", warmup=60,
        events=events_fresh, calendar=meta_fresh,
    )

    # stale run — same content, just touch mtime backwards
    _set_mtime_age(p, age_hours=72.0)
    events_stale, meta_stale, _ = _load_events_for_run(
        events_path=p, no_events=False,
        policy=_CALENDAR_POLICY_BACKTEST,
    )
    res_stale = run_engine_backtest(
        df, "USDJPY=X", interval="1h", warmup=60,
        events=events_stale, calendar=meta_stale,
    )

    # Trade list, hold reasons, event_high count must all match.
    assert _trade_summary(res_fresh) == _trade_summary(res_stale)
    assert res_fresh.hold_reasons == res_stale.hold_reasons
    fresh_event_high = res_fresh.hold_reasons.get("event_high", 0)
    stale_event_high = res_stale.hold_reasons.get("event_high", 0)
    assert fresh_event_high == stale_event_high
    # Sanity: events were actually loaded in BOTH runs.
    assert len(events_fresh) == len(events_stale) == 3
    # And the file content didn't actually change — sha8 matches.
    assert (
        meta_fresh["events_file_sha8"] == meta_stale["events_file_sha8"]
    )


# ─── T3 / T4 / T5 ────────────────────────────────────────────────────────


def test_run_metadata_calendar_warning_on_stale(tmp_path):
    """T3: stale status flows into the calendar block + warnings."""
    p = _make_calendar(tmp_path)
    _set_mtime_age(p, age_hours=48.0)
    _, meta, warnings = _load_events_for_run(
        events_path=p, no_events=False,
        policy=_CALENDAR_POLICY_BACKTEST,
    )
    assert meta["calendar_freshness_status"] == "stale"
    assert meta["calendar_freshness_age_hours"] >= 47.0
    assert meta["warnings"], "warnings must be non-empty when stale"
    assert any("stale" in w for w in warnings)


def test_missing_events_disabled_with_warning(tmp_path):
    """T4: non-existent events.json → policy=disabled_missing, warning."""
    p = tmp_path / "absent.json"
    events, meta, warnings = _load_events_for_run(
        events_path=p, no_events=False,
        policy=_CALENDAR_POLICY_BACKTEST,
    )
    assert events == ()
    assert meta["calendar_policy"] == _CALENDAR_POLICY_DISABLED_MISSING
    assert meta["events_loaded"] is False
    assert meta["event_high_enabled"] is False
    assert any("not found" in w for w in warnings)


def test_no_events_explicit_in_metadata(tmp_path):
    """T5: --no-events → policy=disabled_explicit (even if file is OK)."""
    p = _make_calendar(tmp_path)
    events, meta, warnings = _load_events_for_run(
        events_path=p, no_events=True,
        policy=_CALENDAR_POLICY_BACKTEST,
    )
    assert events == ()
    assert meta["calendar_policy"] == _CALENDAR_POLICY_DISABLED_EXPLICIT
    assert meta["event_high_enabled"] is False
    assert any("--no-events" in w for w in warnings)


# ─── T6: malformed entry diagnostics ─────────────────────────────────────


def test_load_events_diagnostics_drops_malformed(tmp_path):
    """T6: malformed entries surface in dropped_count + dropped_examples
    rather than disappearing silently."""
    payload = [
        {  # well-formed
            "when": "2026-02-11T13:30:00+00:00", "currency": "USD",
            "title": "OK", "impact": "high",
        },
        {"when": "not-a-timestamp", "currency": "USD",  # malformed when
         "title": "BadWhen", "impact": "high"},
        {"currency": "USD", "title": "MissingWhen",  # missing key
         "impact": "high"},
    ]
    p = tmp_path / "events.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    events, diag = load_events_with_diagnostics(p)
    assert isinstance(diag, LoadEventsDiagnostics)
    assert diag.loaded_count == 1
    assert diag.dropped_count == 2
    assert len(diag.dropped_examples) == 2
    # First dropped should reference BadWhen by title.
    titles = [d.get("title") for d in diag.dropped_examples]
    assert "BadWhen" in titles
    assert "MissingWhen" in titles
    assert diag.parse_errors  # at least one error captured


# ─── T7 ─────────────────────────────────────────────────────────────────


def test_calendar_coverage_start_end_in_metadata(tmp_path):
    """T7: events_coverage_start/end derived from min/max event.when."""
    p = _make_calendar(tmp_path)
    _, meta, _ = _load_events_for_run(
        events_path=p, no_events=False,
        policy=_CALENDAR_POLICY_BACKTEST,
    )
    assert meta["events_coverage_start"] == "2026-02-11T13:30:00+00:00"
    assert meta["events_coverage_end"] == "2026-03-19T17:00:00+00:00"


# ─── T8: per-kind window units ─────────────────────────────────────────


@pytest.mark.parametrize("title,expected_kind,expected_window", [
    ("FOMC Statement & Rate Decision", "FOMC", 6),
    ("BOJ Policy Rate (Monetary Policy Meeting)", "BOJ", 6),
    ("ECB Rate Decision", "ECB", 4),
    ("BoE Rate Decision (MPC)", "BOE", 6),
    ("CPI (Consumer Price Index)", "CPI", 2),
    ("Non-Farm Payrolls (Employment Situation)", "NFP", 2),
])
def test_per_kind_event_window(title, expected_kind, expected_window):
    """T8 + T9 + T10: each canonical kind resolves to its spec window
    even when the title doesn't substring-match the bare keyword.
    Includes the two PR-#18 bug-fix cases:
      - "Non-Farm Payrolls (Employment Situation)" → NFP / 2h (was 4h)
      - "BoE Rate Decision (MPC)"                   → BOE / 6h (was 4h)
    """
    e = Event(
        when=datetime(2026, 1, 1, tzinfo=timezone.utc),
        currency="USD", title=title, impact="high",
    )
    assert infer_event_kind(e) == expected_kind, (
        f"title={title!r} expected kind={expected_kind} got "
        f"{infer_event_kind(e)}"
    )
    assert _window_hours_for(e) == expected_window


def test_rate_decision_generic_spelling_resolves_to_6h():
    """T11: "Rate Decision" without a central-bank brand resolves via
    the RATE_DECISION alias (6h), not via the legacy RATE_DECISION
    underscore-keyword that never matched."""
    e = Event(
        when=datetime(2026, 1, 1, tzinfo=timezone.utc),
        currency="EUR", title="Interest Rate Decision",
        impact="high",
    )
    assert infer_event_kind(e) == "RATE_DECISION"
    assert _window_hours_for(e) == HIGH_IMPACT_WINDOWS_HOURS["RATE_DECISION"]


def test_explicit_event_kind_wins_over_title():
    """Event.kind explicitly given short-circuits title parsing."""
    e = Event(
        when=datetime(2026, 1, 1, tzinfo=timezone.utc),
        currency="USD", title="Some unrelated title",
        impact="high", kind="FOMC",
    )
    assert infer_event_kind(e) == "FOMC"
    assert _window_hours_for(e) == 6


def test_explicit_window_hours_wins_over_kind_and_title():
    """Event.window_hours overrides everything — feed-supplied cadence
    short-circuits the matcher entirely."""
    e = Event(
        when=datetime(2026, 1, 1, tzinfo=timezone.utc),
        currency="USD", title="CPI", impact="high",
        window_hours=8.0,
    )
    assert _window_hours_for(e) == 8.0


# ─── T12: live policy ──────────────────────────────────────────────────


def test_live_policy_stale_calendar_fails_closed(tmp_path):
    """T12: live_require_fresh + stale → events disabled, warning loud.
    Different from backtest_warn_but_use which would still load the
    events. paper / analyze / backtest and live policies stay separate."""
    p = _make_calendar(tmp_path)
    _set_mtime_age(p, age_hours=72.0)
    events, meta, warnings = _load_events_for_run(
        events_path=p, no_events=False,
        policy=_CALENDAR_POLICY_LIVE,
    )
    assert events == ()
    assert meta["calendar_freshness_status"] == "stale"
    assert meta["calendar_policy"] == _CALENDAR_POLICY_LIVE
    assert meta["events_loaded"] is False
    assert meta["event_high_enabled"] is False
    assert any("stale" in w and "disabled" in w for w in warnings)


def test_live_policy_fresh_calendar_loads_events(tmp_path):
    """Mirror image of T12: live + fresh → events flow normally."""
    p = _make_calendar(tmp_path)
    _set_mtime_age(p, age_hours=1.0)
    events, meta, warnings = _load_events_for_run(
        events_path=p, no_events=False,
        policy=_CALENDAR_POLICY_LIVE,
    )
    assert len(events) == 3
    assert meta["events_loaded"] is True
    assert meta["event_high_enabled"] is True
    assert not warnings or all("stale" not in w for w in warnings)


# ─── Run-metadata block end-to-end ────────────────────────────────────


def test_run_metadata_calendar_block_emitted_via_engine(tmp_path):
    """The calendar dict supplied to run_engine_backtest is preserved
    verbatim on RunMetadata.to_dict()['calendar']."""
    p = _make_calendar(tmp_path)
    df = _ohlcv(150, start="2026-02-04", seed=2)
    events, calendar_meta, _ = _load_events_for_run(
        events_path=p, no_events=False,
        policy=_CALENDAR_POLICY_BACKTEST,
    )
    res = run_engine_backtest(
        df, "USDJPY=X", interval="1h", warmup=60,
        events=events, calendar=calendar_meta,
    )
    rm = res.run_metadata.to_dict()
    assert rm["calendar"] is not None
    cal = rm["calendar"]
    for k in (
        "calendar_policy", "calendar_freshness_status",
        "calendar_freshness_age_hours", "events_loaded",
        "events_loaded_count", "events_dropped_count",
        "events_coverage_start", "events_coverage_end",
        "events_file_sha8", "event_high_enabled",
        "event_window_policy_version",
    ):
        assert k in cal, f"calendar block missing key {k}"
    assert cal["event_window_policy_version"] == EVENT_WINDOW_POLICY_VERSION


def test_run_metadata_calendar_default_none_when_not_passed():
    """Backwards-compat: legacy callers that don't pass calendar=
    keep RunMetadata.calendar = None. cmd_trade live exports never broke."""
    df = _ohlcv(120, start="2026-02-04", seed=2)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    assert res.run_metadata.to_dict()["calendar"] is None


# ─── T13: live data/events.json audit ────────────────────────────────


def test_live_events_json_all_resolve_to_known_kind():
    """T13: every entry in the real data/events.json must resolve to
    a known kind under the new matcher (no fallback). Catches future
    title-format regressions before they silently fall to the impact
    bucket."""
    if not LIVE_EVENTS_JSON.exists():
        pytest.skip("data/events.json not present")
    events = load_events(LIVE_EVENTS_JSON)
    fallbacks = [e for e in events if infer_event_kind(e) is None]
    assert not fallbacks, (
        f"{len(fallbacks)} events fell through to the impact-fallback: "
        f"{[(e.title, e.impact) for e in fallbacks[:5]]}"
    )


def test_live_events_json_nfp_now_2h():
    """Regression-fix witness for the legacy NFP / BoE bug: events that
    title-match Non-Farm Payrolls in the live calendar must now report
    a 2h window (was 4h fallback)."""
    if not LIVE_EVENTS_JSON.exists():
        pytest.skip("data/events.json not present")
    events = load_events(LIVE_EVENTS_JSON)
    nfp_events = [e for e in events if "Non-Farm" in e.title]
    assert nfp_events, "expected at least one NFP entry"
    for e in nfp_events:
        assert infer_event_kind(e) == "NFP"
        assert _window_hours_for(e) == 2


def test_live_events_json_boe_now_6h():
    if not LIVE_EVENTS_JSON.exists():
        pytest.skip("data/events.json not present")
    events = load_events(LIVE_EVENTS_JSON)
    boe_events = [e for e in events if "BoE" in e.title]
    assert boe_events
    for e in boe_events:
        assert infer_event_kind(e) == "BOE"
        assert _window_hours_for(e) == 6


# ─── sha8 helper sanity ─────────────────────────────────────────────


def test_events_file_sha8_changes_with_content_not_mtime(tmp_path):
    p = _make_calendar(tmp_path)
    sha_a = events_file_sha8(p)
    _set_mtime_age(p, age_hours=72.0)
    sha_b = events_file_sha8(p)
    assert sha_a == sha_b, "sha8 must depend on content, not mtime"
    p.write_text(p.read_text() + "\n", encoding="utf-8")
    sha_c = events_file_sha8(p)
    assert sha_a != sha_c, "sha8 must change when content changes"
