"""Economic-event calendar.

Deliberately simple: events are stored in a local JSON file so you can
seed them from whatever source you trust (ForexFactory scrape, investing.com,
central-bank RSS, paid feed). The analyst just needs to see
*what high-impact events are coming up for the traded currencies*.

Seed with `fx_events.seed_defaults()` for a demo-quality list of known
recurring events (FOMC, BOJ, ECB, BoE). In production, refresh daily from
a real source.

Event schema (one object per entry in the JSON array):
    {
      "when": "2025-05-01T18:00:00+00:00",
      "currency": "USD",
      "title": "FOMC Statement & Rate Decision",
      "impact": "high",   # low | medium | high
      "forecast": "5.25%",
      "previous": "5.25%"
    }

Freshness guarantee
-------------------
Production live/demo trading must NEVER run on a stale calendar — missing
a FOMC decision because events.json was last updated three weeks ago is
exactly the kind of accident the rest of the safety stack is designed
to prevent. `calendar_freshness` returns a structured status the
Decision Engine can fail-closed on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Literal


# Default staleness threshold per spec §11.
# Calendars older than this should not be trusted for live entries.
CALENDAR_MAX_AGE_HOURS = 24.0

FreshnessStatus = Literal["fresh", "stale", "missing", "empty", "unreadable"]


@dataclass(frozen=True)
class Event:
    when: datetime
    currency: str
    title: str
    impact: str
    forecast: str | None = None
    previous: str | None = None

    def to_dict(self) -> dict:
        return {
            "when": self.when.isoformat(),
            "currency": self.currency,
            "title": self.title,
            "impact": self.impact,
            "forecast": self.forecast,
            "previous": self.previous,
        }


# Symbol → list of currency codes the instrument is exposed to.
SYMBOL_CURRENCIES: dict[str, list[str]] = {
    "USDJPY=X": ["USD", "JPY"],
    "EURUSD=X": ["EUR", "USD"],
    "GBPUSD=X": ["GBP", "USD"],
    "AUDUSD=X": ["AUD", "USD"],
    "USDCHF=X": ["USD", "CHF"],
    "USDCAD=X": ["USD", "CAD"],
    "EURJPY=X": ["EUR", "JPY"],
    "GBPJPY=X": ["GBP", "JPY"],
    "BTC-USD": ["USD", "CRYPTO"],
    "ETH-USD": ["USD", "CRYPTO"],
}


def currencies_for(symbol: str) -> list[str]:
    return SYMBOL_CURRENCIES.get(symbol, [])


def load_events(path: Path) -> list[Event]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    events: list[Event] = []
    for item in raw:
        try:
            events.append(
                Event(
                    when=datetime.fromisoformat(item["when"]),
                    currency=item["currency"],
                    title=item["title"],
                    impact=item.get("impact", "medium"),
                    forecast=item.get("forecast"),
                    previous=item.get("previous"),
                )
            )
        except (KeyError, ValueError):
            continue
    events.sort(key=lambda e: e.when)
    return events


def save_events(path: Path, events: Iterable[Event]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [e.to_dict() for e in events]
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def upcoming_for_symbol(
    events: list[Event],
    symbol: str,
    within_hours: int = 24,
    min_impact: str = "medium",
    now: datetime | None = None,
) -> list[Event]:
    """Filter events to those affecting `symbol` within the next window."""
    currencies = set(currencies_for(symbol))
    if not currencies:
        return []
    now = now or datetime.now(timezone.utc)
    horizon = now + timedelta(hours=within_hours)
    impact_rank = {"low": 0, "medium": 1, "high": 2}
    min_rank = impact_rank.get(min_impact, 1)

    return [
        e
        for e in events
        if e.currency in currencies
        and now <= e.when <= horizon
        and impact_rank.get(e.impact, 0) >= min_rank
    ]


@dataclass(frozen=True)
class CalendarFreshness:
    """Health snapshot of the events.json file used by the gate.

    `status`:
      - "fresh"      file exists, has events, last-modified within max_age_hours
      - "stale"      file exists but mtime older than max_age_hours
      - "empty"      file exists, is readable, but contains zero events
      - "missing"    file does not exist
      - "unreadable" file exists but JSON parse failed

    `last_updated_at` is the file mtime (UTC) when known.
    """

    status: FreshnessStatus
    path: str
    last_updated_at: datetime | None
    age_hours: float | None
    event_count: int
    next_24h: int
    next_7d: int
    max_age_hours: float
    detail: str | None = None

    @property
    def is_fresh(self) -> bool:
        return self.status == "fresh"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "path": self.path,
            "last_updated_at": (
                self.last_updated_at.isoformat() if self.last_updated_at else None
            ),
            "age_hours": (
                round(self.age_hours, 2) if self.age_hours is not None else None
            ),
            "event_count": self.event_count,
            "next_24h": self.next_24h,
            "next_7d": self.next_7d,
            "max_age_hours": self.max_age_hours,
            "detail": self.detail,
        }


def calendar_freshness(
    path: Path,
    *,
    max_age_hours: float = CALENDAR_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> CalendarFreshness:
    """Return a structured health view of the calendar file.

    Production callers (live / demo trade) should refuse to enter a new
    position when `status != "fresh"`. The analyst path may downgrade to
    a warning so research workflows stay usable when the user knows the
    feed is stale.
    """
    now = now or datetime.now(timezone.utc)
    p = Path(path)

    if not p.exists():
        return CalendarFreshness(
            status="missing",
            path=str(p),
            last_updated_at=None,
            age_hours=None,
            event_count=0,
            next_24h=0,
            next_7d=0,
            max_age_hours=max_age_hours,
            detail=f"{p} does not exist; run `calendar-seed` or refresh feed",
        )

    try:
        mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    except OSError as e:
        return CalendarFreshness(
            status="unreadable",
            path=str(p),
            last_updated_at=None,
            age_hours=None,
            event_count=0,
            next_24h=0,
            next_7d=0,
            max_age_hours=max_age_hours,
            detail=f"stat failed: {e}",
        )

    age_h = max(0.0, (now - mtime).total_seconds() / 3600.0)

    try:
        events = load_events(p)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        return CalendarFreshness(
            status="unreadable",
            path=str(p),
            last_updated_at=mtime,
            age_hours=age_h,
            event_count=0,
            next_24h=0,
            next_7d=0,
            max_age_hours=max_age_hours,
            detail=f"parse failed: {e}",
        )

    next_24h = sum(1 for e in events if now <= e.when <= now + timedelta(hours=24))
    next_7d = sum(1 for e in events if now <= e.when <= now + timedelta(days=7))

    if not events:
        return CalendarFreshness(
            status="empty",
            path=str(p),
            last_updated_at=mtime,
            age_hours=age_h,
            event_count=0,
            next_24h=0,
            next_7d=0,
            max_age_hours=max_age_hours,
            detail="file parsed but contains zero events",
        )

    if age_h > max_age_hours:
        return CalendarFreshness(
            status="stale",
            path=str(p),
            last_updated_at=mtime,
            age_hours=age_h,
            event_count=len(events),
            next_24h=next_24h,
            next_7d=next_7d,
            max_age_hours=max_age_hours,
            detail=f"last update {age_h:.1f}h ago > {max_age_hours}h threshold",
        )

    return CalendarFreshness(
        status="fresh",
        path=str(p),
        last_updated_at=mtime,
        age_hours=age_h,
        event_count=len(events),
        next_24h=next_24h,
        next_7d=next_7d,
        max_age_hours=max_age_hours,
    )


def seed_defaults(path: Path, now: datetime | None = None) -> list[Event]:
    """Write a tiny placeholder calendar the user can overwrite.

    These are NOT real scheduled events — they're meant as a template showing
    the schema so the pipeline works end-to-end before you plug in a feed.
    """
    now = now or datetime.now(timezone.utc)
    events = [
        Event(
            when=now + timedelta(hours=6),
            currency="USD",
            title="[Template] FOMC Statement",
            impact="high",
            forecast="Unchanged",
            previous="5.25-5.50%",
        ),
        Event(
            when=now + timedelta(hours=30),
            currency="JPY",
            title="[Template] BOJ Policy Rate",
            impact="high",
            forecast="Unchanged",
            previous="0.25%",
        ),
        Event(
            when=now + timedelta(hours=48),
            currency="EUR",
            title="[Template] ECB Press Conference",
            impact="medium",
        ),
    ]
    save_events(path, events)
    return events
