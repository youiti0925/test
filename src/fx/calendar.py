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
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


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
