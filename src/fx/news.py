"""Recent headline fetcher.

Uses yfinance's news endpoint. Works reliably for equities and crypto
(BTC-USD), and is often sparse or empty for FX pairs (e.g., USDJPY=X).
For production FX work, swap in a dedicated feed (Reuters, Bloomberg,
central-bank RSS, forexfactory economic calendar, etc.) while keeping
the Headline dataclass unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import yfinance as yf


@dataclass(frozen=True)
class Headline:
    title: str
    publisher: str
    published_at: datetime
    summary: str

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "publisher": self.publisher,
            "published_at": self.published_at.isoformat(),
            "summary": self.summary,
        }


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _coerce_item(item: dict) -> Headline:
    """yfinance has changed the news payload shape several times.

    Accept both the legacy flat shape and the newer `content`-wrapped shape.
    """
    content = item.get("content") if isinstance(item.get("content"), dict) else item

    title = content.get("title") or item.get("title") or ""
    summary = (
        content.get("summary")
        or content.get("description")
        or item.get("summary")
        or ""
    )

    publisher = "unknown"
    provider = content.get("provider") or item.get("provider")
    if isinstance(provider, dict):
        publisher = provider.get("displayName") or provider.get("name") or publisher
    elif isinstance(provider, str):
        publisher = provider
    elif item.get("publisher"):
        publisher = item["publisher"]

    ts_raw = (
        content.get("pubDate")
        or content.get("displayTime")
        or item.get("providerPublishTime")
        or item.get("pubDate")
    )
    published_at = _parse_timestamp(ts_raw)

    return Headline(
        title=title, publisher=publisher, published_at=published_at, summary=summary
    )


def fetch_headlines(
    symbol: str,
    limit: int = 5,
    ticker_factory=yf.Ticker,
) -> list[Headline]:
    """Return up to `limit` recent headlines, newest first.

    Returns an empty list if the feed is unavailable or yields no usable
    items — callers should treat "no news" as a valid state, not an error.
    """
    try:
        ticker = ticker_factory(symbol)
        raw_news = getattr(ticker, "news", None) or []
    except Exception:  # noqa: BLE001
        return []

    headlines: list[Headline] = []
    for item in raw_news:
        if not isinstance(item, dict):
            continue
        try:
            h = _coerce_item(item)
        except Exception:  # noqa: BLE001
            continue
        if h.title:
            headlines.append(h)

    headlines.sort(key=lambda h: h.published_at, reverse=True)
    return headlines[:limit]
