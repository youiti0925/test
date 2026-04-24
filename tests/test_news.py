"""Tests for the headline fetcher.

No network: we inject a fake ticker factory so yfinance is never called.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.fx.news import Headline, fetch_headlines


class _FakeTicker:
    def __init__(self, symbol: str, news):
        self.symbol = symbol
        self.news = news


def _factory(news):
    return lambda symbol: _FakeTicker(symbol, news)


def test_fetch_headlines_legacy_shape():
    news = [
        {
            "title": "Fed raises rates by 25bp",
            "publisher": "Reuters",
            "providerPublishTime": 1700000000,
            "summary": "The Federal Reserve raised its benchmark rate.",
        }
    ]
    result = fetch_headlines("USDJPY=X", ticker_factory=_factory(news))
    assert len(result) == 1
    h = result[0]
    assert isinstance(h, Headline)
    assert h.title == "Fed raises rates by 25bp"
    assert h.publisher == "Reuters"
    assert h.published_at.year == 2023


def test_fetch_headlines_content_wrapped_shape():
    news = [
        {
            "content": {
                "title": "BOJ holds rates steady",
                "provider": {"displayName": "Bloomberg"},
                "pubDate": "2024-06-14T10:00:00Z",
                "summary": "Bank of Japan kept its policy rate unchanged.",
            }
        }
    ]
    result = fetch_headlines("USDJPY=X", ticker_factory=_factory(news))
    assert len(result) == 1
    assert result[0].publisher == "Bloomberg"
    assert result[0].published_at.tzinfo is not None


def test_fetch_headlines_empty_feed_returns_empty_list():
    result = fetch_headlines("USDJPY=X", ticker_factory=_factory([]))
    assert result == []


def test_fetch_headlines_sorts_newest_first():
    t1 = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    t2 = datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp()
    news = [
        {"title": "older", "publisher": "A", "providerPublishTime": t1},
        {"title": "newer", "publisher": "A", "providerPublishTime": t2},
    ]
    result = fetch_headlines("X", ticker_factory=_factory(news))
    assert [h.title for h in result] == ["newer", "older"]


def test_fetch_headlines_respects_limit():
    news = [
        {"title": f"h{i}", "publisher": "A", "providerPublishTime": 1700000000 + i}
        for i in range(10)
    ]
    result = fetch_headlines("X", limit=3, ticker_factory=_factory(news))
    assert len(result) == 3


def test_fetch_headlines_skips_malformed_items():
    news = [
        "not a dict",
        {"title": "ok", "publisher": "A", "providerPublishTime": 1700000000},
        {"no_title": "skipped"},
    ]
    result = fetch_headlines("X", ticker_factory=_factory(news))
    assert [h.title for h in result] == ["ok"]


def test_fetch_headlines_swallows_factory_errors():
    def broken_factory(symbol):
        raise RuntimeError("yfinance exploded")

    assert fetch_headlines("X", ticker_factory=broken_factory) == []
