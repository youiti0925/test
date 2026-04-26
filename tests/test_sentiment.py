"""Tests for the sentiment subsystem.

All HTTP is mocked — no network. Claude calls are stubbed via injected
fake clients so no API key is required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.sentiment.aggregator import aggregate
from src.sentiment.base import Post, Source
from src.sentiment.collectors.reddit import RedditCollector
from src.sentiment.collectors.rss import RSSCollector
from src.sentiment.collectors.stocktwits import StocktwitsCollector
from src.sentiment.collectors.tradingview import TradingViewCollector
from src.sentiment.collectors.twitter import TwitterCollector
from src.sentiment.refresh import refresh
from src.sentiment.snapshot import (
    SentimentSnapshot,
    load_snapshot,
    snapshot_age_seconds,
    write_snapshot,
)


# --- Fake HTTP helpers ---


@dataclass
class _Resp:
    status_code: int = 200
    _body: str | dict | bytes = ""
    @property
    def text(self) -> str:
        if isinstance(self._body, (dict, list)):
            return json.dumps(self._body)
        if isinstance(self._body, bytes):
            return self._body.decode()
        return self._body
    @property
    def content(self) -> bytes:
        return self.text.encode()
    def json(self):
        if isinstance(self._body, (dict, list)):
            return self._body
        return json.loads(self._body)


def _fake_get(responses: dict[str, _Resp]):
    def _get(url, **kw):
        # match by exact url or by URL prefix
        if url in responses:
            return responses[url]
        for prefix, resp in responses.items():
            if url.startswith(prefix):
                return resp
        return _Resp(status_code=404, _body="")
    return _get


# --- Reddit ---

def test_reddit_returns_posts_from_search_json():
    body = {
        "data": {
            "children": [
                {
                    "data": {
                        "id": "abc123",
                        "title": "USD/JPY breaking out!",
                        "selftext": "Looks bullish",
                        "permalink": "/r/forex/comments/abc123/x",
                        "author": "trader1",
                        "created_utc": 1700000000,
                        "score": 12,
                        "num_comments": 3,
                    }
                }
            ]
        }
    }
    fake = _fake_get({"https://www.reddit.com/r/forex/search.json": _Resp(_body=body)})
    c = RedditCollector(http_get=fake)
    posts = c.fetch("USDJPY=X", limit=10)
    assert len(posts) == 1
    p = posts[0]
    assert p.source == "reddit"
    assert "USD/JPY" in p.text
    assert p.url.startswith("https://reddit.com")


def test_reddit_returns_empty_when_unknown_symbol():
    c = RedditCollector(http_get=_fake_get({}))
    assert c.fetch("UNKNOWN", limit=10) == []


def test_reddit_swallows_http_failure():
    def boom(url, **kw):
        raise ConnectionError("offline")
    c = RedditCollector(http_get=boom)
    assert c.fetch("USDJPY=X", limit=10) == []


# --- Stocktwits ---

def test_stocktwits_parses_messages_and_sentiment():
    body = {
        "messages": [
            {
                "id": 1,
                "body": "I am bullish on USDJPY",
                "created_at": "2025-04-25T10:00:00Z",
                "user": {"username": "alice", "followers": 200},
                "entities": {"sentiment": {"basic": "Bullish"}},
            }
        ]
    }
    fake = _fake_get(
        {"https://api.stocktwits.com/api/2/streams/symbol/USDJPY.json": _Resp(_body=body)}
    )
    c = StocktwitsCollector(http_get=fake)
    posts = c.fetch("USDJPY=X")
    assert len(posts) == 1
    assert posts[0].metadata["user_sentiment"] == "Bullish"


def test_stocktwits_returns_empty_on_bad_status():
    fake = _fake_get({"https://api.stocktwits.com": _Resp(status_code=429, _body="")})
    c = StocktwitsCollector(http_get=fake)
    assert c.fetch("USDJPY=X") == []


# --- TradingView ---

def test_tradingview_fallback_html_parser_picks_up_cards():
    html = """
    <html><body>
    <a class="tv-widget-idea__title" href="/chart/abc/">USDJPY long thesis</a>
    <span class="tv-idea-label-long">Long</span>
    <a class="tv-widget-idea__title" href="/chart/def/">USDJPY range bound</a>
    <span class="tv-idea-label-short">Short</span>
    </body></html>
    """
    fake = _fake_get(
        {"https://www.tradingview.com/symbols/USDJPY/ideas/": _Resp(_body=html)}
    )
    c = TradingViewCollector(http_get=fake)
    posts = c.fetch("USDJPY=X", limit=10)
    assert len(posts) == 2
    biases = {p.metadata.get("bias") for p in posts}
    assert {"long", "short"} <= biases


# --- Twitter ---

def test_twitter_returns_empty_when_all_backends_fail():
    fake = _fake_get({})
    c = TwitterCollector(_http_get=fake)
    assert c.fetch("USDJPY=X", limit=10) == []
    assert c.backend_used == "none"


def test_twitter_skips_when_no_influencers_for_symbol():
    c = TwitterCollector(_http_get=_fake_get({}))
    assert c.fetch("UNKNOWN=X", limit=10) == []


# --- RSS ---

def test_rss_filters_items_by_keyword():
    rss_xml = b"""<?xml version="1.0"?>
    <rss><channel>
      <item><title>BoJ holds rates</title>
            <description>Bank of Japan keeps rates unchanged</description>
            <pubDate>Mon, 21 Apr 2025 10:00:00 +0000</pubDate>
            <link>https://example.com/1</link></item>
      <item><title>Random earnings beat</title>
            <description>Apple reports earnings</description>
            <pubDate>Mon, 21 Apr 2025 11:00:00 +0000</pubDate>
            <link>https://example.com/2</link></item>
    </channel></rss>"""
    fake = _fake_get({"https://example.com/feed": _Resp(_body=rss_xml)})
    c = RSSCollector(feeds=["https://example.com/feed"], http_get=fake)
    posts = c.fetch("USDJPY=X")
    assert len(posts) == 1
    assert "BoJ" in posts[0].text


# --- Aggregator ---

def _post(score, conf, src="reddit", age_h=1, sid=None):
    return Post(
        source=src,
        source_id=sid or f"{src}:{score}:{conf}:{age_h}",
        author="x",
        text="t",
        url="u",
        created_at=datetime.now(timezone.utc) - timedelta(hours=age_h),
        score=score,
        confidence=conf,
    )


def test_aggregate_weighted_score_uses_confidence():
    posts = [_post(1.0, 0.9), _post(-1.0, 0.1)]
    snap = aggregate("X", posts)
    # High-confidence bullish dominates
    assert snap.sentiment_score > 0.5


def test_aggregate_velocity_compares_recent_to_earlier():
    posts = [
        _post(1.0, 0.9, age_h=1),
        _post(-1.0, 0.9, age_h=30),
    ]
    snap = aggregate("X", posts)
    assert snap.sentiment_velocity > 0


def test_aggregate_handles_empty():
    snap = aggregate("X", [])
    assert snap.sentiment_score == 0.0
    assert snap.mention_count_24h == 0


def test_aggregate_by_source_breakdown():
    posts = [
        _post(0.8, 0.9, src="reddit"),
        _post(0.2, 0.5, src="stocktwits"),
    ]
    snap = aggregate("X", posts)
    assert "reddit" in snap.by_source
    assert "stocktwits" in snap.by_source
    assert snap.by_source["reddit"]["count"] == 1


# --- Snapshot persistence ---

def test_write_and_load_snapshot_round_trip(tmp_path: Path):
    path = tmp_path / "sentiment.json"
    snap = aggregate("USDJPY=X", [_post(0.5, 0.8)])
    write_snapshot(path, {"USDJPY=X": snap})
    loaded = load_snapshot(path)
    assert "USDJPY=X" in loaded
    assert loaded["USDJPY=X"].sentiment_score == snap.sentiment_score


def test_load_snapshot_handles_missing_file(tmp_path: Path):
    assert load_snapshot(tmp_path / "nope.json") == {}


def test_snapshot_age_seconds(tmp_path: Path):
    path = tmp_path / "sentiment.json"
    write_snapshot(path, {})
    age = snapshot_age_seconds(path)
    assert age is not None and age >= 0


# --- Refresh end-to-end ---


class _FakeCollector(Source):
    def __init__(self, posts):
        self._posts = posts
    def fetch(self, query, limit=50):
        return self._posts


class _FakeAnthropic:
    """Minimal stand-in for anthropic.Anthropic that returns scored 0.5 for everything."""
    def __init__(self):
        self.messages = self
        self.calls = 0
    def create(self, **kw):
        self.calls += 1
        # Inspect the user prompt to extract source_ids and emit scores.
        user = kw["messages"][0]["content"]
        try:
            payload_start = user.index("```json") + len("```json")
            payload_end = user.index("```", payload_start)
            payload = json.loads(user[payload_start:payload_end].strip())
        except Exception:
            payload = []
        scores = [
            {"source_id": p["source_id"], "score": 0.5, "confidence": 0.7, "label": "bullish"}
            for p in payload
        ]
        text = json.dumps({"scores": scores})
        # Mimic anthropic SDK content blocks
        class _Block:
            type = "text"
            def __init__(self, t): self.text = t
        class _Resp:
            content = [_Block(text)]
        return _Resp()


def test_refresh_writes_snapshot_with_injected_collectors(tmp_path: Path):
    fake_post = Post(
        source="reddit",
        source_id="reddit:test1",
        author="x",
        text="USDJPY looks bullish",
        url="u",
        created_at=datetime.now(timezone.utc),
        language="en",
    )
    factories = {
        "reddit": lambda: _FakeCollector([fake_post]),
        "stocktwits": lambda: _FakeCollector([]),
        "tradingview": lambda: _FakeCollector([]),
        "twitter": lambda: _FakeCollector([]),
        "rss": lambda: _FakeCollector([]),
    }
    out = tmp_path / "sentiment.json"
    result = refresh(
        symbols=["USDJPY=X"],
        output_path=out,
        anthropic_client=_FakeAnthropic(),
        collector_factories=factories,
    )
    assert "USDJPY=X" in result.snapshots
    assert out.exists()
    snap = result.snapshots["USDJPY=X"]
    assert snap.mention_count_24h == 1
    # Score should be ~0.5 * 0.7 / 0.7 = 0.5
    assert snap.sentiment_score == pytest.approx(0.5)


def test_fx_sentiment_reader_reads_what_we_wrote(tmp_path: Path):
    from src.fx.sentiment import read_for

    snap = aggregate("USDJPY=X", [_post(0.5, 0.8)])
    path = tmp_path / "sentiment.json"
    write_snapshot(path, {"USDJPY=X": snap})
    view = read_for("USDJPY=X", path=path)
    assert view is not None
    assert view["sentiment_score"] == snap.sentiment_score


def test_fx_sentiment_reader_returns_none_for_stale(tmp_path: Path):
    from src.fx.sentiment import read_for

    snap = aggregate("USDJPY=X", [_post(0.5, 0.8)])
    path = tmp_path / "sentiment.json"
    write_snapshot(path, {"USDJPY=X": snap})
    # Force the on-disk timestamp to be 2 days old.
    data = json.loads(path.read_text())
    data["as_of"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    path.write_text(json.dumps(data))
    assert read_for("USDJPY=X", path=path, max_age_s=3600) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
