"""End-to-end sentiment refresh: collect → translate → score → aggregate → write.

Used by `fx sentiment refresh` (CLI) and the dashboard's manual button.

Designed to be safe to call as a cron job: any per-source failure is
swallowed and reported in the result dict; the snapshot is still
written for whatever sources DID work.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from .aggregator import aggregate
from .base import Post
from .collectors import (
    RedditCollector,
    RSSCollector,
    StocktwitsCollector,
    TradingViewCollector,
    TwitterCollector,
)
from .scoring import score_posts, translate_to_english
from .snapshot import SentimentSnapshot, write_snapshot


@dataclass
class RefreshResult:
    snapshots: dict[str, SentimentSnapshot] = field(default_factory=dict)
    counts_by_source: dict[str, int] = field(default_factory=dict)
    errors_by_source: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0
    twitter_backend: str = "none"

    def to_dict(self) -> dict:
        return {
            "snapshots": {s: snap.to_dict() for s, snap in self.snapshots.items()},
            "counts_by_source": self.counts_by_source,
            "errors_by_source": self.errors_by_source,
            "elapsed_s": round(self.elapsed_s, 2),
            "twitter_backend": self.twitter_backend,
        }


def refresh(
    symbols: list[str],
    output_path: Path,
    *,
    enable: dict[str, bool] | None = None,
    score_with_llm: bool = True,
    translate: bool = False,
    anthropic_client: anthropic.Anthropic | None = None,
    collector_factories: dict | None = None,
) -> RefreshResult:
    """Run all enabled collectors for each symbol and persist the result.

    `enable` toggles individual sources by name. `collector_factories`
    is for tests — pass {"reddit": lambda: FakeReddit(), ...} to inject
    deterministic behavior.
    """
    enable = enable or {
        "reddit": True,
        "stocktwits": True,
        "tradingview": True,
        "twitter": True,
        "rss": True,
    }
    factories = collector_factories or {
        "reddit": RedditCollector,
        "stocktwits": StocktwitsCollector,
        "tradingview": TradingViewCollector,
        "twitter": TwitterCollector,
        "rss": RSSCollector,
    }

    started = time.perf_counter()
    result = RefreshResult()

    # Build collectors once; re-use across symbols.
    active = {n: factories[n]() for n in factories if enable.get(n)}

    all_posts_by_symbol: dict[str, list[Post]] = {sym: [] for sym in symbols}
    for source_name, collector in active.items():
        result.counts_by_source.setdefault(source_name, 0)
        for sym in symbols:
            try:
                posts = collector.fetch(sym, limit=50) or []
            except Exception as e:  # noqa: BLE001
                result.errors_by_source[source_name] = str(e)
                posts = []
            all_posts_by_symbol[sym].extend(posts)
            result.counts_by_source[source_name] += len(posts)

    if "twitter" in active:
        result.twitter_backend = getattr(active["twitter"], "backend_used", "none")

    if score_with_llm:
        client = anthropic_client or anthropic.Anthropic()
        for sym, posts in all_posts_by_symbol.items():
            if translate:
                posts = translate_to_english(posts, client=client)
            posts = score_posts(posts, client=client)
            all_posts_by_symbol[sym] = posts

    now = datetime.now(timezone.utc)
    for sym, posts in all_posts_by_symbol.items():
        result.snapshots[sym] = aggregate(sym, posts, now=now)

    write_snapshot(output_path, result.snapshots, as_of=now)
    result.elapsed_s = time.perf_counter() - started
    return result
