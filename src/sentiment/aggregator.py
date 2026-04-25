"""Reduce a flat list of scored Post objects into per-symbol metrics.

Output is a SentimentSnapshot suitable for persistence and for feeding
into the FX analyst prompt.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from statistics import mean

from .base import Post
from .snapshot import SentimentSnapshot


def aggregate(
    symbol: str,
    posts: list[Post],
    *,
    now: datetime | None = None,
    velocity_window_h: int = 24,
) -> SentimentSnapshot:
    """Compute aggregate metrics for a single symbol from its posts."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)
    earlier_cutoff = now - timedelta(hours=24 + velocity_window_h)

    recent = [p for p in posts if p.created_at >= cutoff]
    earlier = [p for p in posts if earlier_cutoff <= p.created_at < cutoff]

    sources_count: dict[str, int] = Counter(p.source for p in recent)
    sources_score: dict[str, float] = {}
    for src in sources_count:
        src_posts = [p for p in recent if p.source == src]
        sources_score[src] = _weighted_mean_score(src_posts)

    score_now = _weighted_mean_score(recent)
    score_earlier = _weighted_mean_score(earlier)
    velocity = round(score_now - score_earlier, 4) if earlier else 0.0

    notable = sorted(
        (p for p in recent if (p.confidence or 0.0) >= 0.6),
        key=lambda p: abs(p.score or 0.0) * (p.confidence or 0.0),
        reverse=True,
    )[:5]

    return SentimentSnapshot(
        symbol=symbol,
        as_of=now,
        mention_count_24h=len(recent),
        sentiment_score=round(score_now, 4),
        sentiment_velocity=velocity,
        by_source={
            src: {
                "count": sources_count[src],
                "score": round(sources_score[src], 4),
            }
            for src in sources_count
        },
        notable_posts=[p.to_dict() for p in notable],
    )


def _weighted_mean_score(posts: list[Post]) -> float:
    weights = []
    values = []
    for p in posts:
        if p.score is None or p.confidence is None:
            continue
        if p.confidence <= 0:
            continue
        weights.append(p.confidence)
        values.append(p.score * p.confidence)
    if not weights:
        return 0.0
    return sum(values) / sum(weights)
