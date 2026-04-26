"""Thin reader that pulls the latest sentiment snapshot for a symbol.

This module is the integration point between the standalone sentiment
subsystem (`src/sentiment/`) and the FX analyst pipeline. The pipeline
calls `read_for(symbol)` which returns either a small dict suitable for
the LLM prompt, or None if no snapshot is available.

We intentionally do not run any HTTP, scoring, or LLM calls here — the
pipeline must stay fast. Refreshing the snapshot is an explicit job
(`fx sentiment refresh`).
"""
from __future__ import annotations

from pathlib import Path

from src.sentiment.snapshot import (
    SentimentSnapshot,
    load_snapshot,
    snapshot_age_seconds,
)

DEFAULT_PATH = Path("data/sentiment.json")


def read_for(
    symbol: str,
    path: Path = DEFAULT_PATH,
    max_age_s: int | None = None,
) -> dict | None:
    """Return the latest sentiment for `symbol` as a dict, or None.

    If `max_age_s` is set and the snapshot file is older, returns None
    (and a 'stale' flag would only mislead Claude).
    """
    if max_age_s is not None:
        age = snapshot_age_seconds(path)
        if age is None or age > max_age_s:
            return None

    snaps = load_snapshot(path)
    snap = snaps.get(symbol)
    if snap is None:
        return None
    return _to_prompt_view(snap)


def _to_prompt_view(snap: SentimentSnapshot) -> dict:
    """Trim the on-disk snapshot to what's worth showing to Claude.

    Notable posts are limited to text + score + source so the LLM
    can pick out concrete examples.
    """
    return {
        "as_of": snap.as_of.isoformat(),
        "mention_count_24h": snap.mention_count_24h,
        "sentiment_score": snap.sentiment_score,
        "sentiment_velocity": snap.sentiment_velocity,
        "by_source": snap.by_source,
        "notable_posts": [
            {
                "source": p.get("source"),
                "author": p.get("author"),
                "score": p.get("score"),
                "text": (p.get("text") or "")[:280],
            }
            for p in (snap.notable_posts or [])
        ],
    }
