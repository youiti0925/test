"""Sentiment collection and aggregation.

Independent module from `src.fx`. Outputs a JSON snapshot that the FX
analyst pipeline reads on demand. See `src/sentiment/snapshot.py` for
the on-disk schema.
"""
from .base import Post, Source
from .snapshot import SentimentSnapshot, load_snapshot, write_snapshot

__all__ = [
    "Post",
    "Source",
    "SentimentSnapshot",
    "load_snapshot",
    "write_snapshot",
]
