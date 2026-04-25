"""Shared types for the sentiment subsystem.

Every collector returns a list of Post objects so downstream scoring,
translation, and aggregation never has to know which source the text
came from.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Post:
    source: str           # "reddit" | "stocktwits" | "tradingview" | "twitter" | "rss"
    source_id: str        # unique within source
    author: str
    text: str
    url: str
    created_at: datetime
    language: str = "en"  # ISO 639-1; "en" by default
    metadata: dict[str, Any] = field(default_factory=dict)
    # filled in by scoring / aggregator pipeline
    score: float | None = None        # -1.0 (bearish) .. +1.0 (bullish)
    confidence: float | None = None   # 0.0 .. 1.0

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "source_id": self.source_id,
            "author": self.author,
            "text": self.text,
            "url": self.url,
            "created_at": self.created_at.isoformat(),
            "language": self.language,
            "metadata": self.metadata,
            "score": self.score,
            "confidence": self.confidence,
        }


class Source(ABC):
    """Base class for a data collector.

    Implementations should be I/O-bounded, return at most `limit` posts
    per call, and degrade silently — return an empty list on any
    transient failure rather than raising.
    """

    name: str

    @abstractmethod
    def fetch(self, query: str, limit: int = 50) -> list[Post]:
        """Return recent posts matching `query` (typically a symbol)."""


class CollectorError(RuntimeError):
    """Raised by collectors when a configuration mistake (not a transient
    failure) prevents them from running. Transient HTTP / parse errors
    should NOT raise — they should return [].
    """
