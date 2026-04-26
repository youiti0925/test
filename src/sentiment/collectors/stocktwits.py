"""Stocktwits collector — public REST API, no auth required.

Each post may carry a user-tagged `sentiment` ("Bullish" | "Bearish")
which we surface in metadata; the LLM scorer can use it as a prior.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

import requests

from ..base import Post, Source
from ..config import profile_for

STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


class StocktwitsCollector(Source):
    name = "stocktwits"

    def __init__(
        self,
        timeout_s: float = 8.0,
        http_get: Callable | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self._get = http_get or requests.get

    def fetch(self, query: str, limit: int = 30) -> list[Post]:
        profile = profile_for(query)
        if profile is None:
            return []
        symbol = profile.get("stocktwits_symbol")
        if not symbol:
            return []

        try:
            resp = self._get(
                STREAM_URL.format(symbol=symbol),
                params={"limit": limit},
                timeout=self.timeout_s,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
        except Exception:  # noqa: BLE001
            return []

        posts: list[Post] = []
        for msg in (data.get("messages") or [])[:limit]:
            body = (msg.get("body") or "").strip()
            if not body:
                continue
            try:
                created = datetime.fromisoformat(
                    msg["created_at"].replace("Z", "+00:00")
                )
            except (KeyError, ValueError):
                continue
            user = (msg.get("user") or {}).get("username") or "?"
            entities = msg.get("entities") or {}
            sentiment = (entities.get("sentiment") or {}).get("basic")
            posts.append(
                Post(
                    source=self.name,
                    source_id=f"stocktwits:{msg.get('id')}",
                    author=user,
                    text=body[:1500],
                    url=f"https://stocktwits.com/{user}/message/{msg.get('id')}",
                    created_at=created,
                    language="en",
                    metadata={
                        "user_followers": (msg.get("user") or {}).get("followers"),
                        "user_sentiment": sentiment,  # "Bullish" | "Bearish" | None
                    },
                )
            )
        return posts
