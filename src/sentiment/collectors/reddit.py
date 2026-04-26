"""Reddit collector — uses the public `.json` endpoint, no OAuth required.

Reddit allows unauthenticated reads if you send a meaningful User-Agent.
For higher quotas or NSFW content, swap in PRAW. We only need a small
sample of recent posts per symbol so the unauth path is fine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import requests

from ..base import Post, Source
from ..config import profile_for

REDDIT_UA = "fx-sentiment-bot/0.1 (educational research)"
SEARCH_URL = "https://www.reddit.com/r/{sub}/search.json"


class RedditCollector(Source):
    name = "reddit"

    def __init__(
        self,
        timeout_s: float = 8.0,
        http_get: Callable | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        # Inject `requests.get` so tests can swap in a fake.
        self._get = http_get or requests.get

    def fetch(self, query: str, limit: int = 50) -> list[Post]:
        profile = profile_for(query)
        if profile is None:
            return []
        subs = profile.get("reddit_subs") or []
        keywords = profile.get("reddit_keywords") or [query]
        if not subs:
            return []

        posts: list[Post] = []
        per_sub = max(5, limit // max(len(subs), 1))
        # Comma-OR-join keywords into one search query.
        q = " OR ".join(f'"{k}"' for k in keywords)
        for sub in subs:
            try:
                resp = self._get(
                    SEARCH_URL.format(sub=sub),
                    params={
                        "q": q,
                        "restrict_sr": "1",
                        "sort": "new",
                        "limit": per_sub,
                    },
                    headers={"User-Agent": REDDIT_UA},
                    timeout=self.timeout_s,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
            except Exception:  # noqa: BLE001
                continue
            for child in (data.get("data", {}).get("children") or []):
                d = child.get("data") or {}
                title = d.get("title") or ""
                body = d.get("selftext") or ""
                text = (title + ("\n\n" + body if body else "")).strip()
                if not text:
                    continue
                created = datetime.fromtimestamp(
                    float(d.get("created_utc") or 0), tz=timezone.utc
                )
                posts.append(
                    Post(
                        source=self.name,
                        source_id=f"reddit:{d.get('id')}",
                        author=d.get("author") or "?",
                        text=text[:1500],
                        url=f"https://reddit.com{d.get('permalink', '')}",
                        created_at=created,
                        language="en",
                        metadata={
                            "subreddit": sub,
                            "score": d.get("score"),
                            "num_comments": d.get("num_comments"),
                        },
                    )
                )
                if len(posts) >= limit:
                    return posts
        return posts
