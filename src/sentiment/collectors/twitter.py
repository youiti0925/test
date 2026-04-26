"""Twitter / X collector — curated influencers only.

Twitter actively blocks scrapers, so this collector tries multiple
backends in order and accepts whatever works:

  1. snscrape         — Python lib, sometimes works without auth.
  2. Nitter HTML      — public Nitter instances (configurable, unstable).
  3. Twitter API v2   — only if `TWITTER_BEARER_TOKEN` env var is set.

If all three fail, returns []. The aggregator will simply note that
Twitter contributed 0 posts; nothing else breaks.

We deliberately limit ourselves to a small curated influencer list
(see config.TWITTER_INFLUENCERS) so the per-symbol fetch is cheap
and respectful regardless of which backend ends up serving us.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import requests

from ..base import Post, Source
from ..config import influencers_for

NITTER_INSTANCES = (
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.cz",
)

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class TwitterCollector(Source):
    name: str = "twitter"
    timeout_s: float = 8.0
    per_user_limit: int = 5
    max_age: timedelta = timedelta(days=2)
    _http_get: Callable | None = None
    _backend_used: str = "none"

    def __post_init__(self) -> None:
        self._get = self._http_get or requests.get

    @property
    def backend_used(self) -> str:
        return self._backend_used

    def fetch(self, query: str, limit: int = 50) -> list[Post]:
        accounts = influencers_for(query)
        if not accounts:
            return []

        cutoff = datetime.now(timezone.utc) - self.max_age

        # Backend 1: snscrape
        posts = self._try_snscrape(accounts, limit, cutoff)
        if posts:
            self._backend_used = "snscrape"
            return posts

        # Backend 2: Nitter
        posts = self._try_nitter(accounts, limit, cutoff)
        if posts:
            self._backend_used = "nitter"
            return posts

        # Backend 3: Twitter API v2 (only if token set)
        if os.environ.get("TWITTER_BEARER_TOKEN"):
            posts = self._try_api(accounts, limit, cutoff)
            if posts:
                self._backend_used = "api_v2"
                return posts

        self._backend_used = "none"
        return []

    # --- Backends ---

    def _try_snscrape(
        self, accounts: list[str], limit: int, cutoff: datetime
    ) -> list[Post]:
        try:
            import snscrape.modules.twitter as sntwitter  # type: ignore
        except Exception:  # noqa: BLE001
            return []

        out: list[Post] = []
        for user in accounts:
            try:
                scraper = sntwitter.TwitterUserScraper(user)
                for i, t in enumerate(scraper.get_items()):
                    if i >= self.per_user_limit:
                        break
                    if t.date < cutoff:
                        break
                    out.append(_post_from_snscrape(t))
                    if len(out) >= limit:
                        return out
            except Exception:  # noqa: BLE001
                # Twitter often blocks; treat per-user failure as soft and continue.
                continue
        return out

    def _try_nitter(
        self, accounts: list[str], limit: int, cutoff: datetime
    ) -> list[Post]:
        out: list[Post] = []
        for user in accounts:
            html = _try_nitter_user(user, self._get, self.timeout_s)
            if not html:
                continue
            for post in _parse_nitter(html, user, cutoff)[: self.per_user_limit]:
                out.append(post)
                if len(out) >= limit:
                    return out
        return out

    def _try_api(
        self, accounts: list[str], limit: int, cutoff: datetime
    ) -> list[Post]:
        token = os.environ["TWITTER_BEARER_TOKEN"]
        out: list[Post] = []
        for user in accounts:
            try:
                u_resp = self._get(
                    f"https://api.twitter.com/2/users/by/username/{user}",
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=self.timeout_s,
                )
                if u_resp.status_code != 200:
                    continue
                user_id = (u_resp.json().get("data") or {}).get("id")
                if not user_id:
                    continue
                t_resp = self._get(
                    f"https://api.twitter.com/2/users/{user_id}/tweets",
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        "max_results": self.per_user_limit,
                        "tweet.fields": "created_at,public_metrics,lang",
                    },
                    timeout=self.timeout_s,
                )
                if t_resp.status_code != 200:
                    continue
                for tw in (t_resp.json().get("data") or [])[: self.per_user_limit]:
                    try:
                        created = datetime.fromisoformat(
                            tw["created_at"].replace("Z", "+00:00")
                        )
                    except (KeyError, ValueError):
                        continue
                    if created < cutoff:
                        continue
                    out.append(
                        Post(
                            source="twitter",
                            source_id=f"twitter:{tw.get('id')}",
                            author=user,
                            text=(tw.get("text") or "")[:1500],
                            url=f"https://twitter.com/{user}/status/{tw.get('id')}",
                            created_at=created,
                            language=tw.get("lang") or "en",
                            metadata={
                                "metrics": tw.get("public_metrics"),
                                "backend": "api_v2",
                            },
                        )
                    )
                    if len(out) >= limit:
                        return out
            except Exception:  # noqa: BLE001
                continue
        return out


def _post_from_snscrape(t) -> Post:
    return Post(
        source="twitter",
        source_id=f"twitter:{t.id}",
        author=t.user.username,
        text=(t.rawContent or "")[:1500],
        url=t.url,
        created_at=t.date,
        language=getattr(t, "lang", "en") or "en",
        metadata={
            "likes": t.likeCount,
            "retweets": t.retweetCount,
            "backend": "snscrape",
        },
    )


_NITTER_TWEET_RE = re.compile(
    r'<div class="tweet-content[^"]*">(.+?)</div>', re.DOTALL
)
_NITTER_DATE_RE = re.compile(r'title="([^"]+)" class="tweet-date"')
_NITTER_LINK_RE = re.compile(r'<a class="tweet-link" href="([^"]+)"')


def _try_nitter(user: str, http_get, timeout):  # kept for type annotation reasons
    return _try_nitter_user(user, http_get, timeout)


def _try_nitter_user(user: str, http_get, timeout) -> str | None:
    for instance in NITTER_INSTANCES:
        try:
            resp = http_get(
                f"{instance}/{user}",
                headers={"User-Agent": UA},
                timeout=timeout,
            )
            if resp.status_code == 200 and "tweet-content" in resp.text:
                return resp.text
        except Exception:  # noqa: BLE001
            continue
    return None


def _parse_nitter(html: str, user: str, cutoff: datetime) -> list[Post]:
    contents = _NITTER_TWEET_RE.findall(html)
    dates = _NITTER_DATE_RE.findall(html)
    links = _NITTER_LINK_RE.findall(html)
    out: list[Post] = []
    for i, raw in enumerate(contents):
        text = re.sub(r"<[^>]+>", "", raw).strip()
        if not text:
            continue
        date_str = dates[i] if i < len(dates) else None
        url_path = links[i] if i < len(links) else None
        try:
            created = (
                datetime.strptime(date_str, "%b %d, %Y · %I:%M %p %Z")
                .replace(tzinfo=timezone.utc)
                if date_str
                else datetime.now(timezone.utc)
            )
        except ValueError:
            created = datetime.now(timezone.utc)
        if created < cutoff:
            continue
        out.append(
            Post(
                source="twitter",
                source_id=f"twitter:nitter:{user}:{i}",
                author=user,
                text=text[:1500],
                url=f"https://twitter.com{url_path or ''}",
                created_at=created,
                language="en",
                metadata={"backend": "nitter"},
            )
        )
    return out
