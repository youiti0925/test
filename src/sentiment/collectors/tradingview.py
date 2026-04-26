"""TradingView ideas collector.

TradingView has no public API. We respectfully scrape the public
ideas page, which renders public posts with author, title, summary,
LONG/SHORT bias, and timestamp.

We use a real User-Agent and request only the listing page (no logins,
no rate-limit dodging). The HTML structure changes occasionally; we
keep the parser permissive so a layout change degrades to "0 posts"
rather than crashing.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Callable

import requests

from ..base import Post, Source
from ..config import profile_for

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
IDEAS_URL = "https://www.tradingview.com/symbols/{symbol}/ideas/"


# TradingView embeds an `__INITIAL_STATE__` JSON blob inside a <script>.
_INITIAL_STATE_RE = re.compile(
    r"window\.initialState\s*=\s*(\{.+?\})\s*;\s*</script>",
    re.DOTALL,
)


class TradingViewCollector(Source):
    name = "tradingview"

    def __init__(
        self,
        timeout_s: float = 10.0,
        http_get: Callable | None = None,
    ) -> None:
        self.timeout_s = timeout_s
        self._get = http_get or requests.get

    def fetch(self, query: str, limit: int = 20) -> list[Post]:
        profile = profile_for(query)
        if profile is None:
            return []
        tv_symbol = profile.get("tradingview_symbol")
        if not tv_symbol:
            return []

        try:
            resp = self._get(
                IDEAS_URL.format(symbol=tv_symbol),
                headers={"User-Agent": UA, "Accept": "text/html"},
                timeout=self.timeout_s,
            )
            if resp.status_code != 200:
                return []
            html = resp.text
        except Exception:  # noqa: BLE001
            return []

        # Strategy 1: structured initialState JSON (preferred)
        posts = _parse_initial_state(html, query, tv_symbol, limit)
        if posts:
            return posts

        # Strategy 2: fallback HTML parse for cards.
        return _parse_html_cards(html, query, tv_symbol, limit)


def _parse_initial_state(
    html: str, query: str, tv_symbol: str, limit: int
) -> list[Post]:
    m = _INITIAL_STATE_RE.search(html)
    if not m:
        return []
    try:
        state = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []

    candidates: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("title") and (node.get("description") or node.get("text")) \
               and node.get("created"):
                candidates.append(node)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(state)

    posts: list[Post] = []
    for c in candidates[:limit]:
        text = (c.get("description") or c.get("text") or "")[:1500]
        title = c.get("title") or ""
        body = (title + ("\n\n" + text if text else "")).strip()
        if not body:
            continue
        try:
            created = datetime.fromtimestamp(int(c["created"]), tz=timezone.utc)
        except (KeyError, ValueError, TypeError):
            continue
        side = c.get("side") or c.get("ideaSide")  # 1 = long, -1 = short
        bias = "long" if side == 1 else "short" if side == -1 else None
        author = ((c.get("user") or {}).get("username")) or "?"
        slug = c.get("url") or c.get("idea_url") or ""
        posts.append(
            Post(
                source="tradingview",
                source_id=f"tradingview:{c.get('id') or hash(body)}",
                author=author,
                text=body,
                url=f"https://www.tradingview.com{slug}" if slug else IDEAS_URL.format(symbol=tv_symbol),
                created_at=created,
                language=c.get("lang", "en"),
                metadata={
                    "bias": bias,
                    "likes": c.get("likes_count") or c.get("likes"),
                },
            )
        )
    return posts


def _parse_html_cards(html: str, query: str, tv_symbol: str, limit: int) -> list[Post]:
    """Permissive fallback when no initialState JSON is found.

    Pulls article-like blocks, extracts the title and any visible long/short
    label. Datetimes are not always available so we use 'now' as a
    placeholder; aggregator filters by 24h window so this is harmless for
    older posts that won't ship a timestamp.
    """
    title_re = re.compile(
        r'<a[^>]+class="[^"]*tv-widget-idea__title[^"]*"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
    )
    bias_re = re.compile(r'tv-idea-label-(long|short)')

    posts: list[Post] = []
    matches = title_re.findall(html)
    bias_matches = bias_re.findall(html)
    for i, (href, title) in enumerate(matches[:limit]):
        bias = bias_matches[i] if i < len(bias_matches) else None
        posts.append(
            Post(
                source="tradingview",
                source_id=f"tradingview:html:{href}",
                author="?",
                text=title.strip(),
                url=f"https://www.tradingview.com{href}",
                created_at=datetime.now(timezone.utc),
                language="en",
                metadata={"bias": bias, "parser": "html_fallback"},
            )
        )
    return posts
