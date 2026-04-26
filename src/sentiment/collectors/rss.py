"""RSS / Atom collector — keyword-filtered per symbol.

Uses the stdlib's xml.etree so we don't add `feedparser` as a hard
dependency. Handles RSS 2.0 (<item>) and Atom (<entry>) feeds.

Each feed is fetched independently; any failure for one feed is silent
and just contributes 0 posts.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

import requests

from ..base import Post, Source
from ..config import RSS_FEEDS, profile_for

UA = "fx-sentiment-bot/0.1 (rss reader)"
TAG_RE = re.compile(r"<[^>]+>")


class RSSCollector(Source):
    name = "rss"

    def __init__(
        self,
        feeds: list[str] | None = None,
        timeout_s: float = 8.0,
        http_get: Callable | None = None,
    ) -> None:
        self.feeds = feeds or RSS_FEEDS
        self.timeout_s = timeout_s
        self._get = http_get or requests.get

    def fetch(self, query: str, limit: int = 30) -> list[Post]:
        profile = profile_for(query)
        if profile is None:
            return []
        keywords = [k.lower() for k in (profile.get("rss_keywords") or [query])]

        out: list[Post] = []
        for feed_url in self.feeds:
            try:
                resp = self._get(
                    feed_url,
                    headers={"User-Agent": UA, "Accept": "application/rss+xml,application/atom+xml,application/xml"},
                    timeout=self.timeout_s,
                )
                if resp.status_code != 200 or not resp.content:
                    continue
                items = _parse_feed(resp.content)
            except Exception:  # noqa: BLE001
                continue

            for it in items:
                text = (it["title"] + "\n" + it.get("summary", "")).strip()
                low = text.lower()
                if not any(k in low for k in keywords):
                    continue
                out.append(
                    Post(
                        source=self.name,
                        source_id=f"rss:{it['link']}",
                        author=it.get("author") or _domain(feed_url),
                        text=text[:1500],
                        url=it["link"],
                        created_at=it["created_at"],
                        language="en",
                        metadata={"feed": feed_url},
                    )
                )
                if len(out) >= limit:
                    return out
        return out


def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1) if m else "rss"


def _parse_feed(content: bytes) -> list[dict]:
    """Return a list of {title, summary, link, author, created_at}."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    items: list[dict] = []

    # RSS 2.0
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        desc = TAG_RE.sub("", (item.findtext("description") or "").strip())
        date_text = (
            item.findtext("pubDate")
            or item.findtext("{http://purl.org/dc/elements/1.1/}date")
            or ""
        )
        author = item.findtext("{http://purl.org/dc/elements/1.1/}creator") or item.findtext("author")
        items.append(
            {
                "title": title,
                "link": link,
                "summary": desc,
                "author": author,
                "created_at": _parse_date(date_text),
            }
        )

    # Atom
    atom_ns = "{http://www.w3.org/2005/Atom}"
    for entry in root.iter(f"{atom_ns}entry"):
        title = (entry.findtext(f"{atom_ns}title") or "").strip()
        link_el = entry.find(f"{atom_ns}link")
        link = link_el.attrib.get("href", "") if link_el is not None else ""
        summary = TAG_RE.sub("", (entry.findtext(f"{atom_ns}summary") or "").strip())
        date_text = entry.findtext(f"{atom_ns}updated") or entry.findtext(f"{atom_ns}published") or ""
        author = entry.findtext(f"{atom_ns}author/{atom_ns}name")
        items.append(
            {
                "title": title,
                "link": link,
                "summary": summary,
                "author": author,
                "created_at": _parse_date(date_text),
            }
        )

    return [i for i in items if i["title"] and i["link"]]


def _parse_date(s: str) -> datetime:
    if not s:
        return datetime.now(timezone.utc)
    try:
        return parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc)
