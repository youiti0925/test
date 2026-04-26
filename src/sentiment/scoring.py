"""Sentiment scoring with Claude Haiku.

Each post gets translated to English (if needed) and then scored. We
batch up to N posts per API call to save tokens. Each batch returns a
strict JSON array of {source_id, score, confidence, label}.

Why Haiku: 4.5 is fast and cheap (~$1/$5 per 1M tokens). For 200 posts
of ~150 tokens each, the input is ~30K tokens — pennies per refresh.

Why structured outputs: forcing JSON shape eliminates parsing errors
and lets us use `is_error` paths cleanly.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Iterable

import anthropic

from .base import Post

SCORING_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "score": {"type": "number"},      # -1.0 .. +1.0
                    "confidence": {"type": "number"},  # 0.0 .. 1.0
                    "label": {"type": "string", "enum": ["bullish", "bearish", "neutral"]},
                },
                "required": ["source_id", "score", "confidence", "label"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


SCORING_SYSTEM_PROMPT = """You are a sentiment scorer for financial market posts.

For each post you receive, decide whether the AUTHOR is expressing a
bullish, bearish, or neutral view on the underlying instrument.

Scoring:
  +1.0  strongly bullish (clear conviction price will rise)
  +0.5  mildly bullish
   0.0  neutral / informational / off-topic
  -0.5  mildly bearish
  -1.0  strongly bearish

Confidence reflects HOW SURE you are of the score, not how sure the
author is. Lower confidence for sarcasm, mixed messages, or unclear
context. If a post is in a non-English language and you cannot follow
it, score 0.0 with confidence 0.1.

Stay strict: noise, off-topic chatter, ads, and "thoughts?" questions
without a stance should all be neutral with low confidence.

Output JSON matching the schema. Include EVERY input post by source_id.
"""


def score_posts(
    posts: list[Post],
    client: anthropic.Anthropic | None = None,
    model: str = "claude-haiku-4-5",
    batch_size: int = 25,
) -> list[Post]:
    """Return new Post objects with .score and .confidence populated.

    Posts that fail to score are returned with score=0.0 confidence=0.0
    so the aggregator still sees them but doesn't weight them.
    """
    if not posts:
        return []
    client = client or anthropic.Anthropic()

    scored: dict[str, tuple[float, float]] = {}
    for i in range(0, len(posts), batch_size):
        batch = posts[i : i + batch_size]
        try:
            scored.update(_score_batch(client, model, batch))
        except Exception:  # noqa: BLE001
            # Whole batch failed — leave those posts unscored, continue.
            continue

    out: list[Post] = []
    for p in posts:
        s, c = scored.get(p.source_id, (0.0, 0.0))
        out.append(replace(p, score=s, confidence=c))
    return out


def _score_batch(
    client: anthropic.Anthropic,
    model: str,
    batch: list[Post],
) -> dict[str, tuple[float, float]]:
    payload = [
        {
            "source_id": p.source_id,
            "source": p.source,
            "language": p.language,
            "metadata": {
                k: v for k, v in p.metadata.items()
                if k in ("user_sentiment", "bias")
            },
            "text": p.text[:800],
        }
        for p in batch
    ]
    user_prompt = (
        "Score these posts. Reply with one entry per source_id.\n\n"
        f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    )
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": SCORING_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        output_config={"format": {"type": "json_schema", "schema": SCORING_SCHEMA}},
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    out: dict[str, tuple[float, float]] = {}
    for entry in data.get("scores", []):
        sid = entry.get("source_id")
        if sid is None:
            continue
        out[sid] = (float(entry.get("score", 0.0)), float(entry.get("confidence", 0.0)))
    return out


# --- Optional translation helper ---

def translate_to_english(
    posts: Iterable[Post],
    client: anthropic.Anthropic | None = None,
    model: str = "claude-haiku-4-5",
) -> list[Post]:
    """Translate non-English posts in-place. Cheap when no posts need it."""
    posts = list(posts)
    targets = [p for p in posts if (p.language or "en").lower().split("-")[0] != "en"]
    if not targets:
        return posts
    client = client or anthropic.Anthropic()

    chunked = [targets[i : i + 30] for i in range(0, len(targets), 30)]
    translations: dict[str, str] = {}
    for batch in chunked:
        try:
            translations.update(_translate_batch(client, model, batch))
        except Exception:  # noqa: BLE001
            continue

    out: list[Post] = []
    for p in posts:
        if p.source_id in translations:
            out.append(replace(p, text=translations[p.source_id], language="en"))
        else:
            out.append(p)
    return out


_TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "text_en": {"type": "string"},
                },
                "required": ["source_id", "text_en"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


def _translate_batch(client, model, batch):
    payload = [
        {"source_id": p.source_id, "language": p.language, "text": p.text[:800]}
        for p in batch
    ]
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system="Translate each post's text into concise English. Preserve "
               "tone and any cashtags / tickers verbatim. Reply with JSON.",
        output_config={
            "format": {"type": "json_schema", "schema": _TRANSLATE_SCHEMA}
        },
        messages=[
            {
                "role": "user",
                "content": f"```json\n{json.dumps(payload, indent=2, default=str)}\n```",
            }
        ],
    )
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return {
        e["source_id"]: e["text_en"]
        for e in data.get("translations", [])
        if e.get("source_id") and e.get("text_en")
    }
