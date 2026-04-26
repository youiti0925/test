"""Persist sentiment snapshots as JSON.

On-disk shape:
{
  "as_of": "2025-04-25T10:00:00+00:00",
  "symbols": {
    "USDJPY=X": { ...SentimentSnapshot... },
    ...
  }
}

Single-file JSON (rather than DB rows) because:
  * sentiment is small (<100 KB even with notable posts)
  * the FX pipeline only needs the latest snapshot
  * easy to inspect with `cat data/sentiment.json`
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SentimentSnapshot:
    symbol: str
    as_of: datetime
    mention_count_24h: int
    sentiment_score: float
    sentiment_velocity: float
    by_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    notable_posts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = self.as_of.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SentimentSnapshot":
        return cls(
            symbol=d["symbol"],
            as_of=_parse_dt(d["as_of"]),
            mention_count_24h=int(d.get("mention_count_24h", 0)),
            sentiment_score=float(d.get("sentiment_score", 0.0)),
            sentiment_velocity=float(d.get("sentiment_velocity", 0.0)),
            by_source=dict(d.get("by_source") or {}),
            notable_posts=list(d.get("notable_posts") or []),
        )


def _parse_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def write_snapshot(
    path: Path,
    snapshots: dict[str, SentimentSnapshot],
    as_of: datetime | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "as_of": (as_of or datetime.now(timezone.utc)).isoformat(),
        "symbols": {sym: snap.to_dict() for sym, snap in snapshots.items()},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_snapshot(path: Path) -> dict[str, SentimentSnapshot]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, SentimentSnapshot] = {}
    for sym, d in (data.get("symbols") or {}).items():
        try:
            out[sym] = SentimentSnapshot.from_dict({**d, "symbol": sym})
        except (KeyError, ValueError, TypeError):
            continue
    return out


def snapshot_age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    as_of = data.get("as_of")
    if not as_of:
        return None
    try:
        ts = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - ts).total_seconds()
