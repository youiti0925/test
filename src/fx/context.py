"""MarketContext — every input the Decision Engine and AI need.

Spec §8 lists the fields the AI snapshot must include. This dataclass
collects them all in one place and exposes a single `to_dict()` for
prompts, logs, and dashboard display.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .indicators import Snapshot
from .patterns import PatternResult


@dataclass(frozen=True)
class MarketContext:
    symbol: str
    interval: str
    snapshot: Snapshot                       # numeric technicals
    pattern: PatternResult                   # waveform / structure
    higher_timeframe_trend: str = "UNKNOWN"  # UPTREND | DOWNTREND | RANGE | VOLATILE | UNKNOWN
    spread_state: str = "UNKNOWN"            # NORMAL | WIDE | UNKNOWN
    spread_pct: Optional[float] = None
    event_risk_level: str = "LOW"            # LOW | MEDIUM | HIGH
    economic_events_nearby: list[dict] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    sentiment_velocity: Optional[float] = None
    sentiment_volume_spike: bool = False
    top_sentiment_keywords: list[str] = field(default_factory=list)
    risk_reward: Optional[float] = None
    atr_stop_distance: Optional[float] = None
    technical_signal: str = "HOLD"

    def to_dict(self) -> dict:
        snap = self.snapshot.to_dict()
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "last_close": snap.get("last_close"),
            **{k: v for k, v in snap.items() if k != "last_close" and k != "symbol"},
            "trend_state": self.pattern.trend_state.value,
            "higher_timeframe_trend": self.higher_timeframe_trend,
            "swing_structure": self.pattern.swing_structure,
            "market_structure": self.pattern.market_structure,
            "detected_pattern": self.pattern.detected_pattern,
            "pattern_confidence": round(self.pattern.pattern_confidence, 3),
            "neckline": (
                round(self.pattern.neckline, 6)
                if self.pattern.neckline is not None else None
            ),
            "neckline_broken": self.pattern.neckline_broken,
            "rsi_divergence": self.pattern.rsi_divergence,
            "macd_momentum_weakening": self.pattern.macd_momentum_weakening,
            "event_risk_level": self.event_risk_level,
            "economic_events_nearby": self.economic_events_nearby,
            "spread_state": self.spread_state,
            "spread_pct": self.spread_pct,
            "sentiment_score": self.sentiment_score,
            "sentiment_velocity": self.sentiment_velocity,
            "sentiment_volume_spike": self.sentiment_volume_spike,
            "top_sentiment_keywords": self.top_sentiment_keywords,
            "risk_reward": self.risk_reward,
            "atr_stop_distance": self.atr_stop_distance,
            "technical_signal": self.technical_signal,
        }


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def event_risk_level_from(events: list[dict]) -> str:
    """LOW / MEDIUM / HIGH summary used by the dashboard and AI prompt."""
    if not events:
        return "LOW"
    impacts = [(e.get("impact") or "medium").lower() for e in events]
    if any(i == "high" for i in impacts):
        return "HIGH"
    if any(i == "medium" for i in impacts):
        return "MEDIUM"
    return "LOW"


def detect_volume_spike(sentiment: dict | None, threshold: int = 200) -> bool:
    if not sentiment:
        return False
    return (sentiment.get("mention_count_24h") or 0) >= threshold


def keywords_from_sentiment(sentiment: dict | None, k: int = 5) -> list[str]:
    """Extract simple top keywords from notable_posts text. Best-effort."""
    if not sentiment:
        return []
    from collections import Counter
    import re

    words: list[str] = []
    for post in (sentiment.get("notable_posts") or [])[:30]:
        text = (post.get("text") or "").lower()
        for w in re.findall(r"[a-zA-Z一-鿿]{3,}", text):
            if w in {"the", "and", "for", "this", "that", "with", "from"}:
                continue
            words.append(w)
    return [w for w, _ in Counter(words).most_common(k)]
