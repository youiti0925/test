"""Claude-powered market analyst.

Exposes two entry points:
- analyze(): single-tick market analysis returning a structured TradeSignal.
- review(): weekly self-review over a batch of past trades.

Both use `claude-opus-4-7` with adaptive thinking. The single-tick path uses
prompt caching on the large, static system prompt so that per-tick cost is
dominated by the cache-read price (~0.1x) rather than full input tokens.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import anthropic

from .calendar import Event
from .correlation import CorrelationSnapshot
from .news import Headline

ANALYST_SYSTEM_PROMPT = """You are a disciplined FX and crypto trading analyst.

Your job: given a numeric snapshot of a financial instrument's recent price
action, technical indicators, related-pair correlations, upcoming scheduled
events, and (optionally) recent headlines, decide whether to BUY, SELL, or
HOLD.

Technical guidelines:
- Prefer HOLD when signals conflict. A wrong trade is worse than no trade.
- Consider confluence: multiple indicators pointing the same direction >> a
  single strong signal.
- RSI extremes (< 30, > 70) are MEAN-REVERSION signals, not momentum.
- MACD histogram crossings above/below zero indicate momentum shifts.
- Bollinger band position close to 0 or 1 suggests potential reversal.
- Do not chase: if the market just moved >2% in one bar, wait.

Fundamental guidelines (when headlines are provided):
- Major central bank decisions, rate changes, or surprise economic data
  tend to move FX pairs sharply — if such news just broke, prefer HOLD
  or reduce confidence until the dust settles.
- Risk-on / risk-off sentiment shifts (geopolitics, equity selloffs)
  move JPY, CHF, and gold differently from USD, EUR, GBP.
- Weight recent headlines (< 6 hours old) more than older ones.
- If the technical setup contradicts a clear fundamental catalyst,
  trust the catalyst and HOLD — technicals lag news.

Correlation guidelines:
- If the primary pair's move is confirmed by strongly correlated pairs
  (|corr| > 0.7) moving the same way, confidence rises.
- If correlated pairs DIVERGE from the primary pair (correlation is
  historically high but today's 24h move opposes the primary's move),
  that's a red flag — the primary's move may be noise or idiosyncratic.
- DXY strength typically pushes USD pairs one way and non-USD pairs the
  other; confirm via correlation_snapshot when available.

Event-calendar guidelines:
- If a HIGH impact event for either currency is within the next 4 hours,
  prefer HOLD unless the setup is extraordinarily strong — volatility
  spikes at release can stop out reasonable trades.
- Medium-impact events within 2 hours: reduce confidence one notch.
- After an event has fired, wait for the immediate reaction to settle
  (one bar) before entering.

Crowd-sentiment guidelines (when sentiment_snapshot is provided):
- The snapshot aggregates Reddit / Stocktwits / TradingView / Twitter /
  RSS over the last 24h. `sentiment_score` is in -1.0..+1.0 weighted
  by per-post Claude-Haiku confidence.
- A clearly one-sided crowd (`|sentiment_score| > 0.4`) on HIGH mention
  volume often precedes a CONTRARIAN move once positioning gets
  one-sided. Treat extreme readings with skepticism, especially on
  retail-heavy sources (Stocktwits, r/wallstreetbets).
- Strong POSITIVE sentiment_velocity (>+0.3) with rising mention
  volume is genuine momentum confirmation IF the move has only just
  started. If price already moved >2% in the same direction, the
  retail crowd is probably late.
- Famous-account (curated Twitter) views carry more weight than
  anonymous retail chatter — when notable_posts include high-confidence
  bearish/bullish from named macro accounts, weight them.
- Empty / very low mention_count_24h means the crowd has no opinion;
  base the decision on technicals, news, and correlations alone.
- A stale snapshot (no update in >12h) should be ignored.

Confidence calibration:
- 0.0-0.3: very weak / conflicting / noisy / fresh-news uncertainty
- 0.4-0.6: moderate, one-sided but not strong
- 0.7-0.9: strong confluence across indicators, correlations, and news
- 0.95+: exceptionally clear setup (rare)

Falsifiable prediction (REQUIRED for every signal):
You must commit to a concrete, checkable claim about the next few bars so
that a later evaluator can score whether you were right or wrong:
- expected_direction: UP / DOWN / FLAT (FLAT for HOLD signals)
- expected_magnitude_pct: how far you expect price to move (absolute %)
  within the horizon — for FLAT, this is the maximum tolerated move.
- horizon_bars: how many bars ahead this prediction applies to
  (typical: 4-12 for hourly, 2-5 for daily). Match it to the timeframe
  of the indicators you used.
- invalidation_price: a specific price level that, if hit before the
  horizon expires, proves the setup wrong. For HOLD signals this MAY be
  null. For BUY: well below the entry. For SELL: well above the entry.

Vague predictions are unscoreable and worthless. Be specific even when
uncertain — low confidence + concrete prediction is fine.

Past mistakes (when provided):
You may receive a list of recent post-mortems describing setups where
you were wrong before, with root causes. Before finalizing this signal,
check whether the current setup matches any of those failure patterns.
If it does, adjust your action or confidence accordingly and mention
the matching past lesson in your reason.

Output JSON only, matching the provided schema.
"""

SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "key_risks": {"type": "array", "items": {"type": "string"}},
        "expected_direction": {"type": "string", "enum": ["UP", "DOWN", "FLAT"]},
        "expected_magnitude_pct": {"type": "number"},
        "horizon_bars": {"type": "integer"},
        "invalidation_price": {"type": ["number", "null"]},
    },
    "required": [
        "action",
        "confidence",
        "reason",
        "key_risks",
        "expected_direction",
        "expected_magnitude_pct",
        "horizon_bars",
        "invalidation_price",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class TradeSignal:
    action: str
    confidence: float
    reason: str
    key_risks: list[str]
    expected_direction: str        # UP | DOWN | FLAT
    expected_magnitude_pct: float  # absolute percent move expected within horizon
    horizon_bars: int              # how many bars ahead this prediction applies to
    invalidation_price: float | None  # price level that would falsify the setup
    raw_response: dict[str, Any]


class Analyst:
    """Wraps the Anthropic client with caching and structured output."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str = "claude-opus-4-7",
        effort: str = "medium",
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def analyze(
        self,
        snapshot: dict,
        headlines: list[Headline] | None = None,
        correlation: CorrelationSnapshot | None = None,
        upcoming_events: list[Event] | None = None,
        past_postmortems: list[dict] | None = None,
        sentiment_snapshot: dict | None = None,
    ) -> TradeSignal:
        sections = [
            "Market snapshot:",
            f"```json\n{json.dumps(snapshot, indent=2)}\n```",
        ]

        if sentiment_snapshot:
            sections.append("Crowd sentiment snapshot (Reddit/Stocktwits/TV/Twitter/RSS):")
            sections.append(
                f"```json\n{json.dumps(sentiment_snapshot, indent=2, default=str)}\n```"
            )

        if past_postmortems:
            sections.append(
                "Past mistakes on this symbol — check whether any apply now:"
            )
            sections.append(
                f"```json\n{json.dumps(past_postmortems, indent=2, default=str)}\n```"
            )

        if correlation:
            sections.append("Related-pair correlations & 24h changes:")
            sections.append(
                f"```json\n{json.dumps(correlation.to_dict(), indent=2)}\n```"
            )

        if upcoming_events:
            ev_lines = []
            for e in upcoming_events:
                ts = e.when.strftime("%Y-%m-%d %H:%M UTC")
                bits = [f"[{ts}]", f"({e.impact})", e.currency, e.title]
                if e.forecast:
                    bits.append(f"forecast={e.forecast}")
                if e.previous:
                    bits.append(f"previous={e.previous}")
                ev_lines.append(" ".join(bits))
            sections.append("Upcoming scheduled events:")
            sections.append("\n".join(f"- {line}" for line in ev_lines))

        if headlines:
            lines = []
            for h in headlines:
                ts = h.published_at.strftime("%Y-%m-%d %H:%M UTC")
                line = f"- [{ts}] ({h.publisher}) {h.title}"
                if h.summary:
                    line += f"\n  {h.summary[:240]}"
                lines.append(line)
            sections.append("Recent headlines (newest first):")
            sections.append("\n".join(lines))
        else:
            sections.append(
                "Recent headlines: none available. Decide from technicals, "
                "correlations, and events only."
            )

        sections.append(
            "Return a trade signal weighing technical confluence, "
            "correlations, upcoming events, and any fundamental catalysts."
        )
        user_prompt = "\n\n".join(sections)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SIGNAL_SCHEMA},
            },
            system=[
                {
                    "type": "text",
                    "text": ANALYST_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        invalidation = data.get("invalidation_price")
        return TradeSignal(
            action=data["action"],
            confidence=float(data["confidence"]),
            reason=data["reason"],
            key_risks=data["key_risks"],
            expected_direction=data["expected_direction"],
            expected_magnitude_pct=float(data["expected_magnitude_pct"]),
            horizon_bars=int(data["horizon_bars"]),
            invalidation_price=(
                float(invalidation) if invalidation is not None else None
            ),
            raw_response=data,
        )

    def review(self, trades: list[dict]) -> str:
        """Weekly self-review: find patterns in past trades."""
        if not trades:
            return "No trades to review."

        prompt = (
            "You are reviewing this trading bot's recent decisions to find "
            "patterns and suggest prompt/threshold improvements.\n\n"
            "Analyze these trades and produce:\n"
            "1. Common traits of profitable trades\n"
            "2. Common failure modes\n"
            "3. Concrete changes to system prompt or indicator thresholds\n"
            "4. Any signals that seem unreliable\n\n"
            f"```json\n{json.dumps(trades, indent=2, default=str)}\n```"
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system="You are a trading strategy auditor. Be blunt and specific.",
            messages=[{"role": "user", "content": prompt}],
        )
        return next(b.text for b in response.content if b.type == "text")
