"""External-signal comparator (spec §9).

What this is for
----------------
You ran the engine on a date. A vendor (QuantConnect / TrendSpider /
EconPulse / StockGeist) also produced a signal for the same instrument
on the same date. This module pairs them up by timestamp + symbol and
quantifies agreement, then aggregates a report you can use to ask:

  "Where am I systematically wrong vs. vendor X?"
  "Where do I systematically catch what vendor X misses?"

Critical spec rule (§9 / §17)
-----------------------------
Do NOT blindly inherit a vendor's verdict. The comparison only matters
when a downstream truth (price action) decides who was right. This
module surfaces the disagreement; verifying it against realised returns
is the user's responsibility (and a future feature).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class ExternalSignal:
    """A single decision from an external service.

    Mostly-optional fields — different vendors expose different things.
    Anything missing stays None and is reported as such in the diff.
    """

    source: str
    symbol: str
    timeframe: str
    timestamp: datetime
    action: str | None = None         # BUY | SELL | HOLD | None
    confidence: float | None = None
    pattern: str | None = None
    event_risk: str | None = None
    sentiment_score: float | None = None
    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    raw_payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "confidence": self.confidence,
            "pattern": self.pattern,
            "event_risk": self.event_risk,
            "sentiment_score": self.sentiment_score,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "raw_payload": dict(self.raw_payload),
        }


@dataclass(frozen=True)
class ComparisonResult:
    self_decision: dict           # serialised Decision.to_dict()
    external_signal: ExternalSignal
    action_match: bool
    pattern_match: bool | None
    event_risk_match: bool | None
    sentiment_gap: float | None
    entry_price_gap: float | None
    stop_loss_gap: float | None
    take_profit_gap: float | None
    agreement_score: float        # 0..1
    disagreement_reason: str | None

    def to_dict(self) -> dict:
        return {
            "self_decision": dict(self.self_decision),
            "external_signal": self.external_signal.to_dict(),
            "action_match": self.action_match,
            "pattern_match": self.pattern_match,
            "event_risk_match": self.event_risk_match,
            "sentiment_gap": self.sentiment_gap,
            "entry_price_gap": self.entry_price_gap,
            "stop_loss_gap": self.stop_loss_gap,
            "take_profit_gap": self.take_profit_gap,
            "agreement_score": round(self.agreement_score, 4),
            "disagreement_reason": self.disagreement_reason,
        }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _gap(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a - b)


def _ratio_gap(a: float | None, b: float | None) -> float | None:
    """Percent gap relative to magnitude — useful for prices."""
    if a is None or b is None or a == 0:
        return None
    return float((a - b) / abs(a))


def compare_one(
    self_decision: dict,
    external: ExternalSignal,
) -> ComparisonResult:
    """Compare a single self-decision dict to one external signal.

    `self_decision` is the dict produced by `Decision.to_dict()`. We
    don't take the typed Decision so callers can compare from logs or
    from the predictions table without reconstructing.
    """
    self_action = (self_decision.get("action") or "HOLD").upper()
    ext_action = (external.action or "HOLD").upper()
    action_match = self_action == ext_action

    self_pattern = (self_decision.get("advisory") or {}).get("pattern")
    pattern_match = (
        None if external.pattern is None or self_pattern is None
        else self_pattern.upper() == external.pattern.upper()
    )

    self_event = (
        (self_decision.get("advisory") or {}).get("event_risk_level")
        or self_decision.get("event_risk_level")
    )
    event_match = (
        None if external.event_risk is None or self_event is None
        else self_event.upper() == external.event_risk.upper()
    )

    self_sent = (
        (self_decision.get("advisory") or {}).get("sentiment_score")
        or self_decision.get("sentiment_score")
    )
    sentiment_gap = _gap(self_sent, external.sentiment_score)

    self_entry = self_decision.get("entry_price")
    entry_gap = _ratio_gap(self_entry, external.entry_price)
    stop_gap = _ratio_gap(
        self_decision.get("stop_loss"), external.stop_loss,
    )
    tp_gap = _ratio_gap(
        self_decision.get("take_profit"), external.take_profit,
    )

    # Agreement score: action match dominates; sub-fields each weight 1/4.
    parts: list[float] = [1.0 if action_match else 0.0]
    if pattern_match is not None:
        parts.append(1.0 if pattern_match else 0.0)
    if event_match is not None:
        parts.append(1.0 if event_match else 0.0)
    if sentiment_gap is not None:
        # Sentiment is in [-1, 1]; gap of 0 → 1.0, gap of 2 → 0.0
        parts.append(max(0.0, 1.0 - abs(sentiment_gap) / 2.0))
    score = sum(parts) / len(parts)

    reason: str | None = None
    if not action_match:
        reason = f"actions differ: self={self_action} vs {external.source}={ext_action}"
    elif sentiment_gap is not None and abs(sentiment_gap) > 0.6:
        reason = f"sentiment gap |{sentiment_gap:.2f}| > 0.6"

    return ComparisonResult(
        self_decision=dict(self_decision),
        external_signal=external,
        action_match=action_match,
        pattern_match=pattern_match,
        event_risk_match=event_match,
        sentiment_gap=sentiment_gap,
        entry_price_gap=entry_gap,
        stop_loss_gap=stop_gap,
        take_profit_gap=tp_gap,
        agreement_score=float(score),
        disagreement_reason=reason,
    )


def compare_signals(
    self_decisions: Iterable[dict],
    external_signals: Iterable[ExternalSignal],
    *,
    pair_within: timedelta = timedelta(hours=1),
) -> list[ComparisonResult]:
    """Pair self-decisions with externals on (symbol, ~timestamp) and compare.

    `pair_within` controls the timestamp tolerance for pairing — a self
    decision at 12:00 and a vendor signal at 12:30 still match if you
    set this to ≥30 minutes. We do NOT do any timezone gymnastics:
    callers must pass UTC datetimes throughout.
    """
    self_list = list(self_decisions)
    out: list[ComparisonResult] = []
    for ext in external_signals:
        # Find the closest-in-time self decision for the same symbol.
        candidates = [
            (abs(_to_dt(d.get("ts") or d.get("timestamp")) - ext.timestamp), d)
            for d in self_list
            if d.get("symbol") == ext.symbol
            and (d.get("ts") or d.get("timestamp"))
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda t: t[0])
        gap, best = candidates[0]
        if gap > pair_within:
            continue
        out.append(compare_one(best, ext))
    return out


def _to_dt(ts) -> datetime:
    if isinstance(ts, datetime):
        return ts
    return pd.Timestamp(ts).to_pydatetime()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_comparisons(comparisons: Iterable[ComparisonResult]) -> dict:
    rows = list(comparisons)
    if not rows:
        return {"n": 0, "by_source": {}}

    by_source: dict[str, dict] = {}
    for r in rows:
        src = r.external_signal.source
        bucket = by_source.setdefault(src, {
            "n": 0, "action_match": 0, "self_only_directional": 0,
            "external_only_directional": 0, "agreement_score_sum": 0.0,
        })
        bucket["n"] += 1
        bucket["agreement_score_sum"] += r.agreement_score
        if r.action_match:
            bucket["action_match"] += 1
        # Directional disagreements
        sa = (r.self_decision.get("action") or "HOLD").upper()
        ea = (r.external_signal.action or "HOLD").upper()
        if sa != "HOLD" and ea == "HOLD":
            bucket["self_only_directional"] += 1
        if ea != "HOLD" and sa == "HOLD":
            bucket["external_only_directional"] += 1

    for src, b in by_source.items():
        n = max(1, b["n"])
        b["action_match_rate"] = b.pop("action_match") / n
        b["agreement_score_mean"] = b.pop("agreement_score_sum") / n

    return {
        "n": len(rows),
        "by_source": by_source,
    }


__all__ = [
    "ComparisonResult",
    "ExternalSignal",
    "aggregate_comparisons",
    "compare_one",
    "compare_signals",
]
