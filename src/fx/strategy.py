"""Combine the technical signal and the LLM signal into a final action."""
from __future__ import annotations

from dataclasses import dataclass

from .analyst import TradeSignal


@dataclass(frozen=True)
class Decision:
    action: str  # BUY | SELL | HOLD
    confidence: float
    reason: str


def combine(
    technical: str,
    llm: TradeSignal | None,
    min_confidence: float = 0.6,
) -> Decision:
    """Consensus rule: both must agree and LLM confidence must clear threshold.

    If the LLM is unavailable (offline backtest), fall back to the technical
    signal alone.
    """
    if llm is None:
        return Decision(
            action=technical,
            confidence=0.5,
            reason=f"Technical-only signal (no LLM available): {technical}",
        )

    if llm.confidence < min_confidence:
        return Decision(
            action="HOLD",
            confidence=llm.confidence,
            reason=f"LLM confidence {llm.confidence:.2f} below threshold "
            f"{min_confidence}. LLM wanted {llm.action}; technical={technical}",
        )

    if technical == llm.action and technical != "HOLD":
        return Decision(
            action=technical,
            confidence=llm.confidence,
            reason=f"Consensus {technical}: {llm.reason}",
        )

    return Decision(
        action="HOLD",
        confidence=llm.confidence,
        reason=f"Disagreement: technical={technical}, llm={llm.action}. {llm.reason}",
    )
