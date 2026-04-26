"""DEPRECATED — kept only for backward compatibility.

The real Decision Engine lives in `src/fx/decision_engine.py`. This
file used to host a permissive `combine()` that was the de facto
decision authority — that was the danger AUDIT.md flagged. The new
engine adds a strict Risk Gate ahead of the consensus check and runs
the entire chain laid out in spec §10.

Anything that imports `combine` here will get a thin shim that calls
the real engine with default risk-state. New code MUST use
`decision_engine.decide` directly so the Risk Gate inputs are visible
at the call site.
"""
from __future__ import annotations

from dataclasses import dataclass

from .analyst import TradeSignal
from .decision_engine import decide as _decide
from .risk_gate import RiskState


@dataclass(frozen=True)
class Decision:
    """Backward-compatible Decision shape. New code: decision_engine.Decision."""

    action: str
    confidence: float
    reason: str


def combine(
    technical: str,
    llm: TradeSignal | None,
    min_confidence: float = 0.6,
) -> Decision:
    """Compatibility shim — call decision_engine.decide() directly in new code."""
    result = _decide(
        technical_signal=technical,
        pattern=None,
        higher_timeframe_trend=None,
        risk_reward=None,
        risk_state=RiskState(),
        llm_signal=llm,
        min_confidence=min_confidence,
    )
    return Decision(
        action=result.action,
        confidence=result.confidence,
        reason=result.reason,
    )
