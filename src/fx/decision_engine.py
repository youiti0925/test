"""Fixed-rule Decision Engine.

This is the ONLY place in the codebase that decides BUY/SELL/HOLD for a
real entry. It consults inputs in spec-order:

  Risk Gate
    → Pattern check (e.g. triple-top requires neckline break)
    → Higher-timeframe alignment
    → Risk-reward floor
    → SNS-only veto (sentiment alone never triggers a trade)
    → Confidence floor
    → Technical / LLM agreement
    → Waveform-bias agreement (advisory only)
    → Action

LLM authority — read carefully (asymmetric)
-------------------------------------------
The LLM is "advisory" only in one direction: it CANNOT lift HOLD into
BUY/SELL, and it CANNOT change the side away from the technical signal.

But it DOES have **veto power** in the conservative direction:
  * `llm_signal.confidence < min_confidence` → HOLD
  * `llm_signal.action != technical_signal` → HOLD

So the accurate description is "LLM cannot start a trade, but it can
stop one." This is intentional and matches the spec's
"AIは保守的に止める権限を持つ" framing — the bot would rather miss
a trade than enter one the LLM is unconfident about.

What the LLM can NEVER do:
  * Override the Risk Gate (test_ai_cannot_override_risk_gate pins this).
  * Lift HOLD into BUY/SELL.
  * Pick a direction the technical signal doesn't already favour.

Waveform bias (spec §7.4) follows the same asymmetric pattern:
  * A BUY-only waveform_bias on a HOLD technical signal does NOT trade.
  * A waveform_bias agreeing with technical_signal is recorded in the
    advisory payload and contributes a small confidence bump.
  * A waveform_bias DISAGREEING with the technical signal vetoes to HOLD.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .analyst import TradeSignal
from .patterns import PatternResult, TrendState
from .risk_gate import GateResult, RiskState, evaluate as evaluate_gate
from .waveform_backtest import WaveformBias

MIN_CONFIDENCE = 0.6
MIN_RISK_REWARD = 1.5


@dataclass(frozen=True)
class Decision:
    action: str                # BUY | SELL | HOLD
    confidence: float          # 0.0 .. 1.0
    reason: str
    blocked_by: tuple[str, ...] = ()
    rule_chain: tuple[str, ...] = ()  # gates / rules consulted, in order
    advisory: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "blocked_by": list(self.blocked_by),
            "rule_chain": list(self.rule_chain),
            "advisory": self.advisory,
        }


def _hold(reason: str, *, blocked_by=(), chain=(), confidence=0.0,
          advisory: dict | None = None) -> Decision:
    return Decision(
        action="HOLD",
        confidence=confidence,
        reason=reason,
        blocked_by=tuple(blocked_by),
        rule_chain=tuple(chain),
        advisory=advisory or {},
    )


def decide(
    *,
    technical_signal: str,
    pattern: PatternResult | None = None,
    higher_timeframe_trend: str | None = None,
    risk_reward: float | None = None,
    risk_state: RiskState,
    llm_signal: TradeSignal | None = None,
    waveform_bias: "WaveformBias | None" = None,
    min_confidence: float = MIN_CONFIDENCE,
    min_risk_reward: float = MIN_RISK_REWARD,
) -> Decision:
    """Run the rule chain and return one Decision.

    `risk_state` is the bundle the gate evaluates. Even if you have no
    sentiment data or PnL today, pass `RiskState()` with what you do
    have — the gates skip checks they can't run.
    """
    chain: list[str] = []

    # ── Gate ──────────────────────────────────────────────────────────
    chain.append("risk_gate")
    gate: GateResult = evaluate_gate(risk_state)
    if not gate.allow_trade:
        return _hold(
            f"Risk gate blocked: {gate.block.message if gate.block else 'unknown'}",
            blocked_by=gate.blocked_codes,
            chain=chain,
            advisory=_merge_advisory(llm_signal, waveform_bias),
        )

    # ── Technical signal must be directional ──────────────────────────
    chain.append("technical_directionality")
    if technical_signal not in ("BUY", "SELL"):
        return _hold(
            f"Technical signal is {technical_signal}; nothing to confirm",
            chain=chain,
            advisory=_merge_advisory(llm_signal, waveform_bias),
        )

    # ── Pattern check: top patterns require neckline break ───────────
    chain.append("pattern_check")
    if pattern is not None and pattern.detected_pattern is not None:
        is_top = pattern.detected_pattern in (
            "DOUBLE_TOP_CANDIDATE", "TRIPLE_TOP_CANDIDATE", "HEAD_AND_SHOULDERS",
        )
        is_bottom = pattern.detected_pattern in (
            "DOUBLE_BOTTOM_CANDIDATE", "TRIPLE_BOTTOM_CANDIDATE",
            "INVERSE_HEAD_AND_SHOULDERS",
        )
        if is_top and technical_signal == "SELL" and not pattern.neckline_broken:
            return _hold(
                f"{pattern.detected_pattern}: neckline not yet broken on close",
                chain=chain,
                advisory=_merge_advisory(llm_signal, waveform_bias),
            )
        if is_bottom and technical_signal == "BUY" and not pattern.neckline_broken:
            return _hold(
                f"{pattern.detected_pattern}: neckline not yet broken on close",
                chain=chain,
                advisory=_merge_advisory(llm_signal, waveform_bias),
            )

    # ── Higher-timeframe alignment ───────────────────────────────────
    chain.append("higher_tf_alignment")
    if higher_timeframe_trend:
        ht = higher_timeframe_trend.upper()
        if technical_signal == "BUY" and ht == TrendState.DOWNTREND.value:
            return _hold(
                "Counter-trend BUY against higher-timeframe DOWNTREND",
                chain=chain,
                advisory=_merge_advisory(llm_signal, waveform_bias),
            )
        if technical_signal == "SELL" and ht == TrendState.UPTREND.value:
            return _hold(
                "Counter-trend SELL against higher-timeframe UPTREND",
                chain=chain,
                advisory=_merge_advisory(llm_signal, waveform_bias),
            )

    # ── Risk-reward floor ────────────────────────────────────────────
    chain.append("risk_reward_floor")
    if risk_reward is not None and risk_reward < min_risk_reward:
        return _hold(
            f"Risk-reward {risk_reward:.2f} below floor {min_risk_reward}",
            chain=chain,
            advisory=_merge_advisory(llm_signal, waveform_bias),
        )

    # ── LLM consensus (asymmetric: can veto, cannot promote) ─────────
    # Low confidence or disagreement with the technical signal blocks
    # the trade; the LLM never lifts HOLD into BUY/SELL. See module
    # docstring for the full asymmetry explanation.
    chain.append("llm_advisory")
    if llm_signal is not None:
        if llm_signal.confidence < min_confidence:
            return _hold(
                f"LLM confidence {llm_signal.confidence:.2f} < {min_confidence}",
                chain=chain,
                confidence=llm_signal.confidence,
                advisory=_merge_advisory(llm_signal, waveform_bias),
            )
        if llm_signal.action != technical_signal:
            return _hold(
                f"Tech={technical_signal} disagrees with LLM={llm_signal.action}",
                chain=chain,
                confidence=llm_signal.confidence,
                advisory=_merge_advisory(llm_signal, waveform_bias),
            )

    # ── Waveform bias (advisory veto only) ───────────────────────────
    # Per spec §7.4: a directional waveform bias that DISAGREES with the
    # technical signal forces HOLD; agreement is recorded but doesn't
    # lift confidence above the LLM's, and an absent / HOLD bias is
    # neutral. Waveform alone never produces an entry — that's enforced
    # by the fact that we only get here when technical_signal is
    # directional and all earlier gates passed.
    chain.append("waveform_advisory")
    if waveform_bias is not None and waveform_bias.action in ("BUY", "SELL"):
        if waveform_bias.action != technical_signal:
            return _hold(
                f"Waveform bias {waveform_bias.action} "
                f"({waveform_bias.confidence:.2f}, "
                f"{waveform_bias.sample_count} samples) disagrees with "
                f"technical {technical_signal}",
                chain=chain,
                confidence=(
                    llm_signal.confidence if llm_signal is not None else 0.0
                ),
                advisory=_merge_advisory(llm_signal, waveform_bias),
            )

    # ── All checks passed ────────────────────────────────────────────
    chain.append("approve")
    base_confidence = (
        llm_signal.confidence if llm_signal is not None else 0.55
    )
    # Waveform agreement contributes a small additive bump capped at 0.95
    # — never enough to override the LLM/min_confidence floors above.
    confidence = base_confidence
    if (waveform_bias is not None
            and waveform_bias.action == technical_signal
            and waveform_bias.sample_count > 0):
        confidence = min(0.95, base_confidence + 0.05 * waveform_bias.confidence)

    reason_parts = [f"Rule chain approved {technical_signal}"]
    if llm_signal is not None:
        reason_parts.append(f"LLM agrees ({llm_signal.confidence:.2f}): {llm_signal.reason}")
    if pattern and pattern.detected_pattern:
        reason_parts.append(
            f"pattern={pattern.detected_pattern}, neckline_broken={pattern.neckline_broken}"
        )
    if waveform_bias is not None and waveform_bias.action == technical_signal:
        reason_parts.append(
            f"waveform agrees ({waveform_bias.sample_count} similar past windows, "
            f"conf={waveform_bias.confidence:.2f})"
        )
    return Decision(
        action=technical_signal,
        confidence=confidence,
        reason=" | ".join(reason_parts),
        blocked_by=(),
        rule_chain=tuple(chain),
        advisory=_merge_advisory(llm_signal, waveform_bias),
    )


def _advisory_from_llm(llm: TradeSignal | None) -> dict:
    if llm is None:
        return {}
    return {
        "llm_action": llm.action,
        "llm_confidence": llm.confidence,
        "llm_reason": llm.reason,
        "llm_key_risks": list(llm.key_risks),
        "llm_expected_direction": llm.expected_direction,
        "llm_expected_magnitude_pct": llm.expected_magnitude_pct,
    }


def _advisory_from_waveform(bias: "WaveformBias | None") -> dict:
    if bias is None:
        return {}
    return {"waveform_bias": bias.to_dict()}


def _merge_advisory(
    llm: TradeSignal | None, bias: "WaveformBias | None"
) -> dict:
    out = _advisory_from_llm(llm)
    out.update(_advisory_from_waveform(bias))
    return out
