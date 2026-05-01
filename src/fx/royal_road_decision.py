"""Royal-road decision profile (opt-in BUY/SELL/HOLD).

This module exposes `decide_royal_road(...)`, an alternative to
`decision_engine.decide()` that reads the `technical_confluence_v1`
observation dict and applies the royal-road procedure (see
`docs/technical_baseline_royal_road.md`) to produce a Decision.

Profile contract
----------------
- Default profile remains `current_runtime` (decision_engine.decide).
  When the profile is anything other than `royal_road_decision_v1`,
  this module is NOT consulted, and the engine output is byte-identical
  to PR #21 main.
- When `--decision-profile royal_road_decision_v1` is requested,
  `backtest_engine.run_engine_backtest` calls this function AFTER
  computing the current_runtime decision (so both decisions are
  available for comparison metadata) and uses this function's output
  as the actual entry / exit driver.
- Live / OANDA / paper paths are NEVER routed through this module.
  Live cmd_trade does not call `run_engine_backtest`, so the profile
  flag has no effect on live trading.

V1 design notes
---------------
- The royal-road v1 rules are intentionally STRICT (BUY only on
  STRONG_BUY_SETUP + invalidation_clear + structure_stop + bullish
  evidence + not near_resistance + no avoid_reasons). WEAK_BUY_SETUP
  / WEAK_SELL_SETUP both default to HOLD. The intent is to let the
  next analysis phase show whether tighter rules separate winners,
  not to maximise trade count.
- Risk gate runs FIRST (mirrors `decide_action`). If the gate blocks
  (event_high, daily_loss_cap, etc.), this module returns HOLD.
- Higher-TF counter-trend trades are blocked here too, matching
  decide_action's existing behaviour.
- The actual stop / TP order placement still uses ATR multipliers in
  `backtest_engine`. structure_stop is OBSERVATION ONLY in v1 — the
  royal-road decision considers it for invalidation quality, but does
  not change the actual stop level.
- `compare_decisions` produces the trace metadata so post-hoc analysis
  can answer: which trades did royal-road block that current_runtime
  took? Which trades did royal-road take that current_runtime missed?
- Thresholds (rr floor, "serious avoid" set) live as module constants
  here. They do NOT flow through `parameter_defaults.PARAMETER_BASELINE_V1`
  or `runtime_parameters.py`, so the literature_baseline hash stays
  unchanged.
"""
from __future__ import annotations

from typing import Any, Final

from .decision_engine import Decision, MIN_RISK_REWARD, _hold
from .patterns import PatternResult
from .risk_gate import RiskState, evaluate as evaluate_gate


PROFILE_NAME: Final[str] = "royal_road_decision_v1"

# When `final_confluence.avoid_reasons` contains any of these, the
# royal-road profile blocks. `invalidation_unclear` is intentionally
# NOT here: it surfaces in confluence's avoid_reasons too but is
# already enforced explicitly via `risk_plan_obs.invalidation_clear`.
_SERIOUS_AVOID_REASONS: Final[tuple[str, ...]] = (
    "fake_breakout",
    "regime_volatile",
)

# Confidence floor when royal-road emits BUY/SELL. Score ranges are
# constrained -1..+1 by build_technical_confluence; we map the
# magnitude of |score| onto a 0.55..0.85 confidence band so trace
# consumers see a meaningful signal-strength.
_BASE_CONFIDENCE: Final[float] = 0.55
_CONFIDENCE_FROM_SCORE_GAIN: Final[float] = 0.30


def _confidence_from_score(score: float | None) -> float:
    if score is None:
        return _BASE_CONFIDENCE
    return min(
        0.95,
        _BASE_CONFIDENCE + _CONFIDENCE_FROM_SCORE_GAIN * abs(float(score)),
    )


def _structure_says_bullish(tc: dict) -> bool:
    dow = tc.get("dow_structure") or {}
    code = dow.get("structure_code")
    chart = tc.get("chart_pattern") or {}
    regime = tc.get("market_regime")
    return bool(
        regime == "TREND_UP"
        or code in ("HH", "HL")
        or dow.get("bos_up")
        or chart.get("double_bottom") or chart.get("triple_bottom")
        or chart.get("inverse_head_and_shoulders")
    )


def _structure_says_bearish(tc: dict) -> bool:
    dow = tc.get("dow_structure") or {}
    code = dow.get("structure_code")
    chart = tc.get("chart_pattern") or {}
    regime = tc.get("market_regime")
    return bool(
        regime == "TREND_DOWN"
        or code in ("LL", "LH")
        or dow.get("bos_down")
        or chart.get("double_top") or chart.get("triple_top")
        or chart.get("head_and_shoulders")
    )


def _bullish_candle(tc: dict) -> bool:
    cs = tc.get("candlestick_signal") or {}
    return bool(
        cs.get("bullish_pinbar")
        or cs.get("bullish_engulfing")
        or cs.get("strong_bull_body")
    )


def _bearish_candle(tc: dict) -> bool:
    cs = tc.get("candlestick_signal") or {}
    return bool(
        cs.get("bearish_pinbar")
        or cs.get("bearish_engulfing")
        or cs.get("strong_bear_body")
    )


def _serious_avoid_hits(tc: dict) -> list[str]:
    final = tc.get("final_confluence") or {}
    return [
        r for r in (final.get("avoid_reasons") or [])
        if r in _SERIOUS_AVOID_REASONS
    ]


def _htf_blocker(higher_timeframe_trend: str | None, side: str) -> str | None:
    htf = (higher_timeframe_trend or "UNKNOWN").upper()
    if side == "BUY" and htf == "DOWNTREND":
        return "htf_counter_trend_for_buy"
    if side == "SELL" and htf == "UPTREND":
        return "htf_counter_trend_for_sell"
    return None


def _build_advisory(
    *,
    score: float,
    reasons: list[str],
    block_reasons: list[str],
) -> dict:
    return {
        "profile": PROFILE_NAME,
        "score": float(score),
        "reasons": list(reasons),
        "block_reasons": list(block_reasons),
        "source": "technical_confluence_v1",
    }


def decide_royal_road(
    *,
    technical_signal: str,
    pattern: PatternResult | None,
    higher_timeframe_trend: str | None,
    risk_reward: float | None,
    risk_state: RiskState,
    technical_confluence: dict,
    min_risk_reward: float = MIN_RISK_REWARD,
) -> Decision:
    """Run the royal-road decision pipeline.

    Returns a Decision whose `advisory` block carries the royal-road
    profile audit (score, reasons, block_reasons) so the caller can
    fold it into the trace.

    The function is pure — it does not write to RiskState, does not
    mutate the technical_confluence dict, and does not call out to
    LLMs / external services.
    """
    chain: list[str] = ["risk_gate"]
    reasons: list[str] = []
    block_reasons: list[str] = []

    # Step 1: risk gate (event_high, data_quality, daily_loss_cap, etc.)
    gate = evaluate_gate(risk_state)
    if not gate.allow_trade:
        codes = tuple(gate.blocked_codes)
        return _hold(
            reason="risk gate blocked",
            blocked_by=codes,
            chain=tuple(chain),
            confidence=0.0,
            advisory=_build_advisory(
                score=0.0,
                reasons=[],
                block_reasons=[f"gate:{c}" for c in codes],
            ),
        )

    final = (technical_confluence or {}).get("final_confluence") or {}
    label = final.get("label") or "UNKNOWN"
    score = float(final.get("score") or 0.0)
    bullish_obs = list(final.get("bullish_reasons") or [])
    bearish_obs = list(final.get("bearish_reasons") or [])

    sr = (technical_confluence or {}).get("support_resistance") or {}
    risk_obs = (technical_confluence or {}).get("risk_plan_obs") or {}

    # Step 9: invalidation gates — required for ANY directional setup.
    if not risk_obs.get("invalidation_clear"):
        block_reasons.append("invalidation_unclear")
    if risk_obs.get("structure_stop_price") is None:
        block_reasons.append("structure_stop_missing")

    # Step 9: risk-reward floor. ATR-based RR comes from the engine;
    # the structure-based RR sits in technical_confluence.
    if risk_reward is not None and risk_reward < min_risk_reward:
        block_reasons.append(f"rr<{min_risk_reward}")

    # Step 8 (avoid): serious avoid_reasons block (fake breakout etc.).
    for r in _serious_avoid_hits(technical_confluence):
        block_reasons.append(f"avoid:{r}")

    chain.append("invalidation_check")
    chain.append("avoid_check")

    if label == "STRONG_BUY_SETUP":
        chain.append("strong_buy_path")
        # Step 4: position-quality
        if sr.get("near_resistance"):
            block_reasons.append("near_resistance_for_buy")
        # Step 3 / 4 / 6: at least one bullish evidence axis
        bullish_evidence = (
            _structure_says_bullish(technical_confluence)
            or sr.get("near_support")
            or _bullish_candle(technical_confluence)
        )
        if not bullish_evidence:
            block_reasons.append("no_bullish_evidence")
        # Step 2: higher-TF alignment
        htf_block = _htf_blocker(higher_timeframe_trend, "BUY")
        if htf_block:
            block_reasons.append(htf_block)

        if not block_reasons:
            reasons.extend(bullish_obs)
            reasons.append("STRONG_BUY_SETUP_confirmed")
            return Decision(
                action="BUY",
                confidence=_confidence_from_score(score),
                reason="royal_road_decision_v1: BUY",
                blocked_by=(),
                rule_chain=tuple(chain),
                advisory=_build_advisory(
                    score=score, reasons=reasons, block_reasons=[],
                ),
            )

    elif label == "STRONG_SELL_SETUP":
        chain.append("strong_sell_path")
        if sr.get("near_support"):
            block_reasons.append("near_support_for_sell")
        bearish_evidence = (
            _structure_says_bearish(technical_confluence)
            or sr.get("near_resistance")
            or _bearish_candle(technical_confluence)
        )
        if not bearish_evidence:
            block_reasons.append("no_bearish_evidence")
        htf_block = _htf_blocker(higher_timeframe_trend, "SELL")
        if htf_block:
            block_reasons.append(htf_block)

        if not block_reasons:
            reasons.extend(bearish_obs)
            reasons.append("STRONG_SELL_SETUP_confirmed")
            return Decision(
                action="SELL",
                confidence=_confidence_from_score(score),
                reason="royal_road_decision_v1: SELL",
                blocked_by=(),
                rule_chain=tuple(chain),
                advisory=_build_advisory(
                    score=score, reasons=reasons, block_reasons=[],
                ),
            )

    elif label in ("WEAK_BUY_SETUP", "WEAK_SELL_SETUP"):
        chain.append("weak_setup_hold")
        block_reasons.append(f"weak_setup:{label}")
    else:
        # NO_TRADE / AVOID_TRADE / UNKNOWN
        chain.append("non_directional_label")
        block_reasons.append(f"label:{label}")

    return _hold(
        reason="; ".join(block_reasons) if block_reasons else "no_directional_setup",
        blocked_by=tuple(block_reasons),
        chain=tuple(chain),
        confidence=0.0,
        advisory=_build_advisory(
            score=score, reasons=reasons, block_reasons=block_reasons,
        ),
    )


# ---------------------------------------------------------------------------
# Comparison metadata for trace
# ---------------------------------------------------------------------------


def compare_decisions(
    *,
    decision_current: Decision,
    decision_royal: Decision,
) -> dict:
    """Build the `compared_to_current_runtime` block embedded in trace.

    The closed `difference_type` taxonomy is intentional so post-hoc
    analyses can pivot the trade pool by one of:
      same | current_buy_royal_hold | current_sell_royal_hold |
      current_hold_royal_buy | current_hold_royal_sell |
      opposite_direction | other
    """
    cur = decision_current.action
    roy = decision_royal.action
    if cur == roy:
        diff = "same"
    elif cur == "BUY" and roy == "HOLD":
        diff = "current_buy_royal_hold"
    elif cur == "SELL" and roy == "HOLD":
        diff = "current_sell_royal_hold"
    elif cur == "HOLD" and roy == "BUY":
        diff = "current_hold_royal_buy"
    elif cur == "HOLD" and roy == "SELL":
        diff = "current_hold_royal_sell"
    elif (cur, roy) in {("BUY", "SELL"), ("SELL", "BUY")}:
        diff = "opposite_direction"
    else:
        diff = "other"
    return {
        "current_action": cur,
        "royal_road_action": roy,
        "same_action": (cur == roy),
        "difference_type": diff,
    }


# ---------------------------------------------------------------------------
# Profile name validation helper (used by CLI + backtest_engine)
# ---------------------------------------------------------------------------


SUPPORTED_DECISION_PROFILES: Final[tuple[str, ...]] = (
    "current_runtime",
    PROFILE_NAME,
)


def validate_decision_profile(name: str) -> str:
    if name not in SUPPORTED_DECISION_PROFILES:
        raise ValueError(
            f"unknown --decision-profile: {name!r}. "
            f"Supported: {', '.join(SUPPORTED_DECISION_PROFILES)}"
        )
    return name


__all__ = [
    "PROFILE_NAME",
    "SUPPORTED_DECISION_PROFILES",
    "decide_royal_road",
    "compare_decisions",
    "validate_decision_profile",
]
