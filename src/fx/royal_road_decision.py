"""Royal-road decision profile (opt-in BUY/SELL/HOLD).

This module reads ``technical_confluence_v1`` and returns a Decision for
``royal_road_decision_v1``. The royal-road procedure is separated from the
strictness settings in ``royal_road_decision_modes.py`` so heuristic values
are explicit and reviewable.
"""
from __future__ import annotations

from typing import Final

from .decision_engine import Decision, MIN_RISK_REWARD, _hold
from .patterns import PatternResult
from .risk_gate import RiskState, evaluate as evaluate_gate
from .royal_road_decision_modes import (
    DEFAULT_ROYAL_ROAD_MODE,
    RoyalRoadDecisionModeConfig,
    get_royal_road_mode_config,
    supported_royal_road_modes,
)


PROFILE_NAME: Final[str] = "royal_road_decision_v1"
_BASE_CONFIDENCE: Final[float] = 0.55
_CONFIDENCE_FROM_SCORE_GAIN: Final[float] = 0.30


def _confidence_from_score(score: float | None) -> float:
    if score is None:
        return _BASE_CONFIDENCE
    return min(0.95, _BASE_CONFIDENCE + _CONFIDENCE_FROM_SCORE_GAIN * abs(float(score)))


def _structure_says_bullish(tc: dict) -> bool:
    dow = tc.get("dow_structure") or {}
    chart = tc.get("chart_pattern") or {}
    return bool(
        tc.get("market_regime") == "TREND_UP"
        or dow.get("structure_code") in ("HH", "HL")
        or dow.get("bos_up")
        or chart.get("double_bottom")
        or chart.get("triple_bottom")
        or chart.get("inverse_head_and_shoulders")
    )


def _structure_says_bearish(tc: dict) -> bool:
    dow = tc.get("dow_structure") or {}
    chart = tc.get("chart_pattern") or {}
    return bool(
        tc.get("market_regime") == "TREND_DOWN"
        or dow.get("structure_code") in ("LL", "LH")
        or dow.get("bos_down")
        or chart.get("double_top")
        or chart.get("triple_top")
        or chart.get("head_and_shoulders")
    )


def _bullish_candle(tc: dict) -> bool:
    cs = tc.get("candlestick_signal") or {}
    return bool(cs.get("bullish_pinbar") or cs.get("bullish_engulfing") or cs.get("strong_bull_body"))


def _bearish_candle(tc: dict) -> bool:
    cs = tc.get("candlestick_signal") or {}
    return bool(cs.get("bearish_pinbar") or cs.get("bearish_engulfing") or cs.get("strong_bear_body"))


def _evidence_axes(tc: dict, side: str) -> list[str]:
    sr = tc.get("support_resistance") or {}
    chart = tc.get("chart_pattern") or {}
    axes: list[str] = []
    if side == "BUY":
        if _structure_says_bullish(tc):
            axes.append("bullish_structure")
        if sr.get("near_support"):
            axes.append("near_support")
        if sr.get("role_reversal"):
            axes.append("role_reversal_support_candidate")
        if _bullish_candle(tc):
            axes.append("bullish_candlestick")
        if chart.get("neckline_broken") and (chart.get("double_bottom") or chart.get("triple_bottom")):
            axes.append("bullish_neckline_break")
    else:
        if _structure_says_bearish(tc):
            axes.append("bearish_structure")
        if sr.get("near_resistance"):
            axes.append("near_resistance")
        if sr.get("role_reversal"):
            axes.append("role_reversal_resistance_candidate")
        if _bearish_candle(tc):
            axes.append("bearish_candlestick")
        if chart.get("neckline_broken") and (chart.get("double_top") or chart.get("triple_top")):
            axes.append("bearish_neckline_break")
    return axes


def _label_to_side(label: str) -> str | None:
    if label in ("STRONG_BUY_SETUP", "WEAK_BUY_SETUP"):
        return "BUY"
    if label in ("STRONG_SELL_SETUP", "WEAK_SELL_SETUP"):
        return "SELL"
    return None


def _htf_blocker(higher_timeframe_trend: str | None, side: str) -> str | None:
    htf = (higher_timeframe_trend or "UNKNOWN").upper()
    if side == "BUY" and htf == "DOWNTREND":
        return "htf_counter_trend_for_buy"
    if side == "SELL" and htf == "UPTREND":
        return "htf_counter_trend_for_sell"
    return None


def _serious_avoid_hits(tc: dict, cfg: RoyalRoadDecisionModeConfig) -> list[str]:
    final = tc.get("final_confluence") or {}
    return [r for r in (final.get("avoid_reasons") or []) if r in cfg.serious_avoid_reasons]


def _build_advisory(
    *,
    score: float,
    reasons: list[str],
    block_reasons: list[str],
    mode: str,
    cautions: list[str] | None = None,
) -> dict:
    cfg = get_royal_road_mode_config(mode)
    return {
        "profile": PROFILE_NAME,
        "mode": mode,
        "mode_status": cfg.status,
        "mode_needs_validation": cfg.needs_validation,
        "score": float(score),
        "reasons": list(reasons),
        "block_reasons": list(block_reasons),
        "cautions": list(cautions or []),
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
    mode: str = DEFAULT_ROYAL_ROAD_MODE,
) -> Decision:
    """Run the royal-road decision pipeline.

    Default mode is ``balanced``. ``strict`` is diagnostic and keeps the
    original conservative behaviour; ``exploratory`` is for discovery.
    """
    cfg = get_royal_road_mode_config(mode)
    tc = technical_confluence or {}
    chain: list[str] = ["risk_gate", f"royal_mode:{cfg.name}"]
    reasons: list[str] = []
    block_reasons: list[str] = []
    cautions: list[str] = []

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
                mode=cfg.name,
            ),
        )

    final = tc.get("final_confluence") or {}
    label = final.get("label") or "UNKNOWN"
    score = float(final.get("score") or 0.0)
    side = _label_to_side(label)
    if side is None:
        block_reasons.append(f"label:{label}")
        return _hold(
            reason="; ".join(block_reasons),
            blocked_by=tuple(block_reasons),
            chain=tuple(chain + ["non_directional_label"]),
            confidence=0.0,
            advisory=_build_advisory(score=score, reasons=[], block_reasons=block_reasons, mode=cfg.name),
        )

    is_weak = label.startswith("WEAK_")
    if is_weak and not cfg.allow_weak_entries:
        block_reasons.append(f"weak_setup:{label}")

    sr = tc.get("support_resistance") or {}
    risk_obs = tc.get("risk_plan_obs") or {}
    invalidation_clear = bool(risk_obs.get("invalidation_clear"))
    structure_stop_exists = risk_obs.get("structure_stop_price") is not None

    if not invalidation_clear:
        if cfg.require_invalidation_clear:
            block_reasons.append("invalidation_unclear")
        else:
            cautions.append("invalidation_unclear_soft")
    if not structure_stop_exists:
        if cfg.require_structure_stop:
            block_reasons.append("structure_stop_missing")
        elif cfg.allow_structure_stop_soft_fail:
            cautions.append("structure_stop_missing_soft")

    if risk_reward is not None and risk_reward < min_risk_reward:
        block_reasons.append(f"rr<{min_risk_reward}")
    for r in _serious_avoid_hits(tc, cfg):
        block_reasons.append(f"avoid:{r}")

    if cfg.block_opposite_level:
        if side == "BUY" and sr.get("near_resistance"):
            block_reasons.append("near_resistance_for_buy")
        if side == "SELL" and sr.get("near_support"):
            block_reasons.append("near_support_for_sell")

    if cfg.htf_counter_trend_blocks:
        htf_block = _htf_blocker(higher_timeframe_trend, side)
        if htf_block:
            block_reasons.append(htf_block)

    axes = _evidence_axes(tc, side)
    min_axes = cfg.min_evidence_axes_weak if is_weak else cfg.min_evidence_axes_strong
    if len(axes) < min_axes:
        block_reasons.append(f"insufficient_{side.lower()}_evidence_axes:{len(axes)}<{min_axes}")

    if block_reasons:
        return _hold(
            reason="; ".join(block_reasons),
            blocked_by=tuple(block_reasons),
            chain=tuple(chain + ["risk_quality_check", "evidence_axis_check"]),
            confidence=0.0,
            advisory=_build_advisory(
                score=score,
                reasons=reasons,
                block_reasons=block_reasons,
                cautions=cautions,
                mode=cfg.name,
            ),
        )

    if side == "BUY":
        reasons.extend(list(final.get("bullish_reasons") or []))
    else:
        reasons.extend(list(final.get("bearish_reasons") or []))
    reasons.extend(axes)
    reasons.extend(f"caution:{c}" for c in cautions)
    reasons.append(f"{label}_confirmed_by_{cfg.name}")

    return Decision(
        action=side,
        confidence=_confidence_from_score(score),
        reason=f"royal_road_decision_v1[{cfg.name}]: {side}",
        blocked_by=(),
        rule_chain=tuple(chain + ["risk_quality_check", "evidence_axis_check"]),
        advisory=_build_advisory(score=score, reasons=reasons, block_reasons=[], cautions=cautions, mode=cfg.name),
    )


def compare_decisions(*, decision_current: Decision, decision_royal: Decision) -> dict:
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
        "same_action": cur == roy,
        "difference_type": diff,
    }


SUPPORTED_DECISION_PROFILES: Final[tuple[str, ...]] = ("current_runtime", PROFILE_NAME)


def validate_decision_profile(name: str) -> str:
    if name not in SUPPORTED_DECISION_PROFILES:
        raise ValueError(
            f"unknown --decision-profile: {name!r}. Supported: {', '.join(SUPPORTED_DECISION_PROFILES)}"
        )
    return name


__all__ = [
    "PROFILE_NAME",
    "SUPPORTED_DECISION_PROFILES",
    "decide_royal_road",
    "compare_decisions",
    "validate_decision_profile",
]
