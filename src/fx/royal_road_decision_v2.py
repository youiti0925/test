"""royal_road_decision_v2 — opt-in research decision profile.

v2 reads:
  - technical_confluence_v1 (existing observation layer)
  - support_resistance.detect_levels (level clustering)
  - trendlines.detect_trendlines
  - chart_patterns.detect_patterns
  - lower_timeframe_trigger.detect_lower_tf_trigger (when df_lower_tf attached)
  - macro_alignment.compute_macro_alignment
  - stop_modes.plan_stop

and produces a `Decision` whose advisory carries the v2 audit
(profile=royal_road_decision_v2, mode, evidence axes, all v2 sub
snapshots).

Key differences from v1
-----------------------
- Direction sources broaden: bullish_structure / strong_support /
  role_reversal / chart_pattern_breakout / lower_tf_trigger / macro_score
  all count as evidence axes.
- Stop placement is tied to the v2 layer: when `stop_mode != atr`,
  the engine consults the v2 structure stop plan, and the v2 decision
  must agree the plan is valid (else HOLD).
- Macro alignment can BLOCK on `macro_strong_against` (VERY_HIGH VIX
  with the wrong side). DXY / yield alignment otherwise contribute to
  the score, not to a hard block.
- Lower-TF trigger is REQUIRED for setups that lack other axes (e.g.
  WEAK_*_SETUP with only label-level evidence).

Default profile remains current_runtime. v2 is opt-in via
`--decision-profile royal_road_decision_v2`. Live / OANDA / paper
paths do not call this.
"""
from __future__ import annotations

from typing import Final

import pandas as pd

from .chart_patterns import ChartPatternSnapshot, detect_patterns
from .chart_reconstruction import reconstruct_chart_multi_scale
from .decision_engine import Decision, MIN_RISK_REWARD, _hold
from .lower_timeframe_trigger import (
    LowerTimeframeTrigger,
    detect_lower_tf_trigger,
    empty_trigger,
)
from .macro_alignment import (
    MacroAlignmentSnapshot,
    compute_macro_alignment,
    empty_alignment,
)
from .patterns import PatternResult
from .risk_gate import RiskState, evaluate as evaluate_gate
from .royal_road_decision import (
    PROFILE_NAME as PROFILE_V1,
    compare_decisions as compare_v1_decisions,
)
from .royal_road_decision_modes import (
    DEFAULT_ROYAL_ROAD_MODE,
    get_royal_road_mode_config,
)
from .stop_modes import DEFAULT_STOP_MODE, StopPlan, plan_stop
from .support_resistance import SRSnapshot, detect_levels
from .trendlines import TrendlineContext, detect_trendlines


PROFILE_NAME_V2: Final[str] = "royal_road_decision_v2"

_BASE_CONFIDENCE: Final[float] = 0.55
_CONFIDENCE_FROM_SCORE_GAIN: Final[float] = 0.30
_MACRO_BLOCK_THRESHOLD: Final[float] = -0.7   # ≤ this → strong against trade

# Strong S/R near-side touch threshold mirrors support_resistance
_NEAR_LEVEL_ATR: Final[float] = 0.5

# v2.2: minimum total reconstruction_score for an entry to be allowed.
# Below this, even a "STRONG_BUY_SETUP + axes" entry is HOLDed with
# reason "insufficient_royal_road_reconstruction_quality". Tunable
# heuristic, NOT in PARAMETER_BASELINE_V1.
_MIN_RECONSTRUCTION_SCORE: Final[float] = 0.40


def _confidence_from_score(score: float | None) -> float:
    if score is None:
        return _BASE_CONFIDENCE
    return min(
        0.95,
        _BASE_CONFIDENCE + _CONFIDENCE_FROM_SCORE_GAIN * abs(float(score)),
    )


def _bullish_axes_v2(
    *,
    tc: dict,
    sr: SRSnapshot,
    tl: TrendlineContext,
    cp: ChartPatternSnapshot,
    ltf: LowerTimeframeTrigger,
    macro: MacroAlignmentSnapshot,
) -> dict[str, bool]:
    dow = tc.get("dow_structure") or {}
    return {
        "bullish_structure": bool(
            tc.get("market_regime") == "TREND_UP"
            or dow.get("structure_code") in ("HH", "HL")
            or dow.get("bos_up")
        ),
        "near_strong_support": bool(sr.near_strong_support),
        "ascending_trendline_intact": bool(
            tl.ascending_support is not None and not tl.ascending_support.broken
        ),
        "descending_trendline_broken_up": bool(
            tl.descending_resistance is not None and tl.descending_resistance.broken
        ),
        "chart_pattern_bullish_breakout": bool(cp.bullish_breakout_confirmed),
        "lower_tf_bullish_trigger": bool(ltf.bullish_trigger),
        "macro_buy_score": bool(macro.macro_score >= 0.5),
        "sr_role_reversal_bullish": bool(sr.role_reversal and sr.near_strong_support),
        "sr_pullback_bullish": bool(sr.pullback and not sr.near_strong_resistance),
    }


def _bearish_axes_v2(
    *,
    tc: dict,
    sr: SRSnapshot,
    tl: TrendlineContext,
    cp: ChartPatternSnapshot,
    ltf: LowerTimeframeTrigger,
    macro: MacroAlignmentSnapshot,
) -> dict[str, bool]:
    dow = tc.get("dow_structure") or {}
    return {
        "bearish_structure": bool(
            tc.get("market_regime") == "TREND_DOWN"
            or dow.get("structure_code") in ("LL", "LH")
            or dow.get("bos_down")
        ),
        "near_strong_resistance": bool(sr.near_strong_resistance),
        "descending_trendline_intact": bool(
            tl.descending_resistance is not None
            and not tl.descending_resistance.broken
        ),
        "ascending_trendline_broken_down": bool(
            tl.ascending_support is not None and tl.ascending_support.broken
        ),
        "chart_pattern_bearish_breakout": bool(cp.bearish_breakout_confirmed),
        "lower_tf_bearish_trigger": bool(ltf.bearish_trigger),
        "macro_sell_score": bool(macro.macro_score <= -0.5),
        "sr_role_reversal_bearish": bool(
            sr.role_reversal and sr.near_strong_resistance
        ),
        "sr_pullback_bearish": bool(
            sr.pullback and not sr.near_strong_support
        ),
    }


def _label_to_side(label: str) -> str | None:
    if label in ("STRONG_BUY_SETUP", "WEAK_BUY_SETUP"):
        return "BUY"
    if label in ("STRONG_SELL_SETUP", "WEAK_SELL_SETUP"):
        return "SELL"
    return None


def _reconstruction_quality(
    *,
    sr: SRSnapshot,
    tl: TrendlineContext,
    cp: ChartPatternSnapshot,
    ltf: LowerTimeframeTrigger,
    macro: MacroAlignmentSnapshot,
    stop_plan: StopPlan | None,
) -> dict:
    """Compute per-axis reconstruction quality + total score.

    Each axis ∈ [0, 1]; total is a weighted average. The total is
    consulted by the v2 decision: insufficient quality → HOLD with
    reason `insufficient_royal_road_reconstruction_quality`.
    """
    # level: highest-confidence selected zone, or 0
    if sr.selected_level_zones_top5:
        level_q = max(lvl.confidence for lvl in sr.selected_level_zones_top5)
    else:
        level_q = 0.0
    # trendline: highest confidence among selected
    if tl.selected_trendlines_top3:
        tl_q = max(t.confidence for t in tl.selected_trendlines_top3)
    else:
        tl_q = 0.0
    # pattern: highest pattern_quality_score among selected
    if cp.selected_patterns_top5:
        cp_q = max(p.pattern_quality_score for p in cp.selected_patterns_top5)
    else:
        cp_q = 0.0
    # lower-TF: trigger.confidence (0 if unavailable)
    ltf_q = float(getattr(ltf, "confidence", 0.0)) if ltf.available else 0.0
    # macro: |macro_score| (caps at 1)
    macro_q = float(min(1.0, abs(macro.macro_score)))
    # stop-plan: 1.0 if rr_realized >= 1.5 and stop is valid, 0.5 if
    # plan exists with rr<1.5, 0 if invalid/None
    stop_q = 0.0
    if stop_plan is not None and stop_plan.stop_price is not None:
        rr = stop_plan.rr_realized
        if rr is not None and rr >= 1.5:
            stop_q = 1.0
        else:
            stop_q = 0.5
    weights = {
        "level": 0.25, "trendline": 0.10, "pattern": 0.15,
        "lower_tf": 0.15, "macro": 0.15, "stop_plan": 0.20,
    }
    total = (
        level_q * weights["level"]
        + tl_q * weights["trendline"]
        + cp_q * weights["pattern"]
        + ltf_q * weights["lower_tf"]
        + macro_q * weights["macro"]
        + stop_q * weights["stop_plan"]
    )
    return {
        "level_quality_score": float(level_q),
        "trendline_quality_score": float(tl_q),
        "pattern_quality_score": float(cp_q),
        "lower_tf_quality_score": float(ltf_q),
        "macro_quality_score": float(macro_q),
        "stop_plan_quality_score": float(stop_q),
        "total_reconstruction_score": float(total),
        "weights": dict(weights),
    }


def _build_setup_candidates(
    *,
    side: str,
    bullish_axes: dict,
    bearish_axes: dict,
    sr: SRSnapshot,
    tl: TrendlineContext,
    cp: ChartPatternSnapshot,
    ltf: LowerTimeframeTrigger,
    macro: MacroAlignmentSnapshot,
    stop_plan: StopPlan | None,
    label: str,
    score: float,
    reasons_template: list[str],
    block_reasons_template: list[str],
    rr: float | None,
) -> list[dict]:
    """Generate a list of setup candidate dicts (1+ per side at most).

    A candidate aggregates the v2 evidence axes that fired for this
    side along with the per-axis context, the stop plan, and a
    scalar score. The caller can then pick the best one.
    """
    cands: list[dict] = []
    if side not in ("BUY", "SELL"):
        return cands
    axes = bullish_axes if side == "BUY" else bearish_axes
    n_axes = sum(1 for v in axes.values() if v)
    cand_score = float(score) + 0.05 * n_axes
    if rr is not None:
        cand_score += min(0.2, max(0.0, (rr - 1.5) * 0.1))
    cands.append({
        "side": side,
        "score": float(cand_score),
        "confidence": float(min(0.95, 0.55 + 0.2 * n_axes / max(1, len(axes)))),
        "label": label,
        "level_context": {
            "near_strong_support": bool(sr.near_strong_support),
            "near_strong_resistance": bool(sr.near_strong_resistance),
            "n_selected_level_zones": len(sr.selected_level_zones_top5),
        },
        "trendline_context": {
            "bullish_signal": bool(tl.bullish_signal),
            "bearish_signal": bool(tl.bearish_signal),
            "n_selected_trendlines": len(tl.selected_trendlines_top3),
        },
        "pattern_context": {
            "bullish_breakout_confirmed": bool(cp.bullish_breakout_confirmed),
            "bearish_breakout_confirmed": bool(cp.bearish_breakout_confirmed),
            "n_selected_patterns": len(cp.selected_patterns_top5),
        },
        "lower_tf_context": {
            "available": bool(ltf.available),
            "bullish_trigger": bool(ltf.bullish_trigger),
            "bearish_trigger": bool(ltf.bearish_trigger),
            "trigger_type": getattr(ltf, "trigger_type", None),
            "trigger_strength": getattr(ltf, "trigger_strength", None),
        },
        "macro_context": {
            "currency_bias": macro.currency_bias,
            "macro_score": float(macro.macro_score),
            "macro_strong_against": macro.macro_strong_against,
        },
        "stop_plan": (
            stop_plan.to_dict() if stop_plan is not None else None
        ),
        "rr": rr,
        "reasons": list(reasons_template),
        "reject_reasons": list(block_reasons_template),
    })
    return cands


def _select_best_setup(candidates: list[dict]) -> dict | None:
    """Pick the candidate with the highest score that has no
    reject_reasons. Returns None when nothing passes."""
    eligible = [c for c in candidates if not c.get("reject_reasons")]
    if not eligible:
        return None
    return max(eligible, key=lambda c: c.get("score", 0.0))


def _build_v2_advisory(
    *,
    score: float,
    reasons: list[str],
    block_reasons: list[str],
    cautions: list[str],
    mode: str,
    bullish_axes: dict,
    bearish_axes: dict,
    min_axes_required: int | None,
    sr_snapshot: SRSnapshot,
    trendline_ctx: TrendlineContext,
    chart_pattern: ChartPatternSnapshot,
    lower_tf: LowerTimeframeTrigger,
    macro: MacroAlignmentSnapshot,
    stop_plan: StopPlan | None,
    setup_candidates: list[dict] | None = None,
    best_setup: dict | None = None,
    reconstruction_quality: dict | None = None,
    multi_scale_chart: dict | None = None,
) -> dict:
    cfg = get_royal_road_mode_config(mode)
    return {
        "profile": PROFILE_NAME_V2,
        "mode": mode,
        "mode_status": cfg.status,
        "mode_needs_validation": cfg.needs_validation,
        "score": float(score),
        "reasons": list(reasons),
        "block_reasons": list(block_reasons),
        "cautions": list(cautions),
        "evidence_axes": {
            "bullish": dict(bullish_axes),
            "bearish": dict(bearish_axes),
        },
        "evidence_axes_count": {
            "bullish": sum(1 for v in bullish_axes.values() if v),
            "bearish": sum(1 for v in bearish_axes.values() if v),
        },
        "min_evidence_axes_required": min_axes_required,
        "support_resistance_v2": sr_snapshot.to_dict(),
        "trendline_context": trendline_ctx.to_dict(),
        "chart_pattern_v2": chart_pattern.to_dict(),
        "lower_tf_trigger": lower_tf.to_dict(),
        "macro_alignment": macro.to_dict(),
        "structure_stop_plan": (
            stop_plan.to_dict() if stop_plan is not None else None
        ),
        "setup_candidates": list(setup_candidates or []),
        "best_setup": best_setup,
        "reconstruction_quality": dict(reconstruction_quality or {}),
        "multi_scale_chart": dict(multi_scale_chart or {}),
        "source": "technical_confluence_v1+v2_modules",
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decide_royal_road_v2(
    *,
    df_window: pd.DataFrame,
    technical_confluence: dict,
    pattern: PatternResult | None,
    higher_timeframe_trend: str | None,
    risk_reward: float | None,
    risk_state: RiskState,
    atr_value: float,
    last_close: float,
    symbol: str,
    macro_context: dict | None,
    df_lower_tf: pd.DataFrame | None,
    lower_tf_interval: str | None,
    stop_mode: str = DEFAULT_STOP_MODE,
    stop_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
    base_bar_close_ts: pd.Timestamp,
    mode: str = DEFAULT_ROYAL_ROAD_MODE,
    min_risk_reward: float = MIN_RISK_REWARD,
) -> Decision:
    """Produce a v2 royal-road Decision.

    The function computes all v2 sub-snapshots (S/R, trendlines, chart
    patterns, lower-TF trigger, macro alignment, stop plan) and rolls
    them into evidence axes. Then it applies the same mode-aware rule
    chain as v1, plus v2-specific gates:

      - macro_strong_against → block
      - opposite-side strong level near current close → block
      - structure_stop required when stop_mode != "atr" and the chosen
        plan is invalid → block
    """
    cfg = get_royal_road_mode_config(mode)
    chain: list[str] = ["risk_gate", f"v2_mode:{cfg.name}"]
    reasons: list[str] = []
    block_reasons: list[str] = []
    cautions: list[str] = []

    sr_snapshot = detect_levels(
        df_window, atr_value=atr_value, last_close=last_close,
    )
    trendline_ctx = detect_trendlines(
        df_window, atr_value=atr_value, last_close=last_close,
    )
    chart_pattern = detect_patterns(
        df_window, atr_value=atr_value, last_close=last_close,
    )
    lower_tf = (
        detect_lower_tf_trigger(
            df_lower_tf=df_lower_tf,
            lower_tf_interval=lower_tf_interval,
            base_bar_close_ts=base_bar_close_ts,
        )
        if df_lower_tf is not None
        else empty_trigger(lower_tf_interval, "not_attached")
    )
    macro = compute_macro_alignment(
        symbol=symbol, macro_context=macro_context,
    )

    bullish_axes = _bullish_axes_v2(
        tc=technical_confluence or {}, sr=sr_snapshot, tl=trendline_ctx,
        cp=chart_pattern, ltf=lower_tf, macro=macro,
    )
    bearish_axes = _bearish_axes_v2(
        tc=technical_confluence or {}, sr=sr_snapshot, tl=trendline_ctx,
        cp=chart_pattern, ltf=lower_tf, macro=macro,
    )

    # Always compute reconstruction quality + multi-scale (cheap),
    # so HOLD paths can also surface them in the trace.
    _early_stop_plan = plan_stop(
        mode=stop_mode, side="BUY", entry=last_close, atr=atr_value,
        stop_atr_mult=stop_atr_mult, tp_atr_mult=tp_atr_mult,
        structure_stop_price=(
            (technical_confluence or {}).get("risk_plan_obs", {}).get(
                "structure_stop_price"
            )
        ),
    )
    early_reconstruction = _reconstruction_quality(
        sr=sr_snapshot, tl=trendline_ctx, cp=chart_pattern,
        ltf=lower_tf, macro=macro, stop_plan=_early_stop_plan,
    )
    early_multi_scale = reconstruct_chart_multi_scale(
        df_window, atr_value=atr_value, last_close=last_close,
    )

    # Risk gate (always)
    gate = evaluate_gate(risk_state)
    if not gate.allow_trade:
        codes = tuple(gate.blocked_codes)
        return _hold(
            reason="risk gate blocked",
            blocked_by=codes,
            chain=tuple(chain),
            confidence=0.0,
            advisory=_build_v2_advisory(
                score=0.0, reasons=[], cautions=[],
                block_reasons=[f"gate:{c}" for c in codes],
                mode=cfg.name, bullish_axes=bullish_axes,
                bearish_axes=bearish_axes, min_axes_required=None,
                sr_snapshot=sr_snapshot, trendline_ctx=trendline_ctx,
                chart_pattern=chart_pattern, lower_tf=lower_tf,
                macro=macro, stop_plan=None,
                reconstruction_quality=early_reconstruction,
                multi_scale_chart=early_multi_scale,
            ),
        )

    final = (technical_confluence or {}).get("final_confluence") or {}
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
            advisory=_build_v2_advisory(
                score=score, reasons=[], cautions=[],
                block_reasons=block_reasons, mode=cfg.name,
                bullish_axes=bullish_axes, bearish_axes=bearish_axes,
                min_axes_required=None, sr_snapshot=sr_snapshot,
                trendline_ctx=trendline_ctx, chart_pattern=chart_pattern,
                lower_tf=lower_tf, macro=macro, stop_plan=None,
                reconstruction_quality=early_reconstruction,
                multi_scale_chart=early_multi_scale,
            ),
        )

    is_weak = label.startswith("WEAK_")
    if is_weak and not cfg.allow_weak_entries:
        block_reasons.append(f"weak_setup:{label}")

    risk_obs = (technical_confluence or {}).get("risk_plan_obs") or {}
    invalidation_clear = bool(risk_obs.get("invalidation_clear"))
    structure_stop_price = risk_obs.get("structure_stop_price")
    structure_stop_exists = structure_stop_price is not None

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

    # v2 macro gate: VERY_HIGH VIX against the trade direction blocks.
    if (
        macro.macro_strong_against != "UNKNOWN"
        and macro.macro_strong_against == side
    ):
        block_reasons.extend(macro.macro_block_reasons)
    elif macro.macro_score <= _MACRO_BLOCK_THRESHOLD and side == "BUY":
        block_reasons.append("macro_strongly_against_buy")
    elif macro.macro_score >= -_MACRO_BLOCK_THRESHOLD and side == "SELL":
        block_reasons.append("macro_strongly_against_sell")

    # SR opposite-strong level block
    if cfg.block_opposite_level:
        if side == "BUY" and sr_snapshot.near_strong_resistance:
            block_reasons.append("near_strong_resistance_for_buy")
        if side == "SELL" and sr_snapshot.near_strong_support:
            block_reasons.append("near_strong_support_for_sell")

    # HTF counter-trend
    if cfg.htf_counter_trend_blocks:
        htf = (higher_timeframe_trend or "UNKNOWN").upper()
        if side == "BUY" and htf == "DOWNTREND":
            block_reasons.append("htf_counter_trend_for_buy")
        if side == "SELL" and htf == "UPTREND":
            block_reasons.append("htf_counter_trend_for_sell")

    # Fake breakout block
    if sr_snapshot.fake_breakout:
        block_reasons.append("sr_fake_breakout")

    # Evidence axes count gate
    axes_for_side = bullish_axes if side == "BUY" else bearish_axes
    n_axes = sum(1 for v in axes_for_side.values() if v)
    min_axes = cfg.min_evidence_axes_weak if is_weak else cfg.min_evidence_axes_strong
    if n_axes < min_axes:
        block_reasons.append(
            f"insufficient_{side.lower()}_evidence_axes_v2:{n_axes}<{min_axes}"
        )

    # Structure stop plan (only used by entry — but we surface it here
    # for the trace; the engine separately consults plan_stop when
    # stop_mode != atr to actually place the order).
    stop_plan = plan_stop(
        mode=stop_mode, side=side, entry=last_close, atr=atr_value,
        stop_atr_mult=stop_atr_mult, tp_atr_mult=tp_atr_mult,
        structure_stop_price=structure_stop_price,
    )
    if stop_mode != "atr" and stop_plan.stop_price is None:
        block_reasons.append(
            f"stop_plan_invalid:{stop_plan.invalidation_reason}"
        )

    # Reconstruction quality + multi-scale snapshot (always built, even
    # on HOLD bars, so the trace can correlate quality with outcome).
    reconstruction = _reconstruction_quality(
        sr=sr_snapshot, tl=trendline_ctx, cp=chart_pattern,
        ltf=lower_tf, macro=macro, stop_plan=stop_plan,
    )
    multi_scale = reconstruct_chart_multi_scale(
        df_window, atr_value=atr_value, last_close=last_close,
    )
    # If reconstruction quality is below the threshold, block the entry
    # even when label / axes / SR / macro / stop all agree.
    if reconstruction["total_reconstruction_score"] < _MIN_RECONSTRUCTION_SCORE:
        block_reasons.append("insufficient_royal_road_reconstruction_quality")

    # Build setup candidate(s). v2.2 emits at most one candidate per
    # bar (the directional one). Future v2.3 could enumerate
    # alternatives (different stop modes, different evidence subsets).
    setup_candidates = _build_setup_candidates(
        side=side,
        bullish_axes=bullish_axes, bearish_axes=bearish_axes,
        sr=sr_snapshot, tl=trendline_ctx, cp=chart_pattern,
        ltf=lower_tf, macro=macro, stop_plan=stop_plan,
        label=label, score=score,
        reasons_template=[],
        block_reasons_template=block_reasons,
        rr=stop_plan.rr_realized,
    )
    best_setup = _select_best_setup(setup_candidates)

    if block_reasons:
        return _hold(
            reason="; ".join(block_reasons),
            blocked_by=tuple(block_reasons),
            chain=tuple(chain + [
                "macro_check", "sr_check", "evidence_v2_check",
                "stop_plan_check", "reconstruction_quality_check",
            ]),
            confidence=0.0,
            advisory=_build_v2_advisory(
                score=score, reasons=reasons, block_reasons=block_reasons,
                cautions=cautions, mode=cfg.name,
                bullish_axes=bullish_axes, bearish_axes=bearish_axes,
                min_axes_required=min_axes, sr_snapshot=sr_snapshot,
                trendline_ctx=trendline_ctx, chart_pattern=chart_pattern,
                lower_tf=lower_tf, macro=macro, stop_plan=stop_plan,
                setup_candidates=setup_candidates,
                best_setup=best_setup,
                reconstruction_quality=reconstruction,
                multi_scale_chart=multi_scale,
            ),
        )

    # Build positive reasons
    if side == "BUY":
        reasons.extend(list(final.get("bullish_reasons") or []))
    else:
        reasons.extend(list(final.get("bearish_reasons") or []))
    reasons.extend(
        f"axis:{name}" for name, hit in axes_for_side.items() if hit
    )
    reasons.extend(macro.macro_reasons)
    reasons.extend(f"caution:{c}" for c in cautions)
    reasons.append(f"{label}_confirmed_by_v2_{cfg.name}")

    return Decision(
        action=side,
        confidence=_confidence_from_score(score),
        reason=f"royal_road_decision_v2[{cfg.name}]: {side}",
        blocked_by=(),
        rule_chain=tuple(chain + [
            "macro_check", "sr_check", "evidence_v2_check",
            "stop_plan_check", "reconstruction_quality_check",
        ]),
        advisory=_build_v2_advisory(
            score=score, reasons=reasons, block_reasons=[],
            cautions=cautions, mode=cfg.name,
            bullish_axes=bullish_axes, bearish_axes=bearish_axes,
            min_axes_required=min_axes, sr_snapshot=sr_snapshot,
            trendline_ctx=trendline_ctx, chart_pattern=chart_pattern,
            lower_tf=lower_tf, macro=macro, stop_plan=stop_plan,
            setup_candidates=setup_candidates,
            best_setup=best_setup,
            reconstruction_quality=reconstruction,
            multi_scale_chart=multi_scale,
        ),
    )


def compare_v2_vs_current(
    *,
    decision_current: Decision,
    decision_v2: Decision,
) -> dict:
    """Same closed taxonomy as v1 but renamed for v2 vs current_runtime."""
    return compare_v1_decisions(
        decision_current=decision_current, decision_royal=decision_v2,
    )


def compare_v2_vs_v1(
    *,
    decision_v1: Decision,
    decision_v2: Decision,
) -> dict:
    """v2 vs v1 closed taxonomy (renamed): same / v1_buy_v2_hold / etc."""
    cur = decision_v1.action
    new = decision_v2.action
    if cur == new:
        diff = "same"
    elif cur == "BUY" and new == "HOLD":
        diff = "v1_buy_v2_hold"
    elif cur == "SELL" and new == "HOLD":
        diff = "v1_sell_v2_hold"
    elif cur == "HOLD" and new == "BUY":
        diff = "v1_hold_v2_buy"
    elif cur == "HOLD" and new == "SELL":
        diff = "v1_hold_v2_sell"
    elif (cur, new) in {("BUY", "SELL"), ("SELL", "BUY")}:
        diff = "opposite_direction"
    else:
        diff = "other"
    return {
        "v1_action": cur,
        "v2_action": new,
        "same_action": cur == new,
        "difference_type": diff,
    }


__all__ = [
    "PROFILE_NAME_V2",
    "decide_royal_road_v2",
    "compare_v2_vs_current",
    "compare_v2_vs_v1",
]
