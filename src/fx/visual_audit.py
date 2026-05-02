"""Visual audit for royal_road_decision_v2 chart reconstruction.

Purpose
-------
A human looking at a chart sees zones, trendlines, patterns, lower-TF
triggers, and the structure stop / target plan. v2 already produces all
of those as data; this module turns them into a per-bar audit artefact:

  - sidecar JSON  (always emitted) — full v2 payload PLUS render
    metadata (the bar window actually used, future-leak boundary,
    overlay coordinates), so a human or another process can verify
    the system "saw" the same chart they would draw.
  - PNG image     (optional; only when matplotlib is available) — OHLC
    candlestick overlay of the visible window with the same overlays.
    When matplotlib is missing, an `.image_unavailable` marker file is
    written instead so audits can detect the gap explicitly.

This module is observation-only:
  - it never reads decide_action,
  - it never modifies the trace it's auditing,
  - it never touches live / OANDA / paper paths,
  - it never writes to runs/ or libs/ unless the caller explicitly
    points it at a writable directory.

Future-leak rule
----------------
The render uses ONLY bars with `ts <= parent_bar_ts`. The sidecar
records `parent_bar_ts`, `render_window_start_ts`, `render_window_end_ts`,
and `bars_used_in_render` so consumers can verify the boundary was
respected. Pinned by `tests/test_visual_audit.py::test_future_bars_not_in_render`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable

import pandas as pd


SCHEMA_VERSION: Final[str] = "visual_audit_v1"

# Maximum number of bars to feed into the renderer (so very-long
# windows don't produce unreadable PNGs). The sidecar records the
# truncated bar count too. Pure visualization choice — does not
# influence any decision.
_MAX_RENDER_BARS: Final[int] = 600


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trace_v2_payload(trace: Any) -> dict | None:
    """Extract the royal_road_decision_v2 dict from a trace (object or
    dict). Returns None when the slice is absent."""
    sl = _safe_get(trace, "royal_road_decision_v2")
    if sl is None:
        return None
    if isinstance(sl, dict):
        return sl
    to_dict = getattr(sl, "to_dict", None)
    return to_dict() if callable(to_dict) else None


def _trace_timestamp(trace: Any) -> pd.Timestamp | None:
    ts = _safe_get(trace, "timestamp")
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return ts
    try:
        return pd.Timestamp(ts)
    except Exception:  # noqa: BLE001
        return None


def _trace_technical_confluence(trace: Any) -> dict | None:
    """Return the technical_confluence slice as a dict (or None)."""
    sl = _safe_get(trace, "technical_confluence")
    if sl is None:
        return None
    if isinstance(sl, dict):
        return sl
    to_dict = getattr(sl, "to_dict", None)
    return to_dict() if callable(to_dict) else None


def _trace_technical(trace: Any) -> dict | None:
    """Return the technical slice as a dict (or None)."""
    sl = _safe_get(trace, "technical")
    if sl is None:
        return None
    if isinstance(sl, dict):
        return sl
    to_dict = getattr(sl, "to_dict", None)
    return to_dict() if callable(to_dict) else None


def _trace_market(trace: Any) -> dict | None:
    sl = _safe_get(trace, "market")
    if sl is None:
        return None
    if isinstance(sl, dict):
        return sl
    to_dict = getattr(sl, "to_dict", None)
    return to_dict() if callable(to_dict) else None


# ---------------------------------------------------------------------------
# PDF-derived royal-road checklist panels (visual_audit_v1 extension)
# ---------------------------------------------------------------------------
#
# Each builder reads ONLY from the trace / v2 payload (or, when raw price
# values are needed, from values already on the trace). No fresh
# computation off `df` is performed inside any builder, which means the
# panels inherit the future-leak guarantees already established by
# decision_trace_build.
# ---------------------------------------------------------------------------


def _candlestick_interpretation_panel(
    *, tc: dict | None, v2: dict,
) -> dict:
    """PDF: candle_type / direction / source / context / strength / reason.

    Pure read from technical_confluence.candlestick_signal + v2 SR/pattern
    cross-reference.
    """
    sig = (tc or {}).get("candlestick_signal") or {}
    sr = v2.get("support_resistance_v2") or {}
    cp = v2.get("chart_pattern_v2") or {}
    bull = bool(
        sig.get("bullish_pinbar") or sig.get("bullish_engulfing")
        or sig.get("strong_bull_body")
    )
    bear = bool(
        sig.get("bearish_pinbar") or sig.get("bearish_engulfing")
        or sig.get("strong_bear_body")
    )
    if sig.get("bullish_pinbar"):
        candle_type = "bullish_pinbar"
    elif sig.get("bearish_pinbar"):
        candle_type = "bearish_pinbar"
    elif sig.get("bullish_engulfing"):
        candle_type = "bullish_engulfing"
    elif sig.get("bearish_engulfing"):
        candle_type = "bearish_engulfing"
    elif sig.get("strong_bull_body"):
        candle_type = "strong_bull_body"
    elif sig.get("strong_bear_body"):
        candle_type = "strong_bear_body"
    elif sig.get("harami"):
        candle_type = "harami"
    elif sig.get("rejection_wick"):
        candle_type = "rejection_wick"
    else:
        candle_type = "no_signal"
    if bull and not bear:
        direction = "BUY"
    elif bear and not bull:
        direction = "SELL"
    else:
        direction = "NEUTRAL"
    near_support = bool(sr.get("near_strong_support"))
    near_resistance = bool(sr.get("near_strong_resistance"))
    sel_patterns = cp.get("selected_patterns_top5") or []
    pattern_kinds = [p.get("kind") for p in sel_patterns if p.get("kind")]
    if direction == "BUY" and near_support:
        context = "at_strong_support"
    elif direction == "SELL" and near_resistance:
        context = "at_strong_resistance"
    elif direction == "BUY" and near_resistance:
        context = "into_strong_resistance"
    elif direction == "SELL" and near_support:
        context = "into_strong_support"
    elif candle_type == "no_signal":
        context = "no_candle_signal"
    else:
        context = "midrange"
    if candle_type in (
        "bullish_engulfing", "bearish_engulfing",
        "strong_bull_body", "strong_bear_body",
    ):
        strength = "strong"
    elif candle_type in ("bullish_pinbar", "bearish_pinbar"):
        strength = "medium" if context.startswith("at_") else "weak"
    elif candle_type == "no_signal":
        strength = "none"
    else:
        strength = "weak"
    reason_parts: list[str] = [f"candle={candle_type}"]
    if context != "midrange":
        reason_parts.append(context)
    if pattern_kinds:
        reason_parts.append(f"patterns={pattern_kinds}")
    return {
        "available": tc is not None,
        "candle_type": candle_type,
        "direction": direction,
        "source": "technical_confluence_v1.candlestick_signal",
        "context": context,
        "strength": strength,
        "near_strong_support": near_support,
        "near_strong_resistance": near_resistance,
        "supporting_pattern_kinds": pattern_kinds,
        "reason": "; ".join(reason_parts),
    }


def _rsi_context_panel(*, tc: dict | None, technical: dict | None) -> dict:
    """PDF: rsi_value / regime / overbought / oversold /
    rsi_signal / rsi_signal_valid / rsi_trap_reason.

    The "trap" wording from the PDF maps to rsi_trend_danger: in a
    strong trend, RSI extremes are usually NOT reversal signals.
    """
    ind = (tc or {}).get("indicator_context") or {}
    rsi_value = ind.get("rsi_value")
    if rsi_value is None and technical is not None:
        rsi_value = technical.get("rsi_14")
    rsi_range_valid = ind.get("rsi_range_valid")
    rsi_trend_danger = ind.get("rsi_trend_danger")
    overbought = bool(rsi_value is not None and rsi_value > 70)
    oversold = bool(rsi_value is not None and rsi_value < 30)
    if rsi_value is None:
        regime = "unknown"
    elif rsi_range_valid:
        regime = "range"
    elif rsi_trend_danger:
        regime = "trend_extreme"
    else:
        regime = "trend"
    if overbought:
        rsi_signal = "SELL"
    elif oversold:
        rsi_signal = "BUY"
    else:
        rsi_signal = "NEUTRAL"
    # Per PDF: in a trend the OB/OS read is a TRAP, not a signal.
    rsi_signal_valid = bool(
        rsi_signal != "NEUTRAL" and (rsi_range_valid or False)
        and not rsi_trend_danger
    )
    if rsi_trend_danger and rsi_signal != "NEUTRAL":
        rsi_trap_reason = (
            "rsi_extreme_in_trend_is_continuation_not_reversal"
        )
    elif rsi_signal != "NEUTRAL" and not rsi_range_valid:
        rsi_trap_reason = "rsi_extreme_outside_range_regime"
    else:
        rsi_trap_reason = None
    return {
        "available": tc is not None or technical is not None,
        "rsi_value": rsi_value,
        "regime": regime,
        "overbought": overbought,
        "oversold": oversold,
        "rsi_signal": rsi_signal,
        "rsi_signal_valid": rsi_signal_valid,
        "rsi_trap_reason": rsi_trap_reason,
        "source": "technical_confluence_v1.indicator_context",
    }


def _ma_granville_context_panel(
    *, tc: dict | None, technical: dict | None, market: dict | None,
) -> dict:
    """PDF: ma_periods / ma_slope / price_vs_ma / pullback_to_ma /
    bounce_from_ma / breakdown_from_ma / granville_pattern / direction /
    confidence / reason.

    Computed from already-recorded sma_20 / sma_50 / close on the trace.
    No fresh recompute.
    """
    ind = (tc or {}).get("indicator_context") or {}
    sma20 = sma50 = close = None
    if technical is not None:
        sma20 = technical.get("sma_20")
        sma50 = technical.get("sma_50")
    if market is not None:
        close = market.get("close")
    have_data = (
        sma20 is not None and sma50 is not None and close is not None
    )
    if not have_data:
        return {
            "available": False,
            "unavailable_reason": "missing sma_20 / sma_50 / close on trace",
            "ma_periods": [20, 50],
            "source": "trace.technical + trace.market",
        }
    if sma20 > sma50:
        ma_slope = "rising"
    elif sma20 < sma50:
        ma_slope = "falling"
    else:
        ma_slope = "flat"
    if close > max(sma20, sma50):
        price_vs_ma = "above_both"
    elif close < min(sma20, sma50):
        price_vs_ma = "below_both"
    else:
        price_vs_ma = "between_mas"
    # Granville-ish proximity heuristics (no recompute — pure
    # comparisons of the bar's own close vs MA values already on trace)
    near_sma20 = bool(abs(close - sma20) <= 0.001 * abs(sma20))
    near_sma50 = bool(abs(close - sma50) <= 0.001 * abs(sma50))
    pullback_to_ma = bool(
        ma_slope == "rising" and price_vs_ma in ("between_mas", "above_both")
        and (near_sma20 or near_sma50)
    )
    bounce_from_ma = bool(
        ma_slope == "rising" and price_vs_ma == "above_both"
        and (near_sma20 or near_sma50)
    )
    breakdown_from_ma = bool(
        ma_slope == "rising" and price_vs_ma == "below_both"
    )
    if ma_slope == "rising" and price_vs_ma == "above_both" and pullback_to_ma:
        granville_pattern = "trend_pullback_buy"
        direction = "BUY"
    elif ma_slope == "rising" and breakdown_from_ma:
        granville_pattern = "trend_failure_sell"
        direction = "SELL"
    elif ma_slope == "falling" and price_vs_ma == "below_both" and (near_sma20 or near_sma50):
        granville_pattern = "trend_pullback_sell"
        direction = "SELL"
    elif ma_slope == "falling" and price_vs_ma == "above_both":
        granville_pattern = "trend_failure_buy"
        direction = "BUY"
    elif ma_slope == "rising":
        granville_pattern = "trend_continuation_buy"
        direction = "BUY"
    elif ma_slope == "falling":
        granville_pattern = "trend_continuation_sell"
        direction = "SELL"
    else:
        granville_pattern = "no_pattern"
        direction = "NEUTRAL"
    # Confidence: stronger when slope agrees with price_vs_ma.
    if (ma_slope == "rising" and price_vs_ma == "above_both") or (
        ma_slope == "falling" and price_vs_ma == "below_both"
    ):
        confidence = 0.7
    elif granville_pattern.startswith("trend_failure"):
        confidence = 0.55
    else:
        confidence = 0.4
    reason = (
        f"ma20={sma20:.5f} ma50={sma50:.5f} close={close:.5f} "
        f"slope={ma_slope} price_vs_ma={price_vs_ma} "
        f"pattern={granville_pattern}"
    )
    ma_trend_support = ind.get("ma_trend_support")
    return {
        "available": True,
        "ma_periods": [20, 50],
        "ma_slope": ma_slope,
        "price_vs_ma": price_vs_ma,
        "pullback_to_ma": pullback_to_ma,
        "bounce_from_ma": bounce_from_ma,
        "breakdown_from_ma": breakdown_from_ma,
        "granville_pattern": granville_pattern,
        "direction": direction,
        "confidence": confidence,
        "ma_trend_support": ma_trend_support,
        "sma_20": sma20,
        "sma_50": sma50,
        "close": close,
        "source": "trace.technical + trace.technical_confluence",
        "reason": reason,
    }


def _bollinger_lifecycle_panel(
    *, tc: dict | None, technical: dict | None,
) -> dict:
    """PDF: squeeze / expansion / band_walk / breakout_after_squeeze /
    reversal_risk / bb_signal_valid / reason.

    Pure read from technical_confluence.indicator_context (squeeze /
    expansion / band_walk already computed there, future-leak safe by
    construction).
    """
    ind = (tc or {}).get("indicator_context") or {}
    bb_squeeze = ind.get("bb_squeeze")
    bb_expansion = ind.get("bb_expansion")
    bb_band_walk = ind.get("bb_band_walk")
    bb_position = (
        technical.get("bb_position") if technical is not None else None
    )
    if bb_squeeze is None and bb_expansion is None and bb_band_walk is None:
        return {
            "available": False,
            "unavailable_reason": (
                "indicator_context missing bb_* (insufficient_data?)"
            ),
            "source": "technical_confluence_v1.indicator_context",
        }
    breakout_after_squeeze = bool(bb_expansion and not bb_squeeze)
    # Reversal risk: band_walk = trend continuation, not reversal — but
    # an over-extension into a band can flag pullback risk. We keep it
    # conservative: reversal risk only when band_walk AND bb_position
    # past 0.95/0.05 (extreme).
    reversal_risk = bool(
        bb_band_walk and bb_position is not None
        and (bb_position >= 0.95 or bb_position <= 0.05)
    )
    bb_signal_valid = bool(breakout_after_squeeze or bb_band_walk)
    if bb_squeeze:
        lifecycle_phase = "squeeze"
    elif breakout_after_squeeze:
        lifecycle_phase = "post_squeeze_breakout"
    elif bb_band_walk:
        lifecycle_phase = "band_walk"
    elif bb_expansion:
        lifecycle_phase = "expansion"
    else:
        lifecycle_phase = "neutral"
    reason = (
        f"phase={lifecycle_phase} squeeze={bb_squeeze} "
        f"expansion={bb_expansion} band_walk={bb_band_walk} "
        f"bb_position={bb_position}"
    )
    return {
        "available": True,
        "lifecycle_phase": lifecycle_phase,
        "bb_squeeze": bool(bb_squeeze),
        "bb_expansion": bool(bb_expansion),
        "bb_band_walk": bool(bb_band_walk),
        "breakout_after_squeeze": breakout_after_squeeze,
        "reversal_risk": reversal_risk,
        "bb_signal_valid": bb_signal_valid,
        "bb_position": bb_position,
        "source": "technical_confluence_v1.indicator_context",
        "reason": reason,
    }


def _grand_confluence_checklist_panel(
    *, tc: dict | None, v2: dict, action: str | None,
) -> dict:
    """PDF: 12-item royal-road checklist with pass/warn/block totals.

    Items:
      1. higher_tf_alignment
      2. dow_structure
      3. support_resistance
      4. trendline
      5. chart_pattern
      6. candlestick_confirmation
      7. lower_tf_confirmation
      8. indicator_environment
      9. macro_alignment
      10. invalidation_clear
      11. rr_valid
      12. reconstruction_quality
    """
    block = list(v2.get("block_reasons") or [])
    cautions = list(v2.get("cautions") or [])
    axes_b = (v2.get("evidence_axes") or {}).get("bullish") or {}
    axes_s = (v2.get("evidence_axes") or {}).get("bearish") or {}
    sr = v2.get("support_resistance_v2") or {}
    tl = v2.get("trendline_context") or {}
    cp = v2.get("chart_pattern_v2") or {}
    ltf = v2.get("lower_tf_trigger") or {}
    macro = v2.get("macro_alignment") or {}
    sp = v2.get("structure_stop_plan") or {}
    rq = v2.get("reconstruction_quality") or {}
    risk_obs = (tc or {}).get("risk_plan_obs") or {}
    vote = (tc or {}).get("vote_breakdown") or {}

    items: list[dict] = []

    def add(item_id: str, status: str, detail: str) -> None:
        items.append({"item": item_id, "status": status, "detail": detail})

    # 1. higher_tf_alignment
    htf_aligned = vote.get("htf_alignment", "UNKNOWN")
    if any("htf_counter_trend" in r for r in block):
        add("higher_tf_alignment", "BLOCK",
            "; ".join(r for r in block if "htf_counter_trend" in r))
    elif htf_aligned in ("BUY", "SELL"):
        add("higher_tf_alignment", "PASS", f"htf_alignment={htf_aligned}")
    else:
        add("higher_tf_alignment", "WARN",
            f"htf_alignment={htf_aligned} (no explicit alignment)")
    # 2. dow_structure
    dow_b = bool(axes_b.get("bullish_structure"))
    dow_s = bool(axes_s.get("bearish_structure"))
    if dow_b or dow_s:
        add("dow_structure", "PASS",
            f"bullish={dow_b} bearish={dow_s}")
    else:
        add("dow_structure", "WARN", "no clear Dow structure code")
    # 3. support_resistance
    if any(r in (
        "near_strong_resistance_for_buy",
        "near_strong_support_for_sell",
        "sr_fake_breakout",
    ) for r in block):
        add("support_resistance", "BLOCK",
            "; ".join(r for r in block if r.startswith("near_") or r == "sr_fake_breakout"))
    elif sr.get("near_strong_support") or sr.get("near_strong_resistance"):
        add("support_resistance", "PASS",
            f"strong_support={sr.get('near_strong_support')} "
            f"strong_resistance={sr.get('near_strong_resistance')}")
    else:
        add("support_resistance", "WARN", "no strong S/R near close")
    # 4. trendline
    if tl.get("bullish_signal") or tl.get("bearish_signal"):
        add("trendline", "PASS",
            f"bullish={tl.get('bullish_signal')} bearish={tl.get('bearish_signal')}")
    else:
        add("trendline", "WARN", "no qualifying trendline signal")
    # 5. chart_pattern
    sel = cp.get("selected_patterns_top5") or []
    if cp.get("bullish_breakout_confirmed") or cp.get("bearish_breakout_confirmed"):
        add("chart_pattern", "PASS",
            f"selected={[p.get('kind') for p in sel]}")
    elif sel:
        add("chart_pattern", "WARN",
            f"selected but no breakout: {[p.get('kind') for p in sel]}")
    else:
        add("chart_pattern", "WARN", "no selected pattern")
    # 6. candlestick_confirmation
    cs = (tc or {}).get("candlestick_signal") or {}
    has_bull = bool(
        cs.get("bullish_pinbar") or cs.get("bullish_engulfing")
        or cs.get("strong_bull_body")
    )
    has_bear = bool(
        cs.get("bearish_pinbar") or cs.get("bearish_engulfing")
        or cs.get("strong_bear_body")
    )
    if has_bull or has_bear:
        add("candlestick_confirmation", "PASS",
            f"bull={has_bull} bear={has_bear}")
    else:
        add("candlestick_confirmation", "WARN", "no bull/bear candle signal")
    # 7. lower_tf_confirmation
    if not ltf.get("available"):
        add("lower_tf_confirmation", "WARN",
            f"unavailable: {ltf.get('reason')}")
    elif ltf.get("bullish_trigger") or ltf.get("bearish_trigger"):
        add("lower_tf_confirmation", "PASS",
            f"type={ltf.get('trigger_type')} "
            f"strength={ltf.get('trigger_strength')}")
    else:
        add("lower_tf_confirmation", "WARN", "no trigger")
    # 8. indicator_environment (RSI safe + BB lifecycle ok + MACD aligned)
    ind = (tc or {}).get("indicator_context") or {}
    if ind.get("rsi_trend_danger"):
        add("indicator_environment", "WARN",
            "rsi_trend_danger (extreme in trend)")
    elif ind.get("bb_squeeze") or ind.get("bb_band_walk") or (
        ind.get("macd_momentum_up") or ind.get("macd_momentum_down")
    ):
        add("indicator_environment", "PASS",
            f"rsi_safe bb_squeeze={ind.get('bb_squeeze')} "
            f"bb_walk={ind.get('bb_band_walk')} "
            f"macd_up={ind.get('macd_momentum_up')} "
            f"macd_down={ind.get('macd_momentum_down')}")
    else:
        add("indicator_environment", "WARN", "no indicator confirmation")
    # 9. macro_alignment
    if macro.get("macro_strong_against") not in (None, "UNKNOWN"):
        add("macro_alignment", "BLOCK",
            f"strong_against={macro.get('macro_strong_against')}")
    elif any(
        r in ("macro_strongly_against_buy", "macro_strongly_against_sell")
        or r.startswith("macro_") for r in block
    ):
        add("macro_alignment", "BLOCK",
            "; ".join(r for r in block if r.startswith("macro_")))
    elif macro.get("macro_score") is not None:
        add("macro_alignment", "PASS",
            f"score={macro.get('macro_score'):.2f} "
            f"bias={macro.get('currency_bias')}")
    else:
        add("macro_alignment", "WARN", "no macro context")
    # 10. invalidation_clear
    if "invalidation_unclear" in block:
        add("invalidation_clear", "BLOCK", "invalidation_unclear")
    elif risk_obs.get("invalidation_clear"):
        add("invalidation_clear", "PASS",
            f"distance_atr={risk_obs.get('structure_stop_distance_atr')}")
    else:
        add("invalidation_clear", "WARN",
            "invalidation_unclear (soft) or no structure stop")
    # 11. rr_valid
    rr = sp.get("rr_realized")
    if any(r.startswith("rr<") for r in block):
        add("rr_valid", "BLOCK",
            "; ".join(r for r in block if r.startswith("rr<")))
    elif rr is not None and rr >= 1.5:
        add("rr_valid", "PASS", f"rr_realized={rr:.2f}")
    elif rr is not None:
        add("rr_valid", "WARN", f"rr_realized={rr:.2f} (<1.5)")
    else:
        add("rr_valid", "WARN", "no rr computed")
    # 12. reconstruction_quality
    total_rq = float(rq.get("total_reconstruction_score", 0.0))
    if "insufficient_royal_road_reconstruction_quality" in block:
        add("reconstruction_quality", "BLOCK",
            f"total={total_rq:.3f} below threshold")
    elif total_rq >= 0.5:
        add("reconstruction_quality", "PASS", f"total={total_rq:.3f}")
    else:
        add("reconstruction_quality", "WARN",
            f"total={total_rq:.3f} (borderline)")

    n_pass = sum(1 for it in items if it["status"] == "PASS")
    n_warn = sum(1 for it in items if it["status"] == "WARN")
    n_block = sum(1 for it in items if it["status"] == "BLOCK")
    return {
        "available": True,
        "items": items,
        "total_items": len(items),
        "total_pass": n_pass,
        "total_warn": n_warn,
        "total_block": n_block,
        "final_action": action or "HOLD",
        "block_reasons": block,
        "cautions": cautions,
        "source": "trace.royal_road_decision_v2 + technical_confluence_v1",
    }


def _invalidation_explanation_panel(*, v2: dict) -> dict:
    """PDF: selected_stop_price / atr_stop_price / structure_stop_price /
    invalidation_structure / invalidation_level / invalidation_status /
    why_this_stop_invalidates_the_setup / rr_selected / reason."""
    sp = v2.get("structure_stop_plan")
    sr = v2.get("support_resistance_v2") or {}
    cp = v2.get("chart_pattern_v2") or {}
    if not sp:
        return {
            "available": False,
            "unavailable_reason": "no structure_stop_plan attached",
            "source": "trace.royal_road_decision_v2.structure_stop_plan",
        }
    chosen_mode = sp.get("chosen_mode")
    outcome = sp.get("outcome")
    selected_stop = sp.get("stop_price")
    atr_stop = sp.get("atr_stop_price")
    structure_stop = sp.get("structure_stop_price")
    rr_selected = sp.get("rr_realized")
    # Identify the structural anchor (which level / pattern the stop
    # leans on).
    invalidation_structure = "atr_only"
    invalidation_level = None
    if structure_stop is not None and chosen_mode != "atr":
        # Match against selected zones by closest price.
        for lvl in sr.get("selected_level_zones_top5", []):
            zlow = lvl.get("zone_low")
            zhigh = lvl.get("zone_high")
            if zlow is not None and zhigh is not None and (
                zlow <= structure_stop <= zhigh
            ):
                invalidation_structure = f"sr_zone:{lvl.get('kind')}"
                invalidation_level = {
                    "zone_low": zlow, "zone_high": zhigh,
                    "kind": lvl.get("kind"),
                    "touch_count": lvl.get("touch_count"),
                    "confidence": lvl.get("confidence"),
                }
                break
        if invalidation_level is None:
            for p in cp.get("selected_patterns_top5", []):
                inv = p.get("invalidation_price")
                if inv is not None and abs(inv - structure_stop) < 1e-6:
                    invalidation_structure = f"pattern:{p.get('kind')}"
                    invalidation_level = {
                        "pattern_kind": p.get("kind"),
                        "neckline": p.get("neckline"),
                        "invalidation_price": inv,
                    }
                    break
        if invalidation_structure == "atr_only":
            invalidation_structure = "structure_no_anchor_match"
    if outcome in ("invalid_too_close", "invalid_no_structure"):
        invalidation_status = "invalid"
    elif outcome == "hybrid_atr_fallback":
        invalidation_status = "fallback_atr"
    elif selected_stop is None:
        invalidation_status = "missing"
    else:
        invalidation_status = "valid"
    if invalidation_status == "invalid":
        why = (
            f"stop plan rejected (outcome={outcome}); "
            "the structure stop violated distance / availability rules"
        )
    elif invalidation_status == "fallback_atr":
        why = (
            "no clean structure anchor → fell back to ATR-based stop; "
            "the setup invalidates only at the ATR distance, not at a "
            "specific market structure"
        )
    elif invalidation_structure.startswith("sr_zone"):
        why = (
            f"a close beyond {selected_stop} would breach the {invalidation_structure} "
            f"zone the setup is built on, removing the structural reason for the trade"
        )
    elif invalidation_structure.startswith("pattern"):
        why = (
            f"a close beyond {selected_stop} invalidates the "
            f"{invalidation_structure}, so the breakout / setup no longer holds"
        )
    elif chosen_mode == "atr":
        why = (
            f"ATR-distance stop at {selected_stop}; the trade "
            "invalidates when price moves N×ATR against entry"
        )
    else:
        why = "no clear invalidation path documented for this stop plan"
    reason = (
        f"mode={chosen_mode} outcome={outcome} stop={selected_stop} "
        f"structure={invalidation_structure} status={invalidation_status}"
    )
    return {
        "available": True,
        "selected_stop_price": selected_stop,
        "atr_stop_price": atr_stop,
        "structure_stop_price": structure_stop,
        "chosen_mode": chosen_mode,
        "outcome": outcome,
        "invalidation_structure": invalidation_structure,
        "invalidation_level": invalidation_level,
        "invalidation_status": invalidation_status,
        "why_this_stop_invalidates_the_setup": why,
        "rr_selected": rr_selected,
        "source": "trace.royal_road_decision_v2.structure_stop_plan",
        "reason": reason,
    }


def _current_runtime_indicator_votes_panel(*, tc: dict | None) -> dict:
    """PDF: rsi_vote / macd_vote / sma_vote / bb_vote / voted_action /
    vote_count_buy / vote_count_sell / warning. Pure read of
    technical_confluence.vote_breakdown.
    """
    if tc is None:
        return {
            "available": False,
            "unavailable_reason": "technical_confluence slice absent on trace",
            "source": "trace.technical_confluence.vote_breakdown",
        }
    vote = tc.get("vote_breakdown") or {}
    if not vote:
        return {
            "available": False,
            "unavailable_reason": "vote_breakdown not populated",
            "source": "trace.technical_confluence.vote_breakdown",
        }
    ind = tc.get("indicator_context") or {}
    rsi_value = ind.get("rsi_value")
    if rsi_value is None:
        rsi_vote = "NEUTRAL"
    elif rsi_value < 30:
        rsi_vote = "BUY"
    elif rsi_value > 70:
        rsi_vote = "SELL"
    else:
        rsi_vote = "NEUTRAL"
    macd_up = ind.get("macd_momentum_up")
    macd_down = ind.get("macd_momentum_down")
    if macd_up:
        macd_vote = "BUY"
    elif macd_down:
        macd_vote = "SELL"
    else:
        macd_vote = "NEUTRAL"
    ma_trend = ind.get("ma_trend_support")
    if ma_trend in ("BUY", "SELL"):
        sma_vote = ma_trend
    else:
        sma_vote = "NEUTRAL"
    # bb_vote: derived from buy/sell vote counts minus the other 3
    # indicators we already classified. Without bb_position we cannot
    # tell precisely; mark as UNKNOWN if not derivable.
    n_buy = int(vote.get("indicator_buy_votes") or 0)
    n_sell = int(vote.get("indicator_sell_votes") or 0)
    bb_vote = "UNKNOWN"
    bull_count_others = sum(1 for v in (rsi_vote, macd_vote, sma_vote) if v == "BUY")
    bear_count_others = sum(1 for v in (rsi_vote, macd_vote, sma_vote) if v == "SELL")
    if n_buy - bull_count_others == 1 and n_sell - bear_count_others == 0:
        bb_vote = "BUY"
    elif n_sell - bear_count_others == 1 and n_buy - bull_count_others == 0:
        bb_vote = "SELL"
    elif n_buy - bull_count_others == 0 and n_sell - bear_count_others == 0:
        bb_vote = "NEUTRAL"
    voted_action = vote.get("voted_action") or "HOLD"
    warnings: list[str] = []
    if rsi_vote != "NEUTRAL" and not ind.get("rsi_range_valid", False):
        warnings.append("rsi_extreme_outside_range_regime")
    if ind.get("rsi_trend_danger"):
        warnings.append("rsi_trend_danger")
    return {
        "available": True,
        "rsi_vote": rsi_vote,
        "macd_vote": macd_vote,
        "sma_vote": sma_vote,
        "bb_vote": bb_vote,
        "voted_action": voted_action,
        "vote_count_buy": n_buy,
        "vote_count_sell": n_sell,
        "warning": "; ".join(warnings) if warnings else None,
        "source": "trace.technical_confluence.vote_breakdown",
    }


def _build_entry_summary(
    *,
    action: str | None,
    last_close: float | None,
    stop_plan: dict | None,
) -> dict:
    """Build a compact entry / stop / tp / RR summary for the audit.

    For BUY / SELL: entry_price falls back to the visible window's last
    close (parent_bar close) since the v2 payload does not store an
    explicit entry price. The fallback is signalled via
    `entry_price_source`.

    For HOLD / unknown actions: entry_price is None, RR is None, and
    `entry_price_source` is "no_entry_hold".

    RR is computed only when entry / stop / take_profit are all
    available and stop is on the correct side of entry.
    """
    sp = stop_plan or {}
    if action not in ("BUY", "SELL"):
        return {
            "action": action,
            "entry_price": None,
            "entry_price_source": "no_entry_hold",
            "stop_price": sp.get("stop_price"),
            "structure_stop_price": sp.get("structure_stop_price"),
            "atr_stop_price": sp.get("atr_stop_price"),
            "take_profit_price": sp.get("take_profit_price"),
            "risk": None,
            "reward": None,
            "rr": None,
            "rr_unavailable_reason": "no_entry_hold",
        }
    if last_close is None:
        return {
            "action": action,
            "entry_price": None,
            "entry_price_source": "parent_bar_close_unavailable",
            "stop_price": sp.get("stop_price"),
            "structure_stop_price": sp.get("structure_stop_price"),
            "atr_stop_price": sp.get("atr_stop_price"),
            "take_profit_price": sp.get("take_profit_price"),
            "risk": None, "reward": None, "rr": None,
            "rr_unavailable_reason": "missing_entry_price",
        }
    entry = float(last_close)
    stop = sp.get("stop_price")
    tp = sp.get("take_profit_price")
    risk: float | None = None
    reward: float | None = None
    rr: float | None = None
    rr_reason: str | None = None
    if stop is None or tp is None:
        rr_reason = "missing_stop_or_tp"
    else:
        if action == "BUY":
            risk_val = entry - float(stop)
            reward_val = float(tp) - entry
            if risk_val <= 0 or reward_val <= 0:
                rr_reason = "stop_or_tp_on_wrong_side"
            else:
                risk, reward = risk_val, reward_val
                rr = reward_val / risk_val
        else:  # SELL
            risk_val = float(stop) - entry
            reward_val = entry - float(tp)
            if risk_val <= 0 or reward_val <= 0:
                rr_reason = "stop_or_tp_on_wrong_side"
            else:
                risk, reward = risk_val, reward_val
                rr = reward_val / risk_val
    return {
        "action": action,
        "entry_price": entry,
        "entry_price_source": "parent_bar_close_fallback",
        "stop_price": sp.get("stop_price"),
        "structure_stop_price": sp.get("structure_stop_price"),
        "atr_stop_price": sp.get("atr_stop_price"),
        "take_profit_price": sp.get("take_profit_price"),
        "risk": risk,
        "reward": reward,
        "rr": rr,
        "rr_unavailable_reason": rr_reason,
    }


def _build_checklist_panels(*, trace: Any, v2: dict) -> dict:
    """Aggregate the 7 PDF-derived royal-road checklist panels."""
    tc = _trace_technical_confluence(trace)
    technical = _trace_technical(trace)
    market = _trace_market(trace)
    action = v2.get("action")
    return {
        "candlestick_interpretation": _candlestick_interpretation_panel(
            tc=tc, v2=v2,
        ),
        "rsi_context": _rsi_context_panel(tc=tc, technical=technical),
        "ma_granville_context": _ma_granville_context_panel(
            tc=tc, technical=technical, market=market,
        ),
        "bollinger_lifecycle": _bollinger_lifecycle_panel(
            tc=tc, technical=technical,
        ),
        "grand_confluence_checklist": _grand_confluence_checklist_panel(
            tc=tc, v2=v2, action=action,
        ),
        "invalidation_explanation": _invalidation_explanation_panel(v2=v2),
        "current_runtime_indicator_votes": (
            _current_runtime_indicator_votes_panel(tc=tc)
        ),
    }


def build_visual_audit_payload(
    *,
    trace: Any,
    df: pd.DataFrame,
) -> dict | None:
    """Build the audit sidecar JSON for one trace.

    Returns None when the trace has no `royal_road_decision_v2` slice
    (i.e. the default current_runtime profile or v1-only profile).
    Otherwise returns a dict matching `visual_audit_v1` schema.
    """
    v2 = _trace_v2_payload(trace)
    if v2 is None:
        return None
    parent_ts = _trace_timestamp(trace)
    if parent_ts is None:
        # Fall back to df's last index — still future-leak safe because
        # we never look past it.
        parent_ts = pd.Timestamp(df.index[-1]) if len(df) > 0 else None
    visible_full = (
        df[df.index <= parent_ts] if (parent_ts is not None and len(df) > 0)
        else df.iloc[0:0]
    )
    visible = visible_full.tail(_MAX_RENDER_BARS)
    n_visible = int(len(visible))
    render_start_ts = (
        visible.index[0].isoformat() if n_visible > 0 else None
    )
    render_end_ts = (
        visible.index[-1].isoformat() if n_visible > 0 else None
    )
    parent_iso = (
        parent_ts.isoformat() if isinstance(parent_ts, pd.Timestamp)
        else str(parent_ts) if parent_ts is not None else None
    )
    sr = v2.get("support_resistance_v2") or {}
    tl = v2.get("trendline_context") or {}
    cp = v2.get("chart_pattern_v2") or {}
    ltf = v2.get("lower_tf_trigger") or {}
    stop_plan = v2.get("structure_stop_plan") or {}
    rq = v2.get("reconstruction_quality") or {}
    best = v2.get("best_setup")
    multi_scale = v2.get("multi_scale_chart") or {}
    wave_shape_review_dict = multi_scale.get("wave_shape_review") or {}
    last_close: float | None = None
    if n_visible > 0:
        try:
            last_close = float(visible["close"].iloc[-1])
        except Exception:  # noqa: BLE001
            last_close = None
    entry_summary = _build_entry_summary(
        action=v2.get("action"),
        last_close=last_close,
        stop_plan=stop_plan,
    )
    # Build the wave-derived lines (observation-only). These are the
    # lines a human would draw given the matched pattern; they live
    # alongside the existing SR / trendline overlays so the audit
    # reviewer can compare 既存サポレジ vs 波形由来の線.
    from .wave_derived_lines import build_wave_derived_lines
    best_pattern_dict = (wave_shape_review_dict or {}).get("best_pattern") or {}
    best_skel: dict = {}
    if best_pattern_dict:
        best_scale = best_pattern_dict.get("scale")
        scales_dict = (multi_scale.get("scales") or {})
        best_skel = (scales_dict.get(best_scale) or {}).get("wave_skeleton") or {}
    atr_for_lines: float | None = None
    technical_dict = _trace_technical(trace)
    if technical_dict is not None:
        atr_for_lines = technical_dict.get("atr_14")
    if (atr_for_lines is None or atr_for_lines <= 0) and best_skel:
        atr_for_lines = best_skel.get("atr_value")
    wave_derived_lines = build_wave_derived_lines(
        best_pattern=best_pattern_dict,
        skeleton=best_skel,
        atr_value=atr_for_lines,
    )

    # Per-overlay drawing coordinates that downstream renderers can
    # consume directly. All times are ISO strings; all prices are
    # floats.
    overlays = {
        "level_zones_selected": [
            {
                "kind": lvl.get("kind"),
                "zone_low": lvl.get("zone_low"),
                "zone_high": lvl.get("zone_high"),
                "price": lvl.get("price"),
                "first_touch_ts": lvl.get("first_touch_ts"),
                "last_touch_ts": lvl.get("last_touch_ts"),
                "touch_count": lvl.get("touch_count"),
                "confidence": lvl.get("confidence"),
                "reasons": lvl.get("reasons"),
            }
            for lvl in sr.get("selected_level_zones_top5", [])
        ],
        "level_zones_rejected": [
            {
                "kind": lvl.get("kind"),
                "zone_low": lvl.get("zone_low"),
                "zone_high": lvl.get("zone_high"),
                "price": lvl.get("price"),
                "reject_reason": lvl.get("reject_reason"),
            }
            for lvl in sr.get("rejected_level_zones", [])
        ],
        "trendlines_selected": [
            {
                "kind": t.get("kind"),
                "slope": t.get("slope"),
                "intercept": t.get("intercept"),
                "anchor_indices": t.get("anchor_indices"),
                "anchor_prices": t.get("anchor_prices"),
                "touch_count": t.get("touch_count"),
                "is_strong": t.get("is_strong"),
                "broken": t.get("broken"),
                "confidence": t.get("confidence"),
            }
            for t in tl.get("selected_trendlines_top3", [])
        ],
        "trendlines_rejected": [
            {
                "kind": (rj.get("trendline") or {}).get("kind"),
                "slope": (rj.get("trendline") or {}).get("slope"),
                "anchor_indices": (rj.get("trendline") or {}).get("anchor_indices"),
                "reject_reason": rj.get("reject_reason"),
            }
            for rj in tl.get("rejected_trendlines", [])
        ],
        "patterns_selected": [
            {
                "kind": p.get("kind"),
                "neckline": p.get("neckline"),
                "neckline_broken": p.get("neckline_broken"),
                "retested": p.get("retested"),
                "side_bias": p.get("side_bias"),
                "anchor_indices": p.get("anchor_indices"),
                "upper_line": p.get("upper_line"),
                "lower_line": p.get("lower_line"),
                "invalidation_price": p.get("invalidation_price"),
                "target_price": p.get("target_price"),
                "pattern_quality_score": p.get("pattern_quality_score"),
            }
            for p in cp.get("selected_patterns_top5", [])
        ],
        "patterns_rejected": [
            {
                "kind": (rj.get("pattern") or {}).get("kind"),
                "neckline": (rj.get("pattern") or {}).get("neckline"),
                "side_bias": (rj.get("pattern") or {}).get("side_bias"),
                "reject_reason": rj.get("reject_reason"),
            }
            for rj in cp.get("rejected_patterns", [])
        ],
        "lower_tf_trigger": (
            {
                "interval": ltf.get("interval"),
                "trigger_ts": ltf.get("trigger_ts"),
                "trigger_price": ltf.get("trigger_price"),
                "trigger_type": ltf.get("trigger_type"),
                "trigger_strength": ltf.get("trigger_strength"),
                "available": ltf.get("available"),
            }
            if ltf else None
        ),
        "structure_stop_plan": (
            {
                "chosen_mode": stop_plan.get("chosen_mode"),
                "outcome": stop_plan.get("outcome"),
                "stop_price": stop_plan.get("stop_price"),
                "take_profit_price": stop_plan.get("take_profit_price"),
                "rr_realized": stop_plan.get("rr_realized"),
                "structure_stop_price": stop_plan.get("structure_stop_price"),
                "atr_stop_price": stop_plan.get("atr_stop_price"),
            }
            if stop_plan else None
        ),
    }

    title_parts: list[str] = []
    if best is not None:
        title_parts.append(f"best={best.get('side')} score={best.get('score', 0.0):.3f}")
    else:
        title_parts.append("best=NONE")
    title_parts.append(
        f"quality={rq.get('total_reconstruction_score', 0.0):.3f}"
    )
    if v2.get("action"):
        title_parts.append(f"action={v2['action']}")
    title = " | ".join(title_parts)

    checklist_panels = _build_checklist_panels(trace=trace, v2=v2)

    # Masterclass observation-only panels (16 features). Built from
    # the technical_confluence + visible df + everything already
    # computed above. Carries observation_only=True and
    # used_in_decision=False on every panel; never participates in
    # royal_road_decision_v2's final action.
    from .masterclass_aggregate import build_masterclass_panels
    technical_confluence_dict = _trace_technical_confluence(trace)
    macro_align = v2.get("macro_alignment") or {}
    trace_symbol = _trace_symbol(trace)
    masterclass_panels = build_masterclass_panels(
        visible_df=visible,
        parent_bar_ts=parent_ts,
        technical_dict=technical_dict,
        technical_confluence=technical_confluence_dict,
        overlays=overlays,
        wave_shape_review=wave_shape_review_dict,
        wave_derived_lines=wave_derived_lines,
        entry_summary=entry_summary,
        invalidation_explanation=checklist_panels.get(
            "invalidation_explanation"
        ),
        df_lower_tf=None,
        higher_tf_trend=None,
        macro_score=(
            float(macro_align.get("macro_score"))
            if macro_align.get("macro_score") is not None else None
        ),
        symbol=trace_symbol,
    )

    # Build the decision bridge from the (almost-complete) payload so
    # it can classify every block into USED / PARTIAL / AUDIT_ONLY /
    # NOT_CONNECTED / UNKNOWN. observation-only.
    from .decision_bridge import build_decision_bridge
    bridge_inputs = {
        "royal_road_decision_v2": v2,
        "wave_shape_review": wave_shape_review_dict,
        "wave_derived_lines": wave_derived_lines,
        "masterclass_panels": masterclass_panels,
        "entry_summary": entry_summary,
    }
    decision_bridge = build_decision_bridge(bridge_inputs)

    return {
        "schema_version": SCHEMA_VERSION,
        "parent_bar_ts": parent_iso,
        "render_window_start_ts": render_start_ts,
        "render_window_end_ts": render_end_ts,
        "bars_used_in_render": n_visible,
        "max_render_bars": _MAX_RENDER_BARS,
        "title": title,
        "profile": v2.get("profile"),
        "mode": v2.get("mode"),
        "action": v2.get("action"),
        "best_setup": best,
        "reconstruction_quality": rq,
        "overlays": overlays,
        "checklist_panels": checklist_panels,
        # Observation-only waveform shape review (cross-scale).
        # NOT consumed by royal_road_decision_v2; for human audit only.
        "wave_shape_review": wave_shape_review_dict,
        "entry_summary": entry_summary,
        # Wave-derived lines (WNL / WB1 / WB2 / WSL / WTP / ...).
        # Each line carries used_in_decision=False; observation-only.
        "wave_derived_lines": wave_derived_lines,
        # Masterclass observation-only audit panels (19 features).
        # NOT consumed by royal_road_decision_v2; for human audit only.
        "masterclass_panels": masterclass_panels,
        # Decision bridge: classifies everything into USED / PARTIAL /
        # AUDIT_ONLY / NOT_CONNECTED / UNKNOWN so the reader can tell
        # at a glance what actually drove the final action vs what is
        # purely observation. Observation-only.
        "decision_bridge": decision_bridge,
        # Full v2 payload preserved so the audit is self-contained
        # (no need to read the JSONL trace separately).
        "royal_road_decision_v2": v2,
    }


def _render_image_optional(
    *,
    payload: dict,
    df: pd.DataFrame,
    out_path: Path,
) -> dict:
    """Single-trace candle render with the same three-tier fallback as
    `_render_candle_image`: matplotlib_png → svg_fallback → marker.
    """
    n_render = int(payload.get("bars_used_in_render") or 0)
    end_iso = payload.get("render_window_end_ts")
    if n_render <= 0 or len(df) == 0 or end_iso is None:
        marker = out_path.with_suffix(".image_unavailable")
        marker.write_text("no_visible_bars\n")
        return {
            "image_status": "render_error:no_visible_bars",
            "marker_file": str(marker),
            "renderer": "image_unavailable",
            "path": None,
            "available": False,
        }
    end_ts = pd.Timestamp(end_iso)
    overlays = payload.get("overlays") or {}
    title = payload.get("title", "visual_audit")
    return _render_candle_image(
        df=df, end_ts=end_ts, n_bars=n_render,
        overlays=overlays, title=title, out_path=out_path,
    )


def render_visual_audit(
    *,
    trace: Any,
    df: pd.DataFrame,
    out_dir: Path | str,
    name_prefix: str = "audit",
) -> dict:
    """Render one trace into sidecar JSON (always) and a PNG (best-effort).

    Returns a dict with `status` ∈ {"v2_absent", "rendered", "json_only"},
    `json_path`, and optional `image_path` / `marker_file`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_visual_audit_payload(trace=trace, df=df)
    if payload is None:
        return {
            "status": "v2_absent",
            "reason": (
                "trace.royal_road_decision_v2 is None; default "
                "current_runtime / v1-only profile does not produce "
                "a v2 visual audit"
            ),
        }
    json_path = out_dir / f"{name_prefix}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, default=str)
    )
    image_result = _render_image_optional(
        payload=payload,
        df=df,
        out_path=out_dir / f"{name_prefix}.png",
    )
    out: dict = {
        "status": "rendered" if image_result.get("image_status") == "rendered"
        else "json_only",
        "json_path": str(json_path),
    }
    out.update(image_result)
    return out


__all__ = [
    "SCHEMA_VERSION",
    "BATCH_SCHEMA_VERSION",
    "build_visual_audit_payload",
    "render_visual_audit",
    "render_visual_audit_report",
    "render_visual_audit_mobile_single_file",
    "select_important_cases",
]


# ---------------------------------------------------------------------------
# Batch report (visual_audit_report_v1)
# ---------------------------------------------------------------------------
#
# Top-level orchestrator: generates a static HTML report directory the
# user can open in a browser to audit royal_road_v2 decisions.
#
# Layout:
#   <out_dir>/
#       index.html             - simple table over selected cases
#       cases.json             - JSON list of case summaries
#       summary.json           - aggregate stats (counts per priority)
#       assets/style.css       - inline-friendly CSS (just enough)
#       <symbol>/<safe_ts>/
#           audit.json
#           detail.html
#           feedback_template.json
#           chart_main.png  OR  chart_main.image_unavailable
#           chart_micro.png  OR  ...
#           chart_short.png
#           chart_medium.png
#           chart_long.png
#           chart_lower_tf.png  (only when lower_tf is attached)
#
# Future-leak: every chart writer truncates df to ts <= parent_bar_ts
# (delegated to existing build_visual_audit_payload).
# ---------------------------------------------------------------------------


BATCH_SCHEMA_VERSION: Final[str] = "visual_audit_report_v1"

# Per-priority counters (the "important case" picker bucketises cases)
PRIORITY_LABELS: Final[tuple[str, ...]] = (
    "v2_directional",
    "v2_vs_current_diff",
    "high_quality_hold",
    "low_quality_directional_attempt",
    "block_reason_representative",
    "stop_mode_non_atr",
    "lower_tf_trigger_present",
    "selected_pattern_present",
    "fake_breakout_block",
    "random_hold_filler",
)


def _safe_ts_for_path(ts: str) -> str:
    """Make a ts-string filesystem-safe: 2026-02-10T10:00:00+00:00
    -> 2026-02-10T10-00-00Z."""
    s = str(ts)
    s = s.replace(":", "-").replace("+00-00", "Z").replace("+00:00", "Z")
    s = s.replace("+", "p").replace(" ", "_")
    return s


def _trace_symbol(trace: Any) -> str | None:
    return _safe_get(trace, "symbol")


def select_important_cases(
    traces: Iterable[Any],
    *,
    max_cases: int = 200,
) -> list[dict]:
    """Pick a representative subset of v2 cases for visual audit.

    Per the user-supplied priority list:
      1. v2 BUY/SELL
      2. v2 vs current_runtime difference != "same"
      3. high reconstruction_quality but HOLD
      4. low reconstruction_quality but directional attempt blocked
      5. top block_reasons (one per representative reason)
      6. stop_mode != "atr"
      7. lower_tf trigger fired
      8. selected pattern present
      9. fake_breakout block
     10. random HOLD filler (per symbol)

    Returns a list of {trace, priority, rank, case_id} dicts. Each
    trace appears at most once (first matching priority wins).
    """
    seen_ids: set[str] = set()
    out: list[dict] = []
    by_priority: dict[str, list[Any]] = {p: [] for p in PRIORITY_LABELS}
    block_reason_seen: set[str] = set()

    # First pass: classify each trace.
    for tr in traces:
        v2 = _trace_v2_payload(tr)
        if v2 is None:
            continue
        sym = _trace_symbol(tr) or "UNKNOWN"
        ts = _safe_get(tr, "timestamp") or ""
        cid = f"{sym}_{_safe_ts_for_path(ts)}"
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        action = v2.get("action") or "HOLD"
        rq_total = float(
            (v2.get("reconstruction_quality") or {}).get(
                "total_reconstruction_score", 0.0
            )
        )
        block_reasons = list(v2.get("block_reasons") or [])
        cmp_v1 = v2.get("compared_to_current_runtime") or {}
        diff = cmp_v1.get("difference_type", "same")
        ltf = (v2.get("lower_tf_trigger") or {})
        sp = (v2.get("structure_stop_plan") or {})
        chart_p = (v2.get("chart_pattern_v2") or {})
        # Priority bucket selection (first match wins)
        if action in ("BUY", "SELL"):
            bucket = "v2_directional"
        elif diff != "same":
            bucket = "v2_vs_current_diff"
        elif rq_total >= 0.5:
            bucket = "high_quality_hold"
        elif any(
            "insufficient_royal_road_reconstruction_quality" in r
            for r in block_reasons
        ):
            bucket = "low_quality_directional_attempt"
        elif any(r.startswith("avoid:fake_breakout") or r == "sr_fake_breakout"
                 for r in block_reasons):
            bucket = "fake_breakout_block"
        elif sp.get("chosen_mode") and sp["chosen_mode"] != "atr":
            bucket = "stop_mode_non_atr"
        elif ltf.get("available") and (
            ltf.get("bullish_trigger") or ltf.get("bearish_trigger")
        ):
            bucket = "lower_tf_trigger_present"
        elif (chart_p.get("selected_patterns_top5") or []):
            bucket = "selected_pattern_present"
        elif block_reasons:
            # Take ONE per representative block reason so the audit
            # surfaces a wide variety, not 100 of the same kind.
            top_reason = block_reasons[0]
            if top_reason in block_reason_seen:
                bucket = "random_hold_filler"
            else:
                block_reason_seen.add(top_reason)
                bucket = "block_reason_representative"
        else:
            bucket = "random_hold_filler"
        by_priority[bucket].append({
            "trace": tr, "case_id": cid, "symbol": sym, "ts": ts,
            "v2": v2, "action": action, "rq": rq_total, "diff": diff,
            "block_reasons": block_reasons,
        })

    # Second pass: round-robin pull from each bucket so the report has
    # diversity instead of "200 v2 BUYs in a row".
    cursors = {p: 0 for p in PRIORITY_LABELS}
    while len(out) < max_cases:
        added = 0
        for p in PRIORITY_LABELS:
            if len(out) >= max_cases:
                break
            i = cursors[p]
            if i >= len(by_priority[p]):
                continue
            entry = by_priority[p][i]
            entry["priority"] = p
            entry["rank"] = i + 1
            out.append(entry)
            cursors[p] += 1
            added += 1
        if added == 0:
            break
    return out


def _build_candle_svg_xml(
    *,
    df: pd.DataFrame,
    end_ts: pd.Timestamp,
    n_bars: int,
    overlays: dict | None,
    title: str,
    wave_overlay: dict | None = None,
    wave_derived_lines: list[dict] | None = None,
) -> str | None:
    """Build the SVG candle-chart as an XML string.

    Returns None when there is nothing to draw (no visible bars within
    parent_bar_ts boundary). Future-leak safe: only bars with
    `df.index <= end_ts` are visited.

    All overlay rendering is included (selected/rejected SR zones,
    selected/rejected trendlines, pattern lines, lower_tf trigger,
    stop / take_profit / structure_stop / atr_stop, parent_bar marker).

    `wave_overlay` (optional, observation-only): when provided,
    overlays the wave skeleton polyline + pivot dots + matched-part
    labels + neckline. Expected shape:
        {
            "skeleton": {pivots: [...], scale: str, ...},
            "matched_parts": {part_name: source_df_index},
            "kind": str,            # double_bottom / head_and_shoulders / ...
            "side_bias": str,       # BUY / SELL / NEUTRAL
            "human_label": str,
        }
    Pivots whose ts falls outside the visible window are skipped.
    """
    if df is None or len(df) == 0 or n_bars <= 0:
        return None
    visible = df[df.index <= end_ts].tail(n_bars)
    if len(visible) == 0:
        return None
    # ---- coordinate system -------------------------------------
    W, H = 1200, 600
    margin_l, margin_r, margin_t, margin_b = 60, 20, 28, 28
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b
    n = len(visible)
    bar_w = max(2.0, plot_w / max(1, n))
    body_w = max(1.5, bar_w * 0.6)
    opens = visible["open"].astype(float).to_numpy()
    highs = visible["high"].astype(float).to_numpy()
    lows = visible["low"].astype(float).to_numpy()
    closes = visible["close"].astype(float).to_numpy()
    price_lo = float(min(lows.min(), opens.min(), closes.min()))
    price_hi = float(max(highs.max(), opens.max(), closes.max()))
    ovl_prices: list[float] = []
    ov = overlays or {}
    for lvl in (ov.get("level_zones_selected", [])
                + ov.get("level_zones_rejected", [])):
        for k in ("zone_low", "zone_high"):
            v = lvl.get(k)
            if v is not None:
                ovl_prices.append(float(v))
    for p in ov.get("patterns_selected", []):
        for k in ("neckline", "invalidation_price", "target_price"):
            v = p.get(k)
            if v is None or isinstance(v, dict):
                continue
            try:
                ovl_prices.append(float(v))
            except (TypeError, ValueError):
                continue
    sp = ov.get("structure_stop_plan") or {}
    for k in (
        "stop_price", "take_profit_price",
        "structure_stop_price", "atr_stop_price",
    ):
        v = sp.get(k)
        if v is not None:
            ovl_prices.append(float(v))
    ltf = ov.get("lower_tf_trigger")
    if ltf and ltf.get("trigger_price") is not None:
        ovl_prices.append(float(ltf["trigger_price"]))
    # Extend chart price range with wave-derived line prices so the
    # lines are visible (not clipped by the price_lo/price_hi clamp).
    for line in (wave_derived_lines or []):
        v = line.get("price")
        if v is None:
            continue
        try:
            ovl_prices.append(float(v))
        except (TypeError, ValueError):
            continue
    if ovl_prices:
        price_lo = min(price_lo, min(ovl_prices))
        price_hi = max(price_hi, max(ovl_prices))
    if price_hi <= price_lo:
        price_hi = price_lo + 1e-9
    pad = (price_hi - price_lo) * 0.05
    price_lo -= pad
    price_hi += pad
    price_range = price_hi - price_lo

    def x_of(i: int) -> float:
        return margin_l + (i + 0.5) * bar_w

    def y_of(p: float) -> float:
        return margin_t + (1.0 - (p - price_lo) / price_range) * plot_h

    def fmt(v: float, decimals: int = 5) -> str:
        return f"{v:.{decimals}f}"

    parts: list[str] = []
    parts.append(
        f"<?xml version='1.0' encoding='UTF-8'?>\n"
        f"<svg xmlns='http://www.w3.org/2000/svg' "
        f"viewBox='0 0 {W} {H}' width='{W}' height='{H}' "
        f"font-family='-apple-system, sans-serif' font-size='11'>"
    )
    parts.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='#fdfdfd'/>")
    parts.append(
        f"<rect x='{margin_l}' y='{margin_t}' width='{plot_w}' "
        f"height='{plot_h}' fill='white' stroke='#bbb' stroke-width='0.5'/>"
    )
    for i in range(5):
        frac = i / 4.0
        p = price_lo + (1.0 - frac) * price_range
        yp = margin_t + frac * plot_h
        parts.append(
            f"<line x1='{margin_l - 3}' y1='{yp:.1f}' x2='{margin_l}' "
            f"y2='{yp:.1f}' stroke='#888' stroke-width='0.5'/>"
        )
        parts.append(
            f"<text x='{margin_l - 6}' y='{yp + 3:.1f}' text-anchor='end' "
            f"fill='#444'>{fmt(p)}</text>"
        )
    parts.append(
        f"<text x='{margin_l}' y='18' fill='#222' font-weight='bold'>"
        f"{_html_escape(title)}</text>"
    )
    for lvl in ov.get("level_zones_selected", []):
        zlow = lvl.get("zone_low"); zhigh = lvl.get("zone_high")
        if zlow is None or zhigh is None:
            continue
        kind = lvl.get("kind")
        color = (
            "#2e7d32" if kind == "support"
            else "#c62828" if kind == "resistance"
            else "#6a1b9a"
        )
        y_top = y_of(float(zhigh))
        y_bot = y_of(float(zlow))
        parts.append(
            f"<rect class='sr-selected sr-{kind}' "
            f"x='{margin_l}' y='{min(y_top, y_bot):.1f}' "
            f"width='{plot_w}' height='{abs(y_bot - y_top):.1f}' "
            f"fill='{color}' opacity='0.18'/>"
        )
    for lvl in ov.get("level_zones_rejected", []):
        zlow = lvl.get("zone_low"); zhigh = lvl.get("zone_high")
        if zlow is None or zhigh is None:
            continue
        y_top = y_of(float(zhigh))
        y_bot = y_of(float(zlow))
        parts.append(
            f"<rect class='sr-rejected' "
            f"x='{margin_l}' y='{min(y_top, y_bot):.1f}' "
            f"width='{plot_w}' height='{abs(y_bot - y_top):.1f}' "
            f"fill='#888' opacity='0.08'/>"
        )
    for i in range(n):
        xi = x_of(i)
        y_h = y_of(highs[i]); y_l = y_of(lows[i])
        y_o = y_of(opens[i]); y_c = y_of(closes[i])
        parts.append(
            f"<line class='wick' x1='{xi:.1f}' y1='{y_h:.1f}' "
            f"x2='{xi:.1f}' y2='{y_l:.1f}' stroke='#222' "
            f"stroke-width='0.6'/>"
        )
        bull = closes[i] >= opens[i]
        color = "#2e7d32" if bull else "#c62828"
        body_top = min(y_o, y_c)
        body_h = max(0.5, abs(y_c - y_o))
        parts.append(
            f"<rect class='body candle-{'bull' if bull else 'bear'}' "
            f"x='{xi - body_w / 2:.1f}' y='{body_top:.1f}' "
            f"width='{body_w:.1f}' height='{body_h:.1f}' "
            f"fill='{color}'/>"
        )
    for t in ov.get("trendlines_selected", []):
        slope = t.get("slope"); intercept = t.get("intercept")
        anchors = t.get("anchor_indices") or []
        if slope is None or intercept is None or len(anchors) < 2:
            continue
        i0 = int(anchors[0]); i1 = int(anchors[-1])
        if i1 >= n:
            i1 = n - 1
        if i0 >= n or i1 <= i0:
            continue
        y0 = y_of(slope * i0 + intercept)
        y1 = y_of(slope * i1 + intercept)
        broken = bool(t.get("broken"))
        opacity = "0.55" if broken else "0.9"
        parts.append(
            f"<line class='trendline-selected' x1='{x_of(i0):.1f}' "
            f"y1='{y0:.1f}' x2='{x_of(i1):.1f}' y2='{y1:.1f}' "
            f"stroke='#1565c0' stroke-width='1.4' opacity='{opacity}'/>"
        )
    for rj in ov.get("trendlines_rejected", []):
        tdict = rj.get("trendline") or {}
        slope = tdict.get("slope"); intercept = tdict.get("intercept")
        anchors = tdict.get("anchor_indices") or []
        if slope is None or intercept is None or len(anchors) < 2:
            continue
        i0 = int(anchors[0]); i1 = int(anchors[-1])
        if i1 >= n:
            i1 = n - 1
        if i0 >= n or i1 <= i0:
            continue
        y0 = y_of(slope * i0 + intercept)
        y1 = y_of(slope * i1 + intercept)
        parts.append(
            f"<line class='trendline-rejected' x1='{x_of(i0):.1f}' "
            f"y1='{y0:.1f}' x2='{x_of(i1):.1f}' y2='{y1:.1f}' "
            f"stroke='#888' stroke-width='0.7' opacity='0.5' "
            f"stroke-dasharray='3,3'/>"
        )
    for p in ov.get("patterns_selected", []):
        side = p.get("side_bias")
        color = (
            "#1b5e20" if side == "BUY"
            else "#b71c1c" if side == "SELL"
            else "#ef6c00"
        )
        nl = p.get("neckline")
        if nl is not None and not isinstance(nl, dict):
            yv = y_of(float(nl))
            parts.append(
                f"<line class='pattern-neckline' "
                f"x1='{margin_l}' y1='{yv:.1f}' "
                f"x2='{margin_l + plot_w}' y2='{yv:.1f}' "
                f"stroke='{color}' stroke-width='1.0' "
                f"stroke-dasharray='6,3' opacity='0.8'/>"
            )
        for line_key in ("upper_line", "lower_line"):
            v = p.get(line_key)
            if v is None:
                continue
            if isinstance(v, dict):
                slope = v.get("slope"); intercept = v.get("intercept")
                anchors = v.get("anchors") or v.get("anchor_indices") or []
                if slope is None or intercept is None or len(anchors) < 2:
                    continue
                i0 = int(anchors[0]); i1 = int(anchors[-1])
                if i1 >= n:
                    i1 = n - 1
                if i0 >= n or i1 <= i0:
                    continue
                y0 = y_of(slope * i0 + intercept)
                y1 = y_of(slope * i1 + intercept)
                parts.append(
                    f"<line class='pattern-{line_key}' "
                    f"x1='{x_of(i0):.1f}' y1='{y0:.1f}' "
                    f"x2='{x_of(i1):.1f}' y2='{y1:.1f}' "
                    f"stroke='{color}' stroke-width='1.0' "
                    f"stroke-dasharray='6,3' opacity='0.8'/>"
                )
            else:
                try:
                    yv = y_of(float(v))
                except (TypeError, ValueError):
                    continue
                parts.append(
                    f"<line class='pattern-{line_key}' "
                    f"x1='{margin_l}' y1='{yv:.1f}' "
                    f"x2='{margin_l + plot_w}' y2='{yv:.1f}' "
                    f"stroke='{color}' stroke-width='1.0' "
                    f"stroke-dasharray='6,3' opacity='0.8'/>"
                )
    ltf = ov.get("lower_tf_trigger")
    if ltf and ltf.get("trigger_price") is not None:
        yp = y_of(float(ltf["trigger_price"]))
        cx = x_of(n - 1)
        triangle = (
            f"M {cx - 6} {yp + 6} L {cx + 6} {yp + 6} L {cx} {yp - 6} Z"
        )
        parts.append(
            f"<path class='lower-tf-trigger' d='{triangle}' "
            f"fill='#fb8c00' opacity='0.9'/>"
        )
        parts.append(
            f"<text x='{cx + 8}' y='{yp + 3:.1f}' fill='#bf360c'>"
            f"trigger:{_html_escape(str(ltf.get('trigger_type')))}</text>"
        )
    sp = ov.get("structure_stop_plan") or {}
    for key, color, label in (
        ("stop_price", "#c62828", "stop"),
        ("take_profit_price", "#2e7d32", "tp"),
        ("structure_stop_price", "#6a1b9a", "structure_stop"),
        ("atr_stop_price", "#ef6c00", "atr_stop"),
    ):
        v = sp.get(key)
        if v is None:
            continue
        yv = y_of(float(v))
        parts.append(
            f"<line class='{label.replace('_','-')}-line' "
            f"x1='{margin_l}' y1='{yv:.1f}' "
            f"x2='{margin_l + plot_w}' y2='{yv:.1f}' "
            f"stroke='{color}' stroke-width='1.2' "
            f"stroke-dasharray='4,2' opacity='0.75'/>"
        )
        parts.append(
            f"<text x='{margin_l + plot_w - 4}' y='{yv - 2:.1f}' "
            f"text-anchor='end' fill='{color}'>{label}={fmt(float(v))}</text>"
        )
    parent_x = x_of(n - 1)
    parts.append(
        f"<line class='parent-bar-marker' x1='{parent_x:.1f}' "
        f"y1='{margin_t}' x2='{parent_x:.1f}' "
        f"y2='{margin_t + plot_h}' stroke='#222' stroke-width='0.7' "
        f"stroke-dasharray='2,4' opacity='0.55'/>"
    )
    parts.append(
        f"<text x='{parent_x - 4:.1f}' y='{margin_t + plot_h - 4:.1f}' "
        f"text-anchor='end' fill='#222'>parent_bar_ts={end_ts.isoformat()}</text>"
    )
    parts.append(
        f"<text x='{margin_l}' y='{margin_t + plot_h + 16}' fill='#444'>"
        f"{visible.index[0].isoformat()}</text>"
    )
    parts.append(
        f"<text x='{margin_l + plot_w}' y='{margin_t + plot_h + 16}' "
        f"text-anchor='end' fill='#444'>{visible.index[-1].isoformat()}</text>"
    )

    # ---- wave-derived horizontal lines (observation-only) -------
    if wave_derived_lines:
        parts.append(
            _wave_derived_lines_svg_fragment(
                wave_derived_lines=wave_derived_lines,
                y_of=y_of,
                price_lo=price_lo, price_hi=price_hi,
                margin_l=margin_l, margin_t=margin_t,
                plot_w=plot_w, plot_h=plot_h,
            )
        )

    # ---- wave skeleton overlay (observation-only) ---------------
    if wave_overlay:
        parts.append(
            _wave_overlay_svg_fragment(
                wave_overlay=wave_overlay,
                visible_index=list(visible.index),
                x_of=x_of, y_of=y_of,
                price_lo=price_lo, price_hi=price_hi,
                margin_l=margin_l, margin_t=margin_t,
                plot_w=plot_w, plot_h=plot_h,
            )
        )

    parts.append("</svg>\n")
    return "".join(parts)


_PATTERN_PART_LABELS: Final[dict] = {
    # Double-bottom / double-top pieces
    "first_bottom": "B1",
    "second_bottom": "B2",
    "first_top": "P1",
    "second_top": "P2",
    "neckline_peak": "NL",
    "neckline_trough": "NL",
    # Head & shoulders pieces
    "neckline_left_peak": "NL-L",
    "neckline_right_peak": "NL-R",
    "neckline_left_trough": "NL-L",
    "neckline_right_trough": "NL-R",
    "left_shoulder": "LS",
    "right_shoulder": "RS",
    "head": "H",
    # Continuation patterns / wedges / triangles
    "breakout": "BR",
    "apex": "AP",
    "lower_anchor_1": "A1",
    "upper_anchor_1": "A1",
    "lower_anchor_2": "A2",
    "upper_anchor_2": "A2",
    "lower_anchor_3": "A3",
    "upper_anchor_3": "A3",
    "impulse_start": "I0",
    "impulse_end": "I1",
    "consolidation_high": "C-H",
    "consolidation_low": "C-L",
    "consolidation_mid": "",
    "prior_high": "",
    "prior_low": "",
}


_PATTERN_FAMILY_MARKER: Final[dict] = {
    "double_bottom": "DB1",
    "double_top": "DT1",
    "head_and_shoulders": "HS1",
    "inverse_head_and_shoulders": "IHS1",
    "bullish_flag": "FL1",
    "bearish_flag": "FL1",
    "rising_wedge": "WG1",
    "falling_wedge": "WG1",
    "ascending_triangle": "TR1",
    "descending_triangle": "TR1",
    "symmetric_triangle": "TR1",
}


def _build_wave_only_svg(
    *,
    skeleton: dict | None,
    matched_parts: dict | None,
    kind: str | None,
    human_label: str | None,
    status: str | None,
    side_bias: str | None,
    entry_summary: dict | None,
    wave_derived_lines: list[dict] | None = None,
) -> str | None:
    """Build a self-contained "波形だけ" SVG.

    Returns None when no skeleton / no pivots are available.

    The diagram:
      - polyline through pivots (no candles)
      - pivot dots
      - DB1/DT1/HS1/IHS1 + B1/B2/NL/BR... boxed labels
      - neckline horizontal line (when matched)
      - entry / stop / take_profit horizontal lines (when entry_summary
        has those values)
      - bottom-of-chart status banner (e.g. 「未ブレイク」「形成中」)
    """
    skel = skeleton or {}
    pivots = skel.get("pivots") or []
    if len(pivots) < 2:
        return None

    W, H = 800, 320
    margin_l, margin_r, margin_t, margin_b = 50, 16, 24, 36
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    prices = [float(p.get("price", 0.0)) for p in pivots]
    p_lo, p_hi = min(prices), max(prices)
    es = entry_summary or {}
    extra_prices: list[float] = []
    for k in ("entry_price", "stop_price", "structure_stop_price",
              "atr_stop_price", "take_profit_price"):
        v = es.get(k)
        if v is None:
            continue
        try:
            extra_prices.append(float(v))
        except (TypeError, ValueError):
            continue
    for line in (wave_derived_lines or []):
        v = line.get("price")
        if v is not None:
            try:
                extra_prices.append(float(v))
            except (TypeError, ValueError):
                continue
    if extra_prices:
        p_lo = min(p_lo, min(extra_prices))
        p_hi = max(p_hi, max(extra_prices))
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-9
    pad = (p_hi - p_lo) * 0.10
    p_lo -= pad
    p_hi += pad
    p_range = p_hi - p_lo

    indices = [int(p.get("index", 0)) for p in pivots]
    i_min, i_max = min(indices), max(indices)
    if i_min == i_max:
        i_max = i_min + 1
    i_span = i_max - i_min

    def x_of(idx: int) -> float:
        return margin_l + (idx - i_min) / i_span * plot_w

    def y_of(price: float) -> float:
        return margin_t + (1.0 - (price - p_lo) / p_range) * plot_h

    parts: list[str] = []
    parts.append(
        f"<svg xmlns='http://www.w3.org/2000/svg' class='wave-only-chart' "
        f"viewBox='0 0 {W} {H}' width='{W}' height='{H}' "
        f"font-family='-apple-system, sans-serif' font-size='11'>"
    )
    parts.append(
        f"<rect x='0' y='0' width='{W}' height='{H}' fill='#fbfcff'/>"
    )
    parts.append(
        f"<rect x='{margin_l}' y='{margin_t}' width='{plot_w}' "
        f"height='{plot_h}' fill='white' stroke='#bbb' stroke-width='0.5'/>"
    )

    # entry/stop/tp horizontal lines (only when present)
    line_specs = (
        ("entry_price", "#1565c0", "entry"),
        ("structure_stop_price", "#c62828", "structure_stop"),
        ("atr_stop_price", "#ef6c00", "atr_stop"),
        ("take_profit_price", "#2e7d32", "take_profit"),
    )
    for k, color, label in line_specs:
        v = es.get(k)
        if v is None:
            continue
        try:
            yv = y_of(float(v))
        except (TypeError, ValueError):
            continue
        if not (margin_t <= yv <= margin_t + plot_h):
            continue
        parts.append(
            f"<line x1='{margin_l}' y1='{yv:.1f}' "
            f"x2='{margin_l + plot_w}' y2='{yv:.1f}' "
            f"stroke='{color}' stroke-width='1.4' stroke-dasharray='4,4' "
            f"opacity='0.85'/>"
        )
        parts.append(
            f"<text x='{margin_l + plot_w - 4}' y='{yv - 3:.1f}' "
            f"text-anchor='end' fill='{color}'>"
            f"{_html_escape(label)}={float(v):.5f}</text>"
        )

    # 0. Wave-derived horizontal lines (drawn behind the skeleton)
    for line in (wave_derived_lines or []):
        price = line.get("price")
        if price is None:
            continue
        try:
            yv = y_of(float(price))
        except (TypeError, ValueError):
            continue
        if not (margin_t <= yv <= margin_t + plot_h):
            continue
        kind_str = line.get("kind", "")
        style = _WAVE_LINE_STYLE.get(kind_str, {
            "color": "#7b1fa2", "dash": "6,4", "width": 1.6,
        })
        line_id = line.get("id", "")
        parts.append(
            f"<line class='wave-derived-line wave-derived-{_html_escape(kind_str)}' "
            f"x1='{margin_l}' y1='{yv:.1f}' "
            f"x2='{margin_l + plot_w}' y2='{yv:.1f}' "
            f"stroke='{style['color']}' stroke-width='{style['width']}' "
            f"stroke-dasharray='{style['dash']}' opacity='0.85'/>"
        )
        # Right-edge label so it doesn't collide with the family marker
        label_w = max(28, 10 * len(line_id) + 8)
        parts.append(
            f"<rect class='wave-derived-line-label-bg' "
            f"x='{margin_l + plot_w - label_w - 4:.1f}' "
            f"y='{yv - 9:.1f}' width='{label_w}' height='18' "
            f"fill='white' stroke='{style['color']}' "
            f"stroke-width='1.2' rx='3'/>"
        )
        parts.append(
            f"<text class='wave-derived-line-label' "
            f"x='{margin_l + plot_w - label_w / 2 - 4:.1f}' "
            f"y='{yv + 4:.1f}' text-anchor='middle' "
            f"fill='{style['color']}' font-weight='bold' "
            f"font-size='11'>{_html_escape(line_id)}</text>"
        )

    # 1. Skeleton polyline
    poly_pts = " ".join(
        f"{x_of(int(p.get('index', 0))):.1f},{y_of(float(p.get('price', 0))):.1f}"
        for p in pivots
    )
    parts.append(
        f"<polyline class='wave-skeleton-line' points='{poly_pts}' "
        f"fill='none' stroke='#0d47a1' stroke-width='3.2' "
        f"stroke-linecap='round' stroke-linejoin='round'/>"
    )

    # 2. Pivot dots
    for p in pivots:
        x = x_of(int(p.get("index", 0)))
        y = y_of(float(p.get("price", 0.0)))
        parts.append(
            f"<circle class='wave-pivot-dot' cx='{x:.1f}' cy='{y:.1f}' "
            f"r='5' fill='#0d47a1' stroke='white' stroke-width='1.4'/>"
        )

    # 3. Matched part labels
    label_map = _PATTERN_PART_LABELS
    pivot_by_idx: dict[int, dict] = {}
    for p in pivots:
        try:
            pivot_by_idx[int(p.get("index"))] = p
        except (TypeError, ValueError):
            continue
    neckline_price: float | None = None
    neckline_label = "NL"
    parts_dict = matched_parts or {}
    for part_name, src_idx in parts_dict.items():
        if src_idx is None:
            continue
        try:
            piv = pivot_by_idx.get(int(src_idx))
        except (TypeError, ValueError):
            continue
        if piv is None:
            continue
        short = label_map.get(part_name)
        if not short:
            # Skip unmapped (or intentionally blank) parts.
            continue
        x = x_of(int(piv.get("index")))
        y = y_of(float(piv.get("price")))
        is_high = piv.get("kind") == "H"
        ty = (y - 22) if is_high else (y + 6)
        text_w = max(22, 10 * len(short) + 8)
        parts.append(
            f"<rect class='wave-part-label-bg' x='{x - text_w / 2:.1f}' "
            f"y='{ty:.1f}' width='{text_w}' height='18' fill='white' "
            f"stroke='#0d47a1' stroke-width='1.2' rx='3'/>"
        )
        parts.append(
            f"<text class='wave-part-label' x='{x:.1f}' "
            f"y='{ty + 13:.1f}' text-anchor='middle' fill='#0d47a1' "
            f"font-weight='bold' font-size='11'>{_html_escape(short)}</text>"
        )
        if "neckline" in part_name and neckline_price is None:
            try:
                neckline_price = float(piv.get("price"))
                neckline_label = short
            except (TypeError, ValueError):
                pass

    if neckline_price is not None:
        ny = y_of(neckline_price)
        parts.append(
            f"<line class='wave-neckline' x1='{margin_l}' "
            f"y1='{ny:.1f}' x2='{margin_l + plot_w}' y2='{ny:.1f}' "
            f"stroke='#7b1fa2' stroke-width='2.0' stroke-dasharray='6,4' "
            f"opacity='0.85'/>"
        )
        parts.append(
            f"<text x='{margin_l + 6}' y='{ny - 4:.1f}' fill='#7b1fa2' "
            f"font-weight='bold' font-size='11'>"
            f"{_html_escape(neckline_label)} ネックライン</text>"
        )

    # 4. Family marker (DB1 / DT1 / HS1 / IHS1 / ...)
    family = _PATTERN_FAMILY_MARKER.get(kind or "", "")
    title_str = f"{family} {human_label}".strip() if family else (
        human_label or "波形だけ表示"
    )
    parts.append(
        f"<text x='{margin_l}' y='16' fill='#0d47a1' font-weight='bold' "
        f"font-size='13'>{_html_escape(title_str)}</text>"
    )

    # 5. Status banner at bottom (e.g. 「未ブレイク (forming)」)
    status_label_ja = {
        "forming": "形成中 (未ブレイク)",
        "neckline_broken": "ネックラインブレイク済み",
        "retested": "リターンムーブ確認済み",
        "invalidated": "形が崩れた",
        "not_matched": "形として弱い",
    }.get(status or "", status or "")
    if status_label_ja:
        parts.append(
            f"<rect x='{margin_l}' y='{margin_t + plot_h + 6}' "
            f"width='{plot_w}' height='22' fill='#fff8e1' "
            f"stroke='#f9a825' stroke-width='1'/>"
        )
        parts.append(
            f"<text x='{margin_l + plot_w / 2}' "
            f"y='{margin_t + plot_h + 21}' text-anchor='middle' "
            f"fill='#5d4037' font-weight='bold' font-size='12'>"
            f"{_html_escape(status_label_ja)} "
            f"({_html_escape(side_bias or '')})</text>"
        )

    parts.append("</svg>")
    return "".join(parts)


_WAVE_LINE_STYLE: Final[dict] = {
    "neckline":              {"color": "#7b1fa2", "dash": "6,4", "width": 2.0},
    "pivot_low":             {"color": "#0d47a1", "dash": "0",   "width": 1.4},
    "pivot_high":            {"color": "#0d47a1", "dash": "0",   "width": 1.4},
    "shoulder":              {"color": "#0d47a1", "dash": "0",   "width": 1.4},
    "head":                  {"color": "#0d47a1", "dash": "0",   "width": 1.6},
    "pattern_invalidation":  {"color": "#c62828", "dash": "5,3", "width": 2.0},
    "pattern_target":        {"color": "#2e7d32", "dash": "5,3", "width": 2.0},
    "pattern_upper":         {"color": "#ef6c00", "dash": "4,3", "width": 1.6},
    "pattern_lower":         {"color": "#ef6c00", "dash": "4,3", "width": 1.6},
    "pattern_breakout":      {"color": "#7b1fa2", "dash": "6,4", "width": 2.0},
    "fibonacci_retracement": {"color": "#00838f", "dash": "3,3", "width": 1.2},
    "fibonacci_extension":   {"color": "#00695c", "dash": "3,3", "width": 1.2},
}


def _wave_derived_lines_svg_fragment(
    *,
    wave_derived_lines: list[dict] | None,
    y_of,
    price_lo: float, price_hi: float,
    margin_l: float, margin_t: float,
    plot_w: float, plot_h: float,
) -> str:
    """Build an SVG <g> overlay drawing wave-derived horizontal lines
    (WNL / WB1 / WB2 / WSL / WTP / WUP / WLOW / WBR) on top of the
    candle chart.

    Each line is keyed by `kind` to a colour / dash style; the line's
    `id` is rendered as a left-edge label so the user can cross-check
    with the description table below the chart.
    """
    if not wave_derived_lines:
        return ""
    out: list[str] = ["<g class='wave-derived-lines'>"]
    for line in wave_derived_lines:
        price = line.get("price")
        if price is None:
            continue
        try:
            price = float(price)
        except (TypeError, ValueError):
            continue
        if price < price_lo or price > price_hi:
            continue
        y = y_of(price)
        kind = line.get("kind", "")
        style = _WAVE_LINE_STYLE.get(kind, {
            "color": "#7b1fa2", "dash": "6,4", "width": 1.6,
        })
        line_id = line.get("id", "")
        out.append(
            f"<line class='wave-derived-line wave-derived-{_html_escape(kind)}' "
            f"x1='{margin_l:.1f}' y1='{y:.1f}' "
            f"x2='{margin_l + plot_w:.1f}' y2='{y:.1f}' "
            f"stroke='{style['color']}' stroke-width='{style['width']}' "
            f"stroke-dasharray='{style['dash']}' opacity='0.85'/>"
        )
        # Optional zone band (kind=="neckline" usually has a zone)
        zlow = line.get("zone_low")
        zhigh = line.get("zone_high")
        if zlow is not None and zhigh is not None:
            try:
                y_top = y_of(max(float(zlow), float(zhigh)))
                y_bot = y_of(min(float(zlow), float(zhigh)))
            except (TypeError, ValueError):
                y_top = y_bot = None
            if y_top is not None and y_bot is not None:
                out.append(
                    f"<rect class='wave-derived-zone' x='{margin_l:.1f}' "
                    f"y='{y_top:.1f}' width='{plot_w:.1f}' "
                    f"height='{abs(y_bot - y_top):.1f}' "
                    f"fill='{style['color']}' opacity='0.10'/>"
                )
        # Left-edge label
        label_w = max(28, 10 * len(line_id) + 8)
        out.append(
            f"<rect class='wave-derived-line-label-bg' "
            f"x='{margin_l + 4:.1f}' y='{y - 9:.1f}' "
            f"width='{label_w}' height='18' fill='white' "
            f"stroke='{style['color']}' stroke-width='1.2' rx='3'/>"
        )
        out.append(
            f"<text class='wave-derived-line-label' "
            f"x='{margin_l + 4 + label_w / 2:.1f}' y='{y + 4:.1f}' "
            f"text-anchor='middle' fill='{style['color']}' "
            f"font-weight='bold' font-size='11'>"
            f"{_html_escape(line_id)}</text>"
        )
    out.append("</g>")
    return "".join(out)


def _wave_overlay_svg_fragment(
    *,
    wave_overlay: dict,
    visible_index,
    x_of,
    y_of,
    price_lo: float, price_hi: float,
    margin_l: float, margin_t: float,
    plot_w: float, plot_h: float,
) -> str:
    """Build the SVG <g> overlay for the wave skeleton.

    All pivots whose ts is outside the visible window are skipped.
    No price / index outside the supplied chart bounds is referenced.
    """
    skel = wave_overlay.get("skeleton") or {}
    pivots = skel.get("pivots") or []
    matched_parts = wave_overlay.get("matched_parts") or {}
    kind = wave_overlay.get("kind") or ""
    human_label = wave_overlay.get("human_label") or kind
    if not pivots:
        return ""
    # Build a ts-key → bar index map (UTC ns) for fast lookup.
    ts_to_bar: dict[int, int] = {}
    for i, ts in enumerate(visible_index):
        try:
            ts_to_bar[int(pd.Timestamp(ts).value)] = i
        except Exception:  # noqa: BLE001
            continue

    points_xy: list[tuple[float, float, dict]] = []
    for p in pivots:
        ts_str = p.get("ts")
        if ts_str is None:
            continue
        try:
            key = int(pd.Timestamp(ts_str).value)
        except Exception:  # noqa: BLE001
            continue
        bar_i = ts_to_bar.get(key)
        if bar_i is None:
            continue
        try:
            price = float(p.get("price"))
        except (TypeError, ValueError):
            continue
        if price < price_lo or price > price_hi:
            # outside chart vertical extent — clamp at edge so dot is visible
            price = max(price_lo, min(price_hi, price))
        points_xy.append((x_of(bar_i), y_of(price), p))

    if len(points_xy) < 2:
        return ""

    out: list[str] = ["<g class='wave-skeleton-overlay'>"]
    # 1. Polyline through pivots (drawn first → rendered behind dots/labels)
    poly_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points_xy)
    out.append(
        f"<polyline class='wave-skeleton-line' points='{poly_pts}' "
        f"fill='none' stroke='#0d47a1' stroke-width='3.0' "
        f"stroke-linecap='round' stroke-linejoin='round' opacity='0.85'/>"
    )

    # 2. Pivot dots
    for x, y, p in points_xy:
        kind_marker = p.get("kind", "")
        out.append(
            f"<circle class='wave-pivot-dot' cx='{x:.1f}' cy='{y:.1f}' "
            f"r='4' fill='#0d47a1' stroke='white' stroke-width='1.2'/>"
        )
        if kind_marker:
            out.append(
                f"<text class='wave-pivot-kind' x='{x + 6:.1f}' "
                f"y='{y - 6:.1f}' fill='#0d47a1' font-size='10' "
                f"font-weight='bold'>{_html_escape(kind_marker)}</text>"
            )

    # 3. Pattern part labels — map matched_parts to pivots and emit
    #    "B1 (1回目の底)" style boxed labels.
    family_marker_map = _PATTERN_FAMILY_MARKER
    label_map = _PATTERN_PART_LABELS

    # Build {source_df_index → pivot_dict_with_position} so a matched
    # part can be located on the chart.
    by_index: dict[int, tuple[float, float, dict]] = {}
    for x, y, p in points_xy:
        try:
            by_index[int(p.get("index"))] = (x, y, p)
        except (TypeError, ValueError):
            continue

    neckline_price: float | None = None
    neckline_label = "NL"
    for part_name, src_idx in matched_parts.items():
        if src_idx is None:
            continue
        try:
            src_idx = int(src_idx)
        except (TypeError, ValueError):
            continue
        loc = by_index.get(src_idx)
        if loc is None:
            continue
        x, y, p = loc
        short_label = label_map.get(part_name)
        if not short_label:
            # Skip unmapped parts (do not emit ugly fallbacks like "lowe").
            continue
        # Background rect + text so labels stay readable over candles.
        text_w = max(20, 10 * len(short_label) + 8)
        text_h = 18
        # Place above the pivot for high pivots, below for lows.
        is_high = p.get("kind") == "H"
        ty = (y - 18) if is_high else (y + 6)
        out.append(
            f"<rect class='wave-part-label-bg' x='{x - text_w / 2:.1f}' "
            f"y='{ty:.1f}' width='{text_w}' height='{text_h}' "
            f"fill='white' stroke='#0d47a1' stroke-width='1.2' "
            f"rx='3' ry='3'/>"
        )
        out.append(
            f"<text class='wave-part-label' x='{x:.1f}' "
            f"y='{ty + text_h - 5:.1f}' text-anchor='middle' "
            f"fill='#0d47a1' font-weight='bold' font-size='11'>"
            f"{_html_escape(short_label)}</text>"
        )
        if "neckline" in part_name and neckline_price is None:
            try:
                neckline_price = float(p.get("price"))
                neckline_label = short_label
            except (TypeError, ValueError):
                pass

    # 4. Neckline horizontal line (if a neckline part was matched)
    if neckline_price is not None and price_lo <= neckline_price <= price_hi:
        ny = y_of(neckline_price)
        out.append(
            f"<line class='wave-neckline' x1='{margin_l:.1f}' "
            f"y1='{ny:.1f}' x2='{margin_l + plot_w:.1f}' y2='{ny:.1f}' "
            f"stroke='#7b1fa2' stroke-width='2.0' stroke-dasharray='6,4' "
            f"opacity='0.85'/>"
        )
        out.append(
            f"<text class='wave-neckline-label' x='{margin_l + 6:.1f}' "
            f"y='{ny - 4:.1f}' fill='#7b1fa2' font-weight='bold' "
            f"font-size='11'>"
            f"{_html_escape(neckline_label)} ネックライン</text>"
        )

    # 5. Family marker (DB1 / DT1 / HS1 / ...) at the upper-left of the
    #    overlay. Skipped for non-pattern overlays.
    family = family_marker_map.get(kind)
    if family:
        out.append(
            f"<rect class='wave-family-marker-bg' "
            f"x='{margin_l + 8:.1f}' y='{margin_t + 6:.1f}' "
            f"width='{18 + 8 * len(human_label)}' height='22' "
            f"fill='#0d47a1' opacity='0.9' rx='3'/>"
        )
        out.append(
            f"<text class='wave-family-marker' "
            f"x='{margin_l + 16:.1f}' y='{margin_t + 22:.1f}' "
            f"fill='white' font-weight='bold' font-size='11'>"
            f"{_html_escape(family)} {_html_escape(human_label)}</text>"
        )

    out.append("</g>")
    return "".join(out)


def _render_candle_svg(
    *,
    df: pd.DataFrame,
    end_ts: pd.Timestamp,
    n_bars: int,
    overlays: dict | None,
    title: str,
    out_path: Path,
) -> dict:
    """Thin wrapper around `_build_candle_svg_xml` that writes the SVG
    to disk and returns the legacy `image_status` dict shape."""
    try:
        xml = _build_candle_svg_xml(
            df=df, end_ts=end_ts, n_bars=n_bars,
            overlays=overlays, title=title,
        )
        if xml is None:
            marker = out_path.with_suffix(".image_unavailable")
            marker.write_text("no_visible_bars\n")
            return {
                "image_status": "render_error:no_visible_bars",
                "marker_file": str(marker),
            }
        out_path.write_text(xml)
        return {
            "image_status": "rendered",
            "image_path": str(out_path),
        }
    except Exception as e:  # noqa: BLE001
        marker = out_path.with_suffix(".image_unavailable")
        marker.write_text(f"svg_render_error: {type(e).__name__}: {e}\n")
        return {
            "image_status": f"render_error:{type(e).__name__}",
            "marker_file": str(marker),
        }


def _try_matplotlib_render(
    *,
    df: pd.DataFrame,
    end_ts: pd.Timestamp,
    n_bars: int,
    overlays: dict | None,
    title: str,
    out_path: Path,
) -> dict | None:
    """Internal helper that tries matplotlib. Returns the render result
    dict on success (image_status='rendered'), or None if matplotlib
    is unavailable / fails (so the caller can chain to SVG fallback).
    """
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None
    try:
        if df is None or len(df) == 0 or n_bars <= 0:
            return None
        visible = df[df.index <= end_ts].tail(n_bars)
        if len(visible) == 0:
            return None
        fig, ax = plt.subplots(figsize=(14, 8))
        x = list(visible.index)
        o = visible["open"].to_numpy()
        h = visible["high"].to_numpy()
        l = visible["low"].to_numpy()
        c = visible["close"].to_numpy()
        ax.vlines(x, l, h, color="black", linewidth=0.5, alpha=0.6)
        bullish = c >= o
        bull_x = [xi for xi, b in zip(x, bullish) if b]
        bull_lo = [oi for oi, b in zip(o, bullish) if b]
        bull_hi = [ci for ci, b in zip(c, bullish) if b]
        bear_x = [xi for xi, b in zip(x, bullish) if not b]
        bear_lo = [ci for ci, b in zip(c, bullish) if not b]
        bear_hi = [oi for oi, b in zip(o, bullish) if not b]
        if bull_x:
            ax.vlines(bull_x, bull_lo, bull_hi, color="green", linewidth=2.0)
        if bear_x:
            ax.vlines(bear_x, bear_lo, bear_hi, color="red", linewidth=2.0)
        if overlays is not None:
            for lvl in overlays.get("level_zones_selected", []):
                zlow = lvl.get("zone_low")
                zhigh = lvl.get("zone_high")
                if zlow is None or zhigh is None:
                    continue
                color = (
                    "green" if lvl.get("kind") == "support"
                    else "red" if lvl.get("kind") == "resistance"
                    else "purple"
                )
                ax.axhspan(zlow, zhigh, color=color, alpha=0.15)
            for lvl in overlays.get("level_zones_rejected", []):
                zlow = lvl.get("zone_low")
                zhigh = lvl.get("zone_high")
                if zlow is None or zhigh is None:
                    continue
                ax.axhspan(zlow, zhigh, color="gray", alpha=0.05)
            for t in overlays.get("trendlines_selected", []):
                slope = t.get("slope")
                intercept = t.get("intercept")
                anchors = t.get("anchor_indices") or []
                if slope is None or intercept is None or len(anchors) < 2:
                    continue
                idxs = list(range(anchors[0], anchors[-1] + 1))
                ts_xs = [visible.index[i] for i in idxs if i < len(visible)]
                ys = [slope * i + intercept for i in idxs[: len(ts_xs)]]
                if ts_xs:
                    ax.plot(
                        ts_xs, ys, color="blue", linewidth=0.8,
                        alpha=0.4 if t.get("broken") else 0.85,
                    )
            for rj in overlays.get("trendlines_rejected", []):
                tdict = rj.get("trendline") or {}
                slope = tdict.get("slope")
                intercept = tdict.get("intercept")
                anchors = tdict.get("anchor_indices") or []
                if slope is None or intercept is None or len(anchors) < 2:
                    continue
                idxs = list(range(anchors[0], anchors[-1] + 1))
                ts_xs = [visible.index[i] for i in idxs if i < len(visible)]
                ys = [slope * i + intercept for i in idxs[: len(ts_xs)]]
                if ts_xs:
                    ax.plot(
                        ts_xs, ys, color="gray", linewidth=0.5,
                        alpha=0.3, linestyle=":",
                    )
            for p in overlays.get("patterns_selected", []):
                nl = p.get("neckline")
                if nl is None:
                    continue
                color = (
                    "darkgreen" if p.get("side_bias") == "BUY"
                    else "darkred" if p.get("side_bias") == "SELL"
                    else "darkorange"
                )
                ax.axhline(nl, color=color, linestyle="--",
                           linewidth=0.8, alpha=0.7)
            ltf = overlays.get("lower_tf_trigger")
            if ltf and ltf.get("trigger_ts") and ltf.get("trigger_price"):
                try:
                    t_ts = pd.Timestamp(ltf["trigger_ts"])
                    ax.scatter(
                        [t_ts], [ltf["trigger_price"]],
                        color="orange", s=60, marker="^",
                        label=ltf.get("trigger_type"),
                    )
                except Exception:  # noqa: BLE001
                    pass
            sp = overlays.get("structure_stop_plan")
            if sp:
                if sp.get("stop_price") is not None:
                    ax.axhline(
                        sp["stop_price"], color="red", linestyle=":",
                        linewidth=1.0, alpha=0.7, label="stop",
                    )
                if sp.get("take_profit_price") is not None:
                    ax.axhline(
                        sp["take_profit_price"], color="green",
                        linestyle=":", linewidth=1.0, alpha=0.7, label="tp",
                    )
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return {"image_status": "rendered", "image_path": str(out_path)}
    except Exception:  # noqa: BLE001
        return None


def _render_candle_image(
    *,
    df: pd.DataFrame,
    end_ts: pd.Timestamp,
    n_bars: int,
    overlays: dict | None,
    title: str,
    out_path: Path,
) -> dict:
    """Best-effort candle render with three-tier fallback:

      1. matplotlib_png   →  out_path (PNG)
      2. svg_fallback     →  out_path.with_suffix('.svg')
      3. image_unavailable→  out_path.with_suffix('.image_unavailable')

    Returns a dict carrying both the legacy `image_status` field (for
    backward-compatible callers) AND the new explicit fields:
      `renderer`             — matplotlib_png | svg_fallback | image_unavailable
      `path`                 — chosen output path (relative-suitable str)
      `available`            — bool

    Future-leak safe in every path: only df bars with index <= end_ts
    are drawn.
    """
    # Tier 1 — matplotlib
    mp = _try_matplotlib_render(
        df=df, end_ts=end_ts, n_bars=n_bars,
        overlays=overlays, title=title, out_path=out_path,
    )
    if mp is not None and mp.get("image_status") == "rendered":
        return {
            "image_status": "rendered",
            "image_path": mp["image_path"],
            "renderer": "matplotlib_png",
            "path": mp["image_path"],
            "available": True,
        }
    # Tier 2 — SVG fallback (writes <out>.svg)
    svg_path = out_path.with_suffix(".svg")
    sv = _render_candle_svg(
        df=df, end_ts=end_ts, n_bars=n_bars,
        overlays=overlays, title=title, out_path=svg_path,
    )
    if sv.get("image_status") == "rendered":
        return {
            "image_status": "rendered",
            "image_path": sv["image_path"],
            "renderer": "svg_fallback",
            "path": sv["image_path"],
            "available": True,
        }
    # Tier 3 — image_unavailable marker
    marker = out_path.with_suffix(".image_unavailable")
    if not marker.exists():
        marker.write_text(
            f"image_unavailable: matplotlib failed and svg_fallback failed "
            f"({sv.get('image_status', 'unknown')})\n"
        )
    return {
        "image_status": sv.get("image_status", "render_error"),
        "image_path": None,
        "marker_file": str(marker),
        "renderer": "image_unavailable",
        "path": None,
        "available": False,
        "unavailable_reason": sv.get(
            "image_status", "image_unavailable"
        ),
    }


def _multi_scale_image_status(
    *,
    df: pd.DataFrame,
    payload: dict,
    case_dir: Path,
) -> dict:
    """Render one chart per scale (micro/short/medium/long).

    Each entry is a dict with keys (closed taxonomy):
      `available`             — bool
      `renderer`              — matplotlib_png | svg_fallback |
                                short_history | image_unavailable | none
      `path`                  — chosen output filename (relative) or None
      `unavailable_reason`    — closed-taxonomy reason or None
      `marker_file`           — path to .image_unavailable marker if any

    `path` is the file the HTML <img> should reference (PNG OR SVG).
    """
    out: dict[str, dict] = {}
    overlays = (payload.get("overlays") or {})
    scales = (
        payload.get("royal_road_decision_v2") or {}
    ).get("multi_scale_chart", {}).get("scales", {})
    parent_iso = payload.get("parent_bar_ts")
    end_ts = pd.Timestamp(parent_iso) if parent_iso else None
    for scale, n_bars in (
        ("micro", 30), ("short", 100),
        ("medium", 300), ("long", 1000),
    ):
        sc_info = scales.get(scale, {})
        png_path = case_dir / f"chart_{scale}.png"
        if not sc_info.get("available", False) or end_ts is None:
            marker = png_path.with_suffix(".image_unavailable")
            reason = sc_info.get("unavailable_reason", "short_history")
            marker.write_text(f"unavailable: {reason}\n")
            out[scale] = {
                "available": False,
                "renderer": "short_history",
                "path": None,
                "unavailable_reason": reason,
                "marker_file": str(marker),
            }
            continue
        title = (
            f"{scale} bars={n_bars} "
            f"vq={sc_info.get('visual_quality', 0.0):.2f} "
            f"trend={(sc_info.get('wave_structure') or {}).get('trend')}"
        )
        res = _render_candle_image(
            df=df, end_ts=end_ts, n_bars=n_bars,
            overlays=overlays, title=title, out_path=png_path,
        )
        # Map absolute path → filename (relative to case_dir) for HTML.
        rel_path = (
            Path(res["path"]).name
            if res.get("path") else None
        )
        out[scale] = {
            "available": bool(res.get("available")),
            "renderer": res.get("renderer", "image_unavailable"),
            "path": rel_path,
            "unavailable_reason": (
                None if res.get("available")
                else res.get("unavailable_reason", res.get("image_status"))
            ),
            "marker_file": res.get("marker_file"),
        }
    return out


def _lower_tf_image_status(
    *,
    df_lower: pd.DataFrame | None,
    payload: dict,
    case_dir: Path,
) -> dict:
    """Render lower_tf candles + trigger marker when df_lower attached
    and the trigger payload is available; otherwise emit marker only.

    Returns a chart_status entry matching the multi-scale taxonomy:
      `available`, `renderer`, `path`, `unavailable_reason`, `marker_file`.
    """
    png_path = case_dir / "chart_lower_tf.png"
    overlays = payload.get("overlays") or {}
    ltf = overlays.get("lower_tf_trigger")
    parent_iso = payload.get("parent_bar_ts")
    if df_lower is None or len(df_lower) == 0 or not ltf or not ltf.get("available"):
        marker = png_path.with_suffix(".image_unavailable")
        reason = (
            "no_lower_tf_data" if df_lower is None or len(df_lower) == 0
            else "lower_tf_unavailable_in_payload"
        )
        marker.write_text(f"unavailable: {reason}\n")
        return {
            "available": False,
            "renderer": "lower_tf_unavailable",
            "path": None,
            "unavailable_reason": reason,
            "marker_file": str(marker),
        }
    end_ts = pd.Timestamp(parent_iso) if parent_iso else None
    title = (
        f"lower_tf={ltf.get('interval')} "
        f"trigger={ltf.get('trigger_type')} "
        f"strength={ltf.get('trigger_strength')}"
    )
    overlays_for_ltf = {"lower_tf_trigger": ltf}
    res = _render_candle_image(
        df=df_lower, end_ts=end_ts, n_bars=200,
        overlays=overlays_for_ltf, title=title, out_path=png_path,
    )
    rel_path = Path(res["path"]).name if res.get("path") else None
    return {
        "available": bool(res.get("available")),
        "renderer": res.get("renderer", "image_unavailable"),
        "path": rel_path,
        "unavailable_reason": (
            None if res.get("available")
            else res.get("unavailable_reason", res.get("image_status"))
        ),
        "marker_file": res.get("marker_file"),
    }


# ---------------- HTML templates (static, no JS framework) ---------------


_DEFAULT_CSS = """\
body { font-family: -apple-system, sans-serif; margin: 16px; color: #222; }
h1, h2, h3 { color: #14202c; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; }
th { background: #f4f6f9; }
tr.priority-v2_directional { background: #fde8e8; }
tr.priority-v2_vs_current_diff { background: #fff5d6; }
tr.priority-high_quality_hold { background: #d9eaff; }
tr.priority-low_quality_directional_attempt { background: #ececec; }
tr.priority-fake_breakout_block { background: #ffe1d4; }
tr.priority-stop_mode_non_atr { background: #e7d6ff; }
tr.priority-lower_tf_trigger_present { background: #d4ffe1; }
tr.priority-selected_pattern_present { background: #e8f4ff; }
tr.priority-block_reason_representative { background: #f4f4f4; }
tr.priority-random_hold_filler { background: #fafafa; }
.case-detail { display: flex; gap: 16px; }
.case-detail .left { flex: 2; min-width: 0; }
.case-detail .right { flex: 1; min-width: 280px; }
pre { background: #f4f4f4; padding: 8px; overflow-x: auto; font-size: 12px; }
img { max-width: 100%; border: 1px solid #ddd; }
.section { margin-top: 16px; }
.placeholder { color: #888; font-style: italic; }
.checklist td.status-PASS { background: #d9f7d9; font-weight: bold; }
.checklist td.status-WARN { background: #fff3cd; font-weight: bold; }
.checklist td.status-BLOCK { background: #f8d7da; font-weight: bold; }
.panel { background: #fafbfd; padding: 8px 10px; border: 1px solid #e0e0e0;
  border-radius: 4px; margin-bottom: 10px; font-size: 13px; }
.panel h3 { margin-top: 0; font-size: 14px; }
.panel-unavailable { color: #888; font-style: italic; }
img.thumb { width: 220px; height: auto; border: 1px solid #ddd; }
.renderer-tag { color: #666; font-size: 11px; margin: 2px 0 0 0; }
.demo-banner { background: #fff8e1; border: 1px solid #f9a825;
  padding: 8px 12px; border-radius: 4px; margin: 8px 0; }

/* Mobile / narrow viewport: collapse the 2-column detail layout, make
   tables horizontally scrollable, and let charts use the full width.
   Pinned by tests/test_visual_audit.py (responsive_css tests). */
@media (max-width: 800px) {
  body { margin: 4px; padding: 8px; font-size: 14px; }
  h1 { font-size: 18px; }
  h2 { font-size: 16px; }
  h3 { font-size: 14px; }
  .case-detail { display: block; }
  .case-detail .left,
  .case-detail .right { flex: none; min-width: 0; }
  table { display: block; overflow-x: auto; white-space: nowrap; }
  table.checklist { white-space: normal; }
  img, svg { max-width: 100%; height: auto; }
  img.thumb { width: 100%; max-width: 360px; }
  pre { white-space: pre-wrap; word-break: break-word; }
  .panel { margin-bottom: 12px; padding: 6px 8px; }
  .demo-banner { font-size: 13px; }
  .case-section { margin-bottom: 24px; }
}

/* observation-only waveform shape review section */
.wave-shape-review { background: #f7f9ff; border: 1px solid #c5d2eb;
  border-radius: 4px; padding: 8px 12px; margin: 8px 0; }
.wave-shape-review .wave-table { font-size: 12px; }
.wave-shape-review .wave-best-pattern { background: #fff; padding: 6px 8px;
  border-left: 3px solid #2c5fa1; margin-top: 6px; }
.wave-shape-review .audit-notes { color: #777; font-size: 11px; }
.pattern-dissection { background: #fffaf0; border: 1px solid #f0d49a;
  border-radius: 4px; padding: 6px 10px; margin: 6px 0; }
.pattern-dissection .dissection-table { font-size: 12px; }
.entry-summary-card { background: #f0f8f0; border: 2px solid #6fb56f;
  border-radius: 6px; padding: 10px 14px; margin: 8px 0; }
.entry-summary-card .entry-table { font-size: 13px; max-width: 360px; }
.entry-summary-card .small { color: #555; font-size: 11px; margin: 2px 0; }
.hold-waveform-note { background: #fff5e6; border: 1px solid #d8a662;
  border-radius: 4px; padding: 8px 12px; margin: 8px 0; }
.hold-waveform-note h3 { margin-top: 6px; font-size: 13px; }
.hold-waveform-note .small { color: #555; font-size: 11px; }
.hold-waveform-note h4 { margin: 8px 0 4px; font-size: 13px; }
/* wave-derived lines (entry / stop / target / parts) */
.wave-derived-line { pointer-events: none; }
.wave-derived-line-label-bg, .wave-derived-line-label { pointer-events: none; }
.wave-derived-zone { pointer-events: none; }
.wave-derived-lines-table { background: #f7f9ff; border: 1px solid #c5d2eb;
  border-radius: 4px; padding: 8px 12px; margin: 8px 0; }
.wave-derived-lines-table .small { color: #555; font-size: 11px; }
.wave-derived-table { font-size: 12px; }
.line-count-summary-v2 { background: #fffaf0; border: 1px solid #f0d49a;
  border-radius: 4px; padding: 8px 12px; margin: 8px 0; }
.line-count-summary-v2 .line-count-table { font-size: 12px; max-width: 360px; }
.line-count-summary-v2 .small { color: #555; font-size: 11px; margin-top: 4px; }
.line-count-summary-v2 .note-info { background: #e8f4ff; border-left: 3px solid #1565c0;
  padding: 6px 8px; margin-top: 6px; font-size: 12px; }
.line-count-summary-v2 .note-warn { background: #ffe5e0; border-left: 3px solid #c62828;
  padding: 6px 8px; margin-top: 6px; font-size: 12px; }
.wave-line-legend { background: #f7f9ff; border-left: 3px solid #0d47a1;
  padding: 6px 12px; margin: 6px 0; font-size: 12px; }
.wave-line-legend .small { margin: 2px 0; }
.wave-line-legend-list { margin: 4px 0 4px 18px; padding: 0; }
.wave-line-legend-list li { margin: 2px 0; }
/* Masterclass observation-only panels (16 features) */
.masterclass-panels { margin: 8px 0; }
.masterclass-detail { background: #f9faff; border: 1px solid #d0d7e8;
  border-radius: 4px; margin: 6px 0; padding: 6px 10px; }
.masterclass-detail summary { cursor: pointer; font-weight: 600;
  color: #0d3a6a; font-size: 13px; }
.masterclass-detail .obs-tag { color: #888; font-weight: normal;
  font-size: 11px; margin-left: 4px; }
.masterclass-detail-body { margin-top: 6px; }
.masterclass-category { background: #fafbfd; border: 2px solid #94a3b8;
  border-radius: 6px; padding: 6px 10px; margin: 8px 0; }
.masterclass-category-title { cursor: pointer; font-weight: 700;
  color: #334155; padding: 4px 0; font-size: 14px; }
.masterclass-detail table.kv { font-size: 12px; }
.masterclass-detail table.kv th { width: 30%; background: #eef3fa; }
.masterclass-detail table.kv td.status-PASS { background: #d9f7d9; font-weight: bold; }
.masterclass-detail table.kv td.status-WARN { background: #fff3cd; font-weight: bold; }
.masterclass-detail table.kv td.status-BLOCK { background: #f8d7da; font-weight: bold; }
.masterclass-detail table.kv td.status-UNKNOWN { background: #e0e0e0; }
.masterclass-detail table.kv td.status-YES { background: #d9f7d9; font-weight: bold; }
.masterclass-detail table.kv td.status-NO { background: #f8d7da; font-weight: bold; }
.masterclass-detail .small { color: #555; font-size: 11px; margin: 4px 0; }
/* Tier 1 / Tier 2 summary blocks (3-tier layout: summary → check → detail) */
.masterclass-tier1 { background: #fff8e1; border: 2px solid #f9a825;
  border-radius: 6px; padding: 8px 12px; margin: 8px 0; }
.masterclass-tier1 h4 { color: #5d4037; margin: 0 0 6px; font-size: 14px; }
.masterclass-tier1 .tier1-table { font-size: 12px; width: 100%; }
.masterclass-tier1 .tier1-table th { background: #fff3cd; }
.masterclass-tier2 { background: #f0f8f0; border: 1px solid #6fb56f;
  border-radius: 4px; padding: 8px 12px; margin: 8px 0; }
.masterclass-tier2 h4 { color: #2e5f2e; margin: 0 0 6px; font-size: 13px; }
.masterclass-tier2 .tier2-table { font-size: 12px; width: 100%; }
.masterclass-tier2 .tier2-table th { background: #d9f7d9; }
.masterclass-tier1 .tier1-table td.status-PASS,
.masterclass-tier2 .tier2-table td.status-PASS { background: #d9f7d9; font-weight: bold; }
.masterclass-tier1 .tier1-table td.status-WARN,
.masterclass-tier2 .tier2-table td.status-WARN { background: #fff3cd; font-weight: bold; }
.masterclass-tier1 .tier1-table td.status-BLOCK,
.masterclass-tier2 .tier2-table td.status-BLOCK { background: #f8d7da; font-weight: bold; }
.masterclass-tier1 .tier1-table td.status-UNKNOWN,
.masterclass-tier2 .tier2-table td.status-UNKNOWN { background: #e0e0e0; }
.masterclass-tier1 .tier1-table td.status-USED { background: #d9f7d9; font-weight: bold; }
.masterclass-tier1 .tier1-table td.status-AUDIT_ONLY { background: #fff3cd; font-weight: bold; }
.masterclass-tier1 .tier1-table td.status-NOT_CONNECTED { background: #f8d7da; font-weight: bold; }
/* Decision bridge — top of every case section */
.decision-bridge { background: #fff8e1; border: 3px solid #d84315;
  border-radius: 6px; padding: 10px 14px; margin: 10px 0;
  box-shadow: 0 2px 4px rgba(0,0,0,0.08); }
.decision-bridge h3 { color: #bf360c; margin: 0 0 8px; font-size: 16px; }
.decision-bridge h4 { color: #5d4037; margin: 6px 0 4px; font-size: 13px; }
.decision-bridge .bridge-section { background: white; padding: 8px 10px;
  margin: 6px 0; border-radius: 4px; }
.decision-bridge .bridge-final-line { font-size: 18px; font-weight: bold;
  margin: 4px 0; }
.decision-bridge .bridge-action-BUY { color: #2e7d32; font-size: 1.4em; }
.decision-bridge .bridge-action-SELL { color: #c62828; font-size: 1.4em; }
.decision-bridge .bridge-action-HOLD { color: #5d4037; font-size: 1.4em; }
.decision-bridge .bridge-action-unknown { color: #888; font-size: 1.4em; }
.decision-bridge .bridge-action-message { white-space: pre-line;
  background: #fffbf0; padding: 6px 8px; border-left: 3px solid #f9a825;
  margin: 4px 0; font-size: 12px; }
.decision-bridge .bridge-entry-brief { font-size: 12px; max-width: 320px;
  margin-top: 6px; }
.decision-bridge .bridge-entry-brief th { background: #fff3cd; }
.decision-bridge .bridge-list { padding-left: 18px; margin: 4px 0;
  font-size: 12px; }
.decision-bridge .bridge-list li { margin: 4px 0; padding: 4px 6px;
  background: white; border-radius: 4px; }
.decision-bridge .bridge-used { border-left: 4px solid #2e7d32; }
.decision-bridge .bridge-partial { border-left: 4px solid #1565c0; }
.decision-bridge .bridge-audit-only { border-left: 4px solid #f9a825; }
.decision-bridge .bridge-not-connected { border-left: 4px solid #c62828; }
.decision-bridge .bridge-unknown { border-left: 4px solid #888; }
.decision-bridge .bridge-tag { font-size: 11px; padding: 1px 6px;
  border-radius: 3px; color: white; font-weight: bold; }
.decision-bridge .bridge-tag-used { background: #2e7d32; }
.decision-bridge .bridge-tag-partial { background: #1565c0; }
.decision-bridge .bridge-tag-audit-only { background: #f9a825; }
.decision-bridge .integrated-banner { background: #e3f2fd;
  border: 2px solid #1565c0; border-radius: 4px; padding: 6px 10px;
  margin: 4px 0 10px; font-size: 12px; }
.decision-bridge .integrated-banner p { margin: 2px 0; }
/* Phase F state cards (wave gate / breakout quality / entry status) */
.decision-bridge .phase-f-section { background: #fffbf0;
  border: 2px solid #c97b09; border-radius: 5px; padding: 8px 10px;
  margin: 6px 0; }
.decision-bridge .phase-f-cards h4 { color: #c97b09; margin: 0 0 6px;
  font-size: 13px; }
.decision-bridge .phase-f-card { background: white; border-radius: 4px;
  padding: 6px 8px; margin: 4px 0; border-left: 4px solid #888; }
.decision-bridge .phase-f-pass, .decision-bridge .phase-f-status-ready {
  color: #2e7d32; font-weight: bold; }
.decision-bridge .phase-f-warn,
.decision-bridge .phase-f-status-wait-breakout,
.decision-bridge .phase-f-status-wait-retest { color: #f9a825; font-weight: bold; }
.decision-bridge .phase-f-block, .decision-bridge .phase-f-status-hold {
  color: #c62828; font-weight: bold; }
.decision-bridge .phase-f-bq-table { font-size: 11px; margin-top: 4px;
  border-collapse: collapse; width: 100%; }
.decision-bridge .phase-f-bq-table th { text-align: left;
  padding: 2px 4px; background: #fff3cd; }
.decision-bridge .phase-f-bq-table td { padding: 2px 4px; }
.decision-bridge .bridge-tag-not-connected { background: #c62828; }
.decision-bridge .bridge-tag-unknown { background: #888; }
.decision-bridge .bridge-reason { color: #333; }
.decision-bridge .bridge-check { color: #5d4037; margin: 2px 0 0; }
.decision-bridge .bridge-plain { font-style: italic; color: #555;
  background: #fffbf0; border: 1px dashed #d84315; padding: 6px 8px;
  margin-top: 8px; font-size: 12px; }
/* wave overlay (drawn on top of candle chart) */
.wave-skeleton-line { pointer-events: none; }
.wave-pivot-dot { pointer-events: none; }
.wave-part-label-bg, .wave-part-label { pointer-events: none; }
.wave-neckline { pointer-events: none; }
/* standalone wave-only chart */
.wave-only-wrap { margin: 8px 0; max-width: 100%; overflow-x: auto; }
.wave-only-chart { display: block; max-width: 100%; height: auto;
  border: 1px solid #c5d2eb; border-radius: 4px; background: #fbfcff; }
"""


def _setup_candidate_summary_ja(i: int, c: dict) -> str:
    """User-facing JA summary line for a setup_candidate <details>.

    Replaces the dev-style "candidate #1 side=BUY score=-0.300
    confidence=0.55" string. Raw JSON stays inside <pre> for debug.
    """
    side = (c.get("side") or "?").upper()
    side_ja = {"BUY": "BUY (買い)", "SELL": "SELL (売り)"}.get(side, side)
    raw_score = c.get("score")
    if isinstance(raw_score, (int, float)):
        if raw_score >= 0.5:
            strength_ja = "強い"
        elif raw_score >= 0.0:
            strength_ja = "やや弱い"
        elif raw_score >= -0.5:
            strength_ja = "やや逆"
        else:
            strength_ja = "強く逆"
    else:
        strength_ja = "—"
    label = c.get("label") or ""
    label_part = f", {_html_escape(label)}" if label else ""
    return _html_escape(f"候補 #{i + 1}: {side_ja}{label_part} ({strength_ja})")


def _html_escape(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;").replace("'", "&#39;")
    )


def _render_index_html(
    *,
    cases: list[dict],
    summary: dict,
    out_dir: Path | None = None,
    extra_header_html: str = "",
) -> str:
    """Render the top-level index.html.

    For each row, a thumbnail column attempts (in priority order):
      1. <case_dir>/chart_main.png
      2. <case_dir>/chart_main.svg
      3. textual placeholder
    `out_dir` lets us check existence on disk; when omitted, falls
    back to no thumbnail.
    """
    rows = []
    for c in cases:
        ts_safe = _safe_ts_for_path(c["ts"])
        href = f"{c['symbol']}/{ts_safe}/detail.html"
        rq = c.get("rq", 0.0)
        v2 = c.get("v2", {})
        diff = (v2.get("compared_to_current_runtime") or {}).get(
            "difference_type", ""
        )
        sp = (v2.get("structure_stop_plan") or {})
        cls = f"priority-{c.get('priority', 'random_hold_filler')}"
        thumb_html = "<span class='placeholder'>n/a</span>"
        if out_dir is not None:
            png = out_dir / c["symbol"] / ts_safe / "chart_main.png"
            svg = out_dir / c["symbol"] / ts_safe / "chart_main.svg"
            if png.exists():
                thumb_html = (
                    f"<img class='thumb' src='{_html_escape(c['symbol'])}/"
                    f"{ts_safe}/chart_main.png' alt='thumb'>"
                )
            elif svg.exists():
                thumb_html = (
                    f"<img class='thumb' src='{_html_escape(c['symbol'])}/"
                    f"{ts_safe}/chart_main.svg' alt='thumb'>"
                )
        rows.append(
            f"<tr class='{cls}'>"
            f"<td><a href='{_html_escape(href)}'>open</a></td>"
            f"<td>{thumb_html}</td>"
            f"<td>{_html_escape(c.get('priority', ''))}</td>"
            f"<td>{_html_escape(c['symbol'])}</td>"
            f"<td>{_html_escape(c['ts'])}</td>"
            f"<td>{_html_escape(v2.get('action', ''))}</td>"
            f"<td>{_html_escape(v2.get('mode', ''))}</td>"
            f"<td>{rq:.3f}</td>"
            f"<td>{_html_escape(diff)}</td>"
            f"<td>{_html_escape(sp.get('chosen_mode', ''))}</td>"
            f"<td>{_html_escape(','.join(c.get('block_reasons', [])[:2]))}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows)
    summary_html = (
        f"<p>schema={_html_escape(summary.get('schema_version', ''))} "
        f"total={summary.get('n_cases', 0)} "
        f"max={summary.get('max_cases', 0)}</p>"
        "<p>by_priority: <code>"
        + _html_escape(json.dumps(summary.get("by_priority", {})))
        + "</code></p>"
    )
    return (
        "<html><head><meta charset='utf-8'><title>visual_audit_report_v1</title>"
        "<link rel='stylesheet' href='assets/style.css'></head><body>"
        "<h1>visual_audit_report_v1</h1>"
        + extra_header_html
        + summary_html
        + "<table><thead><tr><th>open</th><th>thumbnail</th><th>priority</th>"
        "<th>symbol</th><th>timestamp</th><th>action</th><th>mode</th>"
        "<th>quality</th><th>diff</th><th>stop_mode</th>"
        "<th>top block_reasons</th></tr></thead>"
        f"<tbody>{rows_html}</tbody></table></body></html>"
    )


def _render_steps_html(v2: dict) -> str:
    """Render a 'royal-road step trace' summary section so reviewers
    can see WHY the system did what it did, in step order."""
    rq = v2.get("reconstruction_quality") or {}
    macro = v2.get("macro_alignment") or {}
    sr = v2.get("support_resistance_v2") or {}
    cp = v2.get("chart_pattern_v2") or {}
    ltf = v2.get("lower_tf_trigger") or {}
    sp = v2.get("structure_stop_plan") or {}
    block = v2.get("block_reasons") or []
    cautions = v2.get("cautions") or []
    rows: list[str] = []

    def step(label: str, status: str, detail: str) -> str:
        return (
            f"<tr><td><b>{_html_escape(label)}</b></td>"
            f"<td>{_html_escape(status)}</td>"
            f"<td>{_html_escape(detail)}</td></tr>"
        )

    rows.append(step(
        "Step 1: Risk Gate",
        "BLOCK" if any(r.startswith("gate:") for r in block) else "PASS",
        ", ".join(r for r in block if r.startswith("gate:")) or "ok",
    ))
    rows.append(step(
        "Step 2: Macro",
        macro.get("macro_strong_against") if macro.get("macro_strong_against") not in (None, "UNKNOWN") else "PASS",
        f"score={macro.get('macro_score')} bias={macro.get('currency_bias')} "
        f"vix={macro.get('vix_regime')}",
    ))
    rows.append(step(
        "Step 3: Higher Timeframe",
        "BLOCK" if any("htf_counter_trend" in r for r in block) else "PASS",
        ", ".join(r for r in block if "htf_counter_trend" in r) or "aligned",
    ))
    rows.append(step(
        "Step 4: Structure",
        "PASS" if (v2.get("evidence_axes_count") or {}).get("bullish", 0) +
                  (v2.get("evidence_axes_count") or {}).get("bearish", 0) > 0
        else "WEAK",
        f"axes={v2.get('evidence_axes_count')}",
    ))
    rows.append(step(
        "Step 5: Support / Resistance",
        "BLOCK" if any(r in (
            "near_strong_resistance_for_buy", "near_strong_support_for_sell",
            "sr_fake_breakout",
        ) for r in block) else "PASS",
        f"selected={len(sr.get('selected_level_zones_top5', []))} "
        f"rejected={len(sr.get('rejected_level_zones', []))}",
    ))
    rows.append(step(
        "Step 6: Pattern",
        "PASS" if (cp.get("selected_patterns_top5") or []) else "NO_MATCH",
        f"selected_kinds={[p.get('kind') for p in cp.get('selected_patterns_top5', [])]}",
    ))
    rows.append(step(
        "Step 7: Lower TF",
        "PASS" if ltf.get("bullish_trigger") or ltf.get("bearish_trigger")
        else ("UNAVAILABLE" if not ltf.get("available") else "NO_TRIGGER"),
        f"type={ltf.get('trigger_type')} strength={ltf.get('trigger_strength')}",
    ))
    rows.append(step(
        "Step 8: Stop / RR",
        "BLOCK" if any("stop_plan_invalid" in r or r.startswith("rr<") for r in block)
        else "PASS",
        f"mode={sp.get('chosen_mode')} outcome={sp.get('outcome')} "
        f"rr={sp.get('rr_realized')}",
    ))
    rows.append(step(
        "Step 9: Reconstruction Quality",
        "BLOCK" if "insufficient_royal_road_reconstruction_quality" in block
        else ("PASS" if rq.get("total_reconstruction_score", 0.0) >= 0.4
              else "WEAK"),
        f"total={rq.get('total_reconstruction_score', 0.0):.3f}",
    ))
    rows.append(step(
        "Final",
        v2.get("action") or "HOLD",
        ("block_reasons=" + ", ".join(block)) if block else "no blocks",
    ))
    if cautions:
        rows.append(step(
            "Cautions", "WARN", ", ".join(cautions),
        ))
    return (
        "<table><thead><tr><th>Step</th><th>Status</th><th>Detail</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _render_grand_confluence_checklist_html(panel: dict) -> str:
    if not panel.get("available"):
        return (
            f"<p class='panel-unavailable'>"
            f"grand_confluence_checklist unavailable: "
            f"{_html_escape(panel.get('unavailable_reason', ''))}</p>"
        )
    rows: list[str] = []
    for it in panel.get("items", []):
        status = it.get("status", "")
        rows.append(
            f"<tr><td>{_html_escape(it.get('item', ''))}</td>"
            f"<td class='status-{_html_escape(status)}'>{_html_escape(status)}</td>"
            f"<td>{_html_escape(it.get('detail', ''))}</td></tr>"
        )
    summary = (
        f"<p>final_action=<b>{_html_escape(panel.get('final_action', ''))}</b> "
        f"PASS={panel.get('total_pass', 0)} "
        f"WARN={panel.get('total_warn', 0)} "
        f"BLOCK={panel.get('total_block', 0)} "
        f"(total={panel.get('total_items', 0)})</p>"
    )
    return (
        summary
        + "<table class='checklist'><thead><tr><th>item</th><th>status</th>"
        "<th>detail</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _safe_cell(v) -> str:
    """User-facing cell value: None → "—", float NaN → "—", floats
    formatted, others stringified. Prevents bare "None" / "NaN" from
    leaking into HTML."""
    if v is None:
        return "—"
    if isinstance(v, float):
        try:
            import math
            if math.isnan(v):
                return "—"
        except Exception:  # noqa: BLE001
            pass
        return f"{v:.5f}"
    return _html_escape(str(v))


def _render_invalidation_explanation_html(panel: dict) -> str:
    if not panel.get("available"):
        return (
            f"<p class='panel-unavailable'>"
            f"invalidation_explanation unavailable: "
            f"{_html_escape(panel.get('unavailable_reason', ''))}</p>"
        )
    return (
        "<table>"
        f"<tr><td>chosen_mode</td><td>{_safe_cell(panel.get('chosen_mode'))}</td></tr>"
        f"<tr><td>outcome</td><td>{_safe_cell(panel.get('outcome'))}</td></tr>"
        f"<tr><td>selected_stop_price</td><td>{_safe_cell(panel.get('selected_stop_price'))}</td></tr>"
        f"<tr><td>atr_stop_price</td><td>{_safe_cell(panel.get('atr_stop_price'))}</td></tr>"
        f"<tr><td>structure_stop_price</td><td>{_safe_cell(panel.get('structure_stop_price'))}</td></tr>"
        f"<tr><td>invalidation_structure</td><td>{_safe_cell(panel.get('invalidation_structure'))}</td></tr>"
        f"<tr><td>invalidation_status</td><td><b>{_safe_cell(panel.get('invalidation_status'))}</b></td></tr>"
        f"<tr><td>rr_selected</td><td>{_safe_cell(panel.get('rr_selected'))}</td></tr>"
        f"<tr><td>why_this_stop_invalidates_the_setup</td><td>"
        f"{_safe_cell(panel.get('why_this_stop_invalidates_the_setup'))}"
        "</td></tr></table>"
    )


def _render_simple_panel_html(name: str, panel: dict) -> str:
    if not panel.get("available"):
        return (
            f"<p class='panel-unavailable'>{_html_escape(name)} unavailable: "
            f"{_html_escape(panel.get('unavailable_reason', ''))}</p>"
        )
    rows: list[str] = []
    for k, v in panel.items():
        if k in ("available", "source"):
            continue
        rows.append(
            f"<tr><td>{_html_escape(k)}</td>"
            f"<td>{_html_escape(json.dumps(v, default=str))}</td></tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def _render_checklist_panels_html(panels: dict | None) -> str:
    if not panels:
        return "<p class='panel-unavailable'>no checklist panels</p>"
    parts: list[str] = []
    parts.append(
        "<div class='panel'><h3>Grand Confluence Checklist</h3>"
        + _render_grand_confluence_checklist_html(
            panels.get("grand_confluence_checklist") or {}
        )
        + "</div>"
    )
    parts.append(
        "<div class='panel'><h3>Invalidation Explanation</h3>"
        + _render_invalidation_explanation_html(
            panels.get("invalidation_explanation") or {}
        )
        + "</div>"
    )
    for label, key in (
        ("Candlestick Interpretation", "candlestick_interpretation"),
        ("RSI Context / Trap", "rsi_context"),
        ("MA Granville Context", "ma_granville_context"),
        ("Bollinger Lifecycle", "bollinger_lifecycle"),
        ("Current Runtime Indicator Votes", "current_runtime_indicator_votes"),
    ):
        parts.append(
            f"<div class='panel'><h3>{label}</h3>"
            + _render_simple_panel_html(label, panels.get(key) or {})
            + "</div>"
        )
    return "".join(parts)


def _render_detail_html(
    *,
    case: dict,
    payload: dict,
    chart_main_status: dict,
    multi_scale_status: dict,
    lower_tf_status: dict,
) -> str:
    v2 = payload.get("royal_road_decision_v2") or {}
    cmp_v1 = v2.get("compared_to_current_runtime") or {}
    sym = case["symbol"]
    ts = case["ts"]
    # User-facing title (no dev-style best=/quality=/score=). The
    # debug-style string is kept on payload["title"] for tests / JSON
    # sidecars; this rendering uses the sanitized JA version.
    action_label = v2.get("action") or "?"
    mode_label = v2.get("mode") or ""
    title = f"{sym} {ts} 判断: {action_label}"
    if mode_label:
        title += f" (mode: {mode_label})"
    # Resolve the actual chart filename to <img>: PNG > SVG > placeholder.
    main_path = chart_main_status.get("path")
    if chart_main_status.get("available") and main_path:
        main_fname = Path(main_path).name
        renderer = chart_main_status.get("renderer", "matplotlib_png")
        main_img = (
            f"<img src='{_html_escape(main_fname)}' "
            f"alt='main chart ({_html_escape(renderer)})'>"
            f"<p class='renderer-tag'>renderer: {_html_escape(renderer)}</p>"
        )
    else:
        main_img = (
            "<p class='placeholder'>chart_main not rendered: "
            f"{_html_escape(chart_main_status.get('unavailable_reason', 'unknown'))}</p>"
        )

    def _img_or_placeholder(scale: str) -> str:
        info = multi_scale_status.get(scale, {})
        path = info.get("path")
        if info.get("available") and path:
            renderer = info.get("renderer", "matplotlib_png")
            return (
                f"<h4>{scale} <span class='renderer-tag'>"
                f"({_html_escape(renderer)})</span></h4>"
                f"<img src='{_html_escape(Path(path).name)}'>"
            )
        return (
            f"<h4>{scale}</h4>"
            f"<p class='placeholder'>{_html_escape(info.get('unavailable_reason', 'unavailable'))}</p>"
        )

    multi_scale_html = "".join(
        _img_or_placeholder(s) for s in ("micro", "short", "medium", "long")
    )
    ltf_path = lower_tf_status.get("path")
    if lower_tf_status.get("available") and ltf_path:
        ltf_renderer = lower_tf_status.get("renderer", "matplotlib_png")
        lower_tf_html = (
            f"<img src='{_html_escape(Path(ltf_path).name)}' alt='lower TF'>"
            f"<p class='renderer-tag'>renderer: {_html_escape(ltf_renderer)}</p>"
        )
    else:
        lower_tf_html = (
            f"<p class='placeholder'>lower_tf: "
            f"{_html_escape(lower_tf_status.get('unavailable_reason', ''))}</p>"
        )
    setup_candidates = v2.get("setup_candidates") or []
    setup_html = "".join(
        f"<details><summary>{_setup_candidate_summary_ja(i, c)}</summary>"
        f"<pre>{_html_escape(json.dumps(c, indent=2, default=str))}</pre>"
        "</details>"
        for i, c in enumerate(setup_candidates)
    ) or "<p class='placeholder'>セットアップ候補なし</p>"
    best_setup = v2.get("best_setup")
    best_html = (
        f"<pre>{_html_escape(json.dumps(best_setup, indent=2, default=str))}</pre>"
        if best_setup else "<p class='placeholder'>該当なし</p>"
    )
    block_reasons = v2.get("block_reasons") or []
    cautions = v2.get("cautions") or []
    block_html = (
        "<ul>" + "".join(
            f"<li><b>{_html_escape(r)}</b></li>" for r in block_reasons
        ) + "</ul>"
        if block_reasons else "<p>(none)</p>"
    )
    cautions_html = (
        "<ul>" + "".join(
            f"<li>{_html_escape(c)}</li>" for c in cautions
        ) + "</ul>"
        if cautions else "<p>(none)</p>"
    )
    rq = v2.get("reconstruction_quality") or {}
    rq_html = (
        f"<pre>{_html_escape(json.dumps(rq, indent=2, default=str))}</pre>"
    )
    cmp_html = (
        f"<table><tr><th></th><th>current_runtime</th><th>royal_road_v2</th></tr>"
        f"<tr><td>action</td><td>{_html_escape(cmp_v1.get('current_action', ''))}</td>"
        f"<td>{_html_escape(cmp_v1.get('royal_road_action', ''))}</td></tr>"
        f"<tr><td>same?</td><td colspan='2'>{cmp_v1.get('same_action')}</td></tr>"
        f"<tr><td>diff</td><td colspan='2'>{_html_escape(cmp_v1.get('difference_type', ''))}</td></tr>"
        "</table>"
    )
    panels_html = _render_checklist_panels_html(
        payload.get("checklist_panels")
    )
    return (
        "<html><head><meta charset='utf-8'>"
        f"<title>{_html_escape(case['case_id'])}</title>"
        "<link rel='stylesheet' href='../../assets/style.css'></head><body>"
        f"<h1>{_html_escape(title)}</h1>"
        f"<p><a href='../../index.html'>&larr; back to index</a></p>"
        "<div class='case-detail'>"
        f"<div class='left'><h2>Main</h2>{main_img}"
        f"<h2 class='section'>Royal Road Checklist Panels</h2>{panels_html}"
        f"<h2 class='section'>Multi Scale</h2>{multi_scale_html}"
        f"<h2 class='section'>Lower TF</h2>{lower_tf_html}"
        "</div>"
        "<div class='right'>"
        f"<h2>Step Trace</h2>{_render_steps_html(v2)}"
        f"<h2 class='section'>current_runtime vs royal_road_v2</h2>{cmp_html}"
        f"<h2 class='section'>best_setup</h2>{best_html}"
        f"<h2 class='section'>setup_candidates</h2>{setup_html}"
        f"<h2 class='section'>block_reasons</h2>{block_html}"
        f"<h2 class='section'>cautions</h2>{cautions_html}"
        f"<h2 class='section'>reconstruction_quality</h2>{rq_html}"
        "<h2 class='section'>Raw audit.json</h2>"
        "<p><a href='audit.json'>open audit.json</a></p>"
        "</div></div></body></html>"
    )


def _feedback_template(case_id: str, v2: dict) -> dict:
    sr = v2.get("support_resistance_v2") or {}
    tl = v2.get("trendline_context") or {}
    cp = v2.get("chart_pattern_v2") or {}
    return {
        "case_id": case_id,
        "human_review": {
            "sr_zones": [
                {"candidate_id": f"S{i+1}", "rating": None, "comment": ""}
                for i in range(len(sr.get("selected_level_zones_top5", [])))
            ],
            "trendlines": [
                {"candidate_id": f"T{i+1}", "rating": None, "comment": ""}
                for i in range(len(tl.get("selected_trendlines_top3", [])))
            ],
            "patterns": [
                {"candidate_id": f"P{i+1}", "rating": None, "comment": ""}
                for i in range(len(cp.get("selected_patterns_top5", [])))
            ],
            "final_decision": {
                "system_action_reasonable": None,
                "preferred_action": None,
                "comment": "",
            },
        },
    }


def render_visual_audit_report(
    *,
    traces: Iterable[Any],
    df_by_symbol: dict[str, pd.DataFrame],
    df_lower_by_symbol: dict[str, pd.DataFrame] | None = None,
    out_dir: Path | str,
    max_cases: int = 200,
    profile: str = "royal_road_decision_v2",
    demo_fixture_banner: str | None = None,
) -> dict:
    """Top-level batch report orchestrator. See module docstring for the
    layout written to `out_dir`.

    `df_by_symbol` and `df_lower_by_symbol` are mappings from symbol →
    OHLC DataFrame (1h base) / OHLC DataFrame (lower TF). Lower-TF dict
    is optional; when omitted (or symbol absent), no lower_tf chart is
    rendered for that symbol.

    `demo_fixture_banner` (optional): when set, the string is rendered as
    a prominent banner at the top of index.html and the `summary.json`
    carries a `demo_fixture_not_backtest_result=True` flag. Use this when
    feeding the orchestrator handcrafted traces for UI verification —
    NOT when emitting real backtest output.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = out_dir / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "style.css").write_text(_DEFAULT_CSS)

    selected = select_important_cases(traces, max_cases=max_cases)
    cases_summary: list[dict] = []
    by_priority: dict[str, int] = {p: 0 for p in PRIORITY_LABELS}

    for case in selected:
        sym = case["symbol"]
        ts_safe = _safe_ts_for_path(case["ts"])
        case_id = case["case_id"]
        case_dir = out_dir / sym / ts_safe
        case_dir.mkdir(parents=True, exist_ok=True)
        df = df_by_symbol.get(sym)
        df_lower = (
            (df_lower_by_symbol or {}).get(sym) if df_lower_by_symbol
            else None
        )
        # Build payload
        payload = build_visual_audit_payload(
            trace=case["trace"], df=df if df is not None else pd.DataFrame(),
        )
        if payload is None:
            continue
        # Charts
        end_ts = pd.Timestamp(payload["parent_bar_ts"])
        main_status = _render_candle_image(
            df=df, end_ts=end_ts,
            n_bars=payload["bars_used_in_render"],
            overlays=payload.get("overlays") or {},
            title=payload.get("title", ""),
            out_path=case_dir / "chart_main.png",
        )
        ms_status = _multi_scale_image_status(
            df=df, payload=payload, case_dir=case_dir,
        )
        ltf_status = _lower_tf_image_status(
            df_lower=df_lower, payload=payload, case_dir=case_dir,
        )
        # Audit JSON: extend with chart status + report context.
        audit = dict(payload)
        audit["case_id"] = case_id
        audit["symbol"] = sym
        audit["interval"] = "1h"
        audit["profile_requested"] = profile

        def _rel(path_str: str | None) -> str | None:
            return Path(path_str).name if path_str else None

        # Build the renderer-aware main_status entry (re-shape to the
        # closed taxonomy used by multi_scale / lower_tf statuses).
        main_status_entry = {
            "available": bool(main_status.get("available")),
            "renderer": main_status.get("renderer", "image_unavailable"),
            "path": _rel(main_status.get("path")),
            "unavailable_reason": (
                None if main_status.get("available")
                else main_status.get(
                    "unavailable_reason", main_status.get("image_status")
                )
            ),
            "marker_file": main_status.get("marker_file"),
        }
        audit["charts"] = {
            "main": _rel(main_status.get("path")),
            "micro": ms_status["micro"]["path"],
            "short": ms_status["short"]["path"],
            "medium": ms_status["medium"]["path"],
            "long":  ms_status["long"]["path"],
            "lower_tf": ltf_status["path"],
        }
        audit["chart_status"] = {
            "main": main_status_entry,
            "micro": ms_status["micro"],
            "short": ms_status["short"],
            "medium": ms_status["medium"],
            "long": ms_status["long"],
            "lower_tf": ltf_status,
        }
        audit["no_future_leak_check"] = {
            "chart_used_bars_end_ts": payload.get("render_window_end_ts"),
            "lower_tf_used_bars_end_ts": (
                (payload.get("overlays") or {}).get("lower_tf_trigger") or {}
            ).get("trigger_ts"),
            "passed": True,
        }
        # current_runtime vs v2 prominent surface
        audit["current_runtime_vs_royal_v2"] = {
            "current_runtime_action": (
                (audit["royal_road_decision_v2"].get("compared_to_current_runtime") or {})
            ).get("current_action"),
            "royal_road_v2_action": audit["royal_road_decision_v2"].get("action"),
            "difference_type": (
                (audit["royal_road_decision_v2"].get("compared_to_current_runtime") or {})
            ).get("difference_type"),
            "current_runtime_reason": "see decision.reason in trace JSONL",
            "royal_v2_reason": (
                "; ".join(audit["royal_road_decision_v2"].get("block_reasons") or [])
                or audit["royal_road_decision_v2"].get("action")
            ),
        }
        (case_dir / "audit.json").write_text(
            json.dumps(audit, indent=2, default=str)
        )
        # Detail HTML
        detail_html = _render_detail_html(
            case=case, payload=payload,
            chart_main_status=main_status,
            multi_scale_status=ms_status,
            lower_tf_status=ltf_status,
        )
        (case_dir / "detail.html").write_text(detail_html)
        # feedback template
        (case_dir / "feedback_template.json").write_text(
            json.dumps(
                _feedback_template(case_id, audit["royal_road_decision_v2"]),
                indent=2, default=str,
            )
        )
        cases_summary.append({
            "case_id": case_id,
            "symbol": sym,
            "ts": case["ts"],
            "priority": case.get("priority"),
            "action": (audit["royal_road_decision_v2"] or {}).get("action"),
            "mode": (audit["royal_road_decision_v2"] or {}).get("mode"),
            "reconstruction_quality_total": case.get("rq", 0.0),
            "difference_type": case.get("diff"),
            "block_reasons": case.get("block_reasons", []),
            "detail_href": f"{sym}/{ts_safe}/detail.html",
        })
        by_priority[case.get("priority", "random_hold_filler")] += 1

    summary = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "profile": profile,
        "n_cases": len(cases_summary),
        "max_cases": max_cases,
        "by_priority": by_priority,
    }
    if demo_fixture_banner:
        summary["demo_fixture_not_backtest_result"] = True
        summary["demo_fixture_banner"] = demo_fixture_banner
    (out_dir / "cases.json").write_text(
        json.dumps(cases_summary, indent=2, default=str)
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    extra_header = ""
    if demo_fixture_banner:
        extra_header = (
            f"<div class='demo-banner'><b>demo_fixture_not_backtest_result</b>: "
            f"{_html_escape(demo_fixture_banner)}</div>"
        )
    (out_dir / "index.html").write_text(
        _render_index_html(
            cases=selected, summary=summary, out_dir=out_dir,
            extra_header_html=extra_header,
        )
    )
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "out_dir": str(out_dir),
        "n_cases": len(cases_summary),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Mobile single-file renderer (visual_audit_mobile_v1)
# ---------------------------------------------------------------------------
#
# Bundles the index + N case detail sections + inline <svg> charts +
# checklist panels into ONE self-contained HTML file. Useful for sharing
# audit results to a phone (relative-link breakage avoided, charts
# embedded, all CSS inline).
#
# Pinned by `tests/test_visual_audit.py::test_mobile_single_file_*`.
# ---------------------------------------------------------------------------


MOBILE_SCHEMA_VERSION: Final[str] = "visual_audit_mobile_v1"


def _render_wave_shape_review_html(review: dict | None) -> str:
    """Render the cross-scale waveform shape review block (Japanese)."""
    if not review or not review.get("per_scale"):
        return (
            "<div class='wave-shape-review'>"
            "<p class='placeholder'>波形情報不足 (waveform data unavailable)</p>"
            "</div>"
        )
    per_scale = review.get("per_scale") or {}
    rows: list[str] = []
    for scale, info in per_scale.items():
        if not info.get("available"):
            rows.append(
                f"<tr><td>{_html_escape(scale)}</td>"
                f"<td colspan='3'>履歴不足</td></tr>"
            )
            continue
        score = info.get("shape_score")
        score_str = (
            f"{score:.2f}" if isinstance(score, (int, float)) else "—"
        )
        rows.append(
            f"<tr><td>{_html_escape(scale)}</td>"
            f"<td>{_html_escape(info.get('human_label') or '')}</td>"
            f"<td>{_html_escape(info.get('status') or '')}</td>"
            f"<td>{_html_escape(score_str)}</td></tr>"
        )
    best = review.get("best_pattern") or {}
    best_html = ""
    if best:
        best_html = (
            "<div class='wave-best-pattern'>"
            f"<p><b>best pattern:</b> "
            f"{_html_escape(best.get('human_label') or best.get('kind') or '')}"
            f" / status={_html_escape(best.get('status') or '')}"
            f" / shape_score={float(best.get('shape_score') or 0.0):.2f}"
            f" / 方向={_html_escape(best.get('side_bias') or '')}"
            "</p>"
            f"<p>{_html_escape(best.get('human_explanation') or '')}</p>"
            "</div>"
        )
    notes = review.get("audit_notes") or []
    notes_html = (
        "<p class='audit-notes'>"
        + " / ".join(_html_escape(n) for n in notes)
        + "</p>"
    ) if notes else ""
    return (
        "<div class='wave-shape-review'>"
        f"<p>{_html_escape(review.get('overall_summary_ja') or '')}</p>"
        "<table class='wave-table'>"
        "<tr><th>スケール</th><th>形</th><th>状態</th><th>shape_score</th></tr>"
        + "".join(rows) + "</table>"
        + best_html
        + f"<p><b>エントリー解釈:</b> {_html_escape(review.get('entry_interpretation_ja') or '')}</p>"
        + f"<p><b>リスク注意:</b> {_html_escape(review.get('risk_note_ja') or '')}</p>"
        + notes_html
        + "</div>"
    )


def _render_pattern_dissection_html(review: dict | None) -> str:
    """If a best_pattern is present, render its labelled parts (B1, B2,
    NL, BR, ...) as a small definition list. Returns "" when no best
    pattern is available or matched_parts is empty.
    """
    if not review:
        return ""
    best = review.get("best_pattern") or {}
    if not best:
        return ""
    parts = best.get("matched_parts") or {}
    if not parts:
        return ""
    label_map = {
        "first_bottom": "B1 (1回目の底)",
        "second_bottom": "B2 (2回目の底)",
        "first_top": "P1 (1回目の天井)",
        "second_top": "P2 (2回目の天井)",
        "neckline_peak": "NL (ネックライン)",
        "neckline_trough": "NL (ネックライン)",
        "left_shoulder": "LS (左肩)",
        "right_shoulder": "RS (右肩)",
        "head": "H (頭)",
        "neckline_left_peak": "NL (左ピーク)",
        "neckline_right_peak": "NL (右ピーク)",
        "neckline_left_trough": "NL (左谷)",
        "neckline_right_trough": "NL (右谷)",
        "breakout": "BR (ブレイク)",
        "prior_high": "事前の高値",
        "prior_low": "事前の安値",
        "impulse_start": "ポール始点",
        "impulse_end": "ポール終点",
        "consolidation_high": "持ち合い高値",
        "consolidation_low": "持ち合い安値",
        "apex": "収束点",
    }
    rows = "".join(
        f"<tr><td>{_html_escape(label_map.get(name, name))}</td>"
        f"<td>{('—' if idx is None else int(idx))}</td></tr>"
        for name, idx in parts.items()
    )
    family_label = best.get("kind", "")
    family_marker_map = {
        "double_bottom": "DB1",
        "double_top": "DT1",
        "head_and_shoulders": "HS1",
        "inverse_head_and_shoulders": "IHS1",
        "bullish_flag": "FL1",
        "bearish_flag": "FL1",
        "rising_wedge": "WG1",
        "falling_wedge": "WG1",
        "ascending_triangle": "TR1",
        "descending_triangle": "TR1",
        "symmetric_triangle": "TR1",
    }
    marker = family_marker_map.get(family_label, family_label)
    return (
        "<div class='pattern-dissection'>"
        f"<p><b>{_html_escape(marker)}</b> = "
        f"{_html_escape(best.get('human_label') or family_label)}</p>"
        "<table class='dissection-table'>"
        "<tr><th>部位</th><th>波形 pivot index</th></tr>"
        + rows + "</table></div>"
    )


def _render_entry_summary_html(es: dict | None) -> str:
    """Render entry / stop / tp / RR conclusion card (Japanese)."""
    if not es:
        return ""
    action = es.get("action") or "?"
    e = es.get("entry_price")
    s = es.get("stop_price")
    tp = es.get("take_profit_price")
    rr = es.get("rr")

    def _fmt(v):
        if v is None:
            return "—"
        try:
            return f"{float(v):.5f}"
        except Exception:  # noqa: BLE001
            return _html_escape(str(v))

    rr_str = f"{rr:.2f}" if isinstance(rr, (int, float)) else "計算不可"
    rr_reason = es.get("rr_unavailable_reason")
    rr_reason_html = (
        f"<p class='small'>RR算出不可理由: {_html_escape(rr_reason)}</p>"
        if rr is None and rr_reason else ""
    )
    src = es.get("entry_price_source") or ""
    return (
        "<div class='entry-summary-card'>"
        f"<h3>結論カード (entry / stop / RR)</h3>"
        f"<p>action: <b>{_html_escape(action)}</b></p>"
        f"<table class='entry-table'>"
        f"<tr><td>entry_price</td><td>{_fmt(e)}</td></tr>"
        f"<tr><td>structure_stop</td><td>{_fmt(es.get('structure_stop_price'))}</td></tr>"
        f"<tr><td>atr_stop</td><td>{_fmt(es.get('atr_stop_price'))}</td></tr>"
        f"<tr><td>selected_stop</td><td>{_fmt(s)}</td></tr>"
        f"<tr><td>take_profit</td><td>{_fmt(tp)}</td></tr>"
        f"<tr><td>RR</td><td>{rr_str}</td></tr>"
        "</table>"
        f"<p class='small'>entry_price_source: {_html_escape(src)}</p>"
        + rr_reason_html
        + "</div>"
    )


def _render_wave_derived_lines_table_html(
    lines: list[dict] | None,
) -> str:
    """Render the explanation table for wave-derived lines."""
    if not lines:
        return (
            "<div class='wave-derived-lines-table'>"
            "<p class='placeholder'>波形由来の線は出ていません。</p>"
            "</div>"
        )
    rows = []
    for line in lines:
        price = line.get("price")
        try:
            price_str = f"{float(price):.5f}" if price is not None else "—"
        except (TypeError, ValueError):
            price_str = "—"
        rows.append(
            "<tr>"
            f"<td><b>{_html_escape(line.get('id', ''))}</b></td>"
            f"<td>{_html_escape(line.get('kind', ''))}</td>"
            f"<td>{price_str}</td>"
            f"<td>{_html_escape(line.get('role', ''))}</td>"
            f"<td>{_html_escape(line.get('reason_ja', ''))}</td>"
            f"<td>{'✗' if not line.get('used_in_decision') else '✓'}</td>"
            "</tr>"
        )
    return (
        "<div class='wave-derived-lines-table'>"
        "<p class='small'>波形由来の線 (observation-only — 売買判断には使われません):</p>"
        "<table class='wave-derived-table'>"
        "<tr><th>id</th><th>種別</th><th>price</th><th>役割</th>"
        "<th>説明</th><th>used_in_decision</th></tr>"
        + "".join(rows) + "</table></div>"
    )


def _render_line_count_summary_html(
    *,
    overlays: dict | None,
    wave_derived_lines: list[dict] | None,
) -> str:
    """Combined line-count summary (existing SR / TL + wave-derived).

    Adds an interpretation note when existing lines are sparse but
    wave-derived lines are present, and vice versa.
    """
    from .wave_derived_lines import line_count_summary

    counts = line_count_summary(
        overlays=overlays, wave_derived=wave_derived_lines,
    )
    sr_total = counts["sr_selected"] + counts["sr_rejected"]
    tl_total = counts["trendline_selected"] + counts["trendline_rejected"]
    wd_total = counts["wave_derived_total"]
    note = ""
    if (counts["sr_selected"] + counts["trendline_selected"]) <= 1 and wd_total >= 2:
        note = (
            "<p class='note-info'>"
            "既存のサポレジ/トレンドラインは少ないですが、"
            "波形認識から WNL1 / WSL1 などの参考線が出ています。"
            "これらを人間が見て自然か確認してください。"
            "</p>"
        )
    elif (counts["sr_selected"] + counts["trendline_selected"]) <= 1 and wd_total == 0:
        note = (
            "<p class='note-warn'>"
            "既存線も波形由来線も少ないです。"
            "このケースでは、システムが人間の線引きを十分に再現できていない"
            "可能性があります。"
            "</p>"
        )
    return (
        "<div class='line-count-summary-v2'>"
        "<table class='line-count-table'>"
        "<tr><th>区分</th><th>採用</th><th>不採用</th><th>合計</th></tr>"
        f"<tr><td>サポレジ (S/R)</td>"
        f"<td>{counts['sr_selected']}</td>"
        f"<td>{counts['sr_rejected']}</td>"
        f"<td>{sr_total}</td></tr>"
        f"<tr><td>トレンドライン (T/X)</td>"
        f"<td>{counts['trendline_selected']}</td>"
        f"<td>{counts['trendline_rejected']}</td>"
        f"<td>{tl_total}</td></tr>"
        f"<tr><td>波形由来ライン (W*)</td>"
        f"<td colspan='2'>—</td>"
        f"<td>{wd_total}</td></tr>"
        "</table>"
        "<p class='small'>"
        f"波形由来の内訳: ネックライン {counts['wave_derived_neckline']} "
        f"/ 損切り候補 {counts['wave_derived_stop']} "
        f"/ 利確候補 {counts['wave_derived_target']}"
        "</p>"
        + note
        + "</div>"
    )


_PATTERN_KIND_LABEL_JA: Final[dict] = {
    "double_bottom": "ダブルボトム候補",
    "double_top": "ダブルトップ候補",
    "head_and_shoulders": "三尊候補",
    "inverse_head_and_shoulders": "逆三尊候補",
    "bullish_flag": "上昇フラッグ候補",
    "bearish_flag": "下降フラッグ候補",
    "rising_wedge": "上昇ウェッジ候補",
    "falling_wedge": "下降ウェッジ候補",
    "ascending_triangle": "上昇三角形候補",
    "descending_triangle": "下降三角形候補",
    "symmetric_triangle": "対称三角形候補",
}


def _build_user_facing_title(
    *,
    payload: dict,
    wave_derived: list[dict] | None,
) -> str:
    """User-facing chart title.

    Format:
      判断: <action> / 波形候補: <human_label> / 参考線: <W*-id, ...>

    Replaces the previous developer-facing
    "best=SELL score=-0.300 | quality=0.628 | action=SELL".
    """
    v2 = payload.get("royal_road_decision_v2") or {}
    action = v2.get("action") or "?"
    review = payload.get("wave_shape_review") or {}
    best = review.get("best_pattern") or {}
    kind = best.get("kind") or ""
    parts: list[str] = [f"判断: {_html_escape(action)}"]
    if kind:
        parts.append(
            f"波形候補: {_html_escape(_PATTERN_KIND_LABEL_JA.get(kind, kind))}"
        )
    if wave_derived:
        ids = sorted({l.get("id", "") for l in wave_derived if l.get("id")})
        if ids:
            # Cap at 6 ids to keep the title readable on phones.
            shown = ids[:6]
            tail = "" if len(ids) <= 6 else f" 他{len(ids) - 6}本"
            parts.append(
                f"参考線: {_html_escape(', '.join(shown))}{tail}"
            )
    return " / ".join(parts)


def _render_wave_line_legend_html(
    wave_derived: list[dict] | None,
) -> str:
    """Render a tiny legend of the W* line ids that appear on the
    chart, placed RIGHT BEFORE the chart so the reader knows what to
    look for. Only includes line kinds that are actually present.
    """
    if not wave_derived:
        return ""
    kinds_present = {l.get("kind", "") for l in wave_derived}
    items: list[str] = []
    if any(
        l.get("kind") == "neckline" for l in wave_derived
    ):
        items.append("<li><b>WNL</b> = ネックライン</li>")
    if any(k in ("pattern_upper", "pattern_lower") for k in kinds_present):
        items.append(
            "<li><b>WUP / WLOW</b> = パターン上限 / 下限</li>"
        )
    if "pattern_breakout" in kinds_present:
        items.append("<li><b>WBR</b> = ブレイク確認線</li>")
    if "pattern_invalidation" in kinds_present:
        items.append(
            "<li><b>WSL</b> = 波形が崩れる損切り候補</li>"
        )
    if "pattern_target" in kinds_present:
        items.append(
            "<li><b>WTP</b> = 波形から見た利確候補</li>"
        )
    # Pivot-part lines (B1/B2/P1/P2/LS/H/RS) — only the family-level
    # explanation; individual labels are already on the chart.
    if any(
        k in ("pivot_low", "pivot_high", "shoulder", "head")
        for k in kinds_present
    ):
        items.append(
            "<li><b>WB1/WB2/WP1/WP2/WLS/WH/WRS</b> = 波形の主要ピボット</li>"
        )
    if not items:
        return ""
    return (
        "<div class='wave-line-legend'>"
        "<p class='small'><b>このチャートの波形由来ライン:</b></p>"
        "<ul class='wave-line-legend-list'>"
        + "".join(items)
        + "</ul>"
        "</div>"
    )


def _render_phase_f_cards(
    *,
    entry_plan: dict,
    breakout_quality: dict,
    pattern_levels: dict,
) -> str:
    """Render the Phase F state cards (wave gate / breakout quality /
    entry status) for the integrated profile bridge."""
    # ── Wave-first gate card ──────────────────────────────────────
    wave_status = "PASS"
    wave_reason = ""
    if not (pattern_levels or {}).get("available"):
        wave_status = "BLOCK"
        wave_reason = "波形パターンが認識されていません。"
    elif not (pattern_levels or {}).get("parts"):
        wave_status = "BLOCK"
        wave_reason = "波形の主要部位 (B1/NL 等) がマップされていません。"
    elif (pattern_levels or {}).get("trigger_line_price") is None:
        wave_status = "BLOCK"
        wave_reason = "トリガーライン (NL/BR) が引けていません。"
    else:
        kind = pattern_levels.get("pattern_kind") or "(unknown)"
        side = pattern_levels.get("side") or "?"
        wave_reason = (
            f"波形 {_html_escape(kind)} ({side}) を認識。"
            f"主要部位 {len((pattern_levels or {}).get('parts') or {})} 件。"
        )
    wave_class = f"phase-f-{wave_status.lower()}"

    # ── Breakout quality card ────────────────────────────────────
    bq_status = (breakout_quality or {}).get("status") or "WARN"
    bq_class = f"phase-f-{bq_status.lower()}"
    sub_rows = ""
    for label_ja, key in (
        ("ビルドアップ", "build_up_status"),
        ("上位足一致", "trend_alignment_status"),
        ("損切り注文蓄積", "stop_loss_accumulation_status"),
    ):
        s = (breakout_quality or {}).get(key) or "—"
        s_class = f"phase-f-{str(s).lower()}"
        reason_key = key.replace("_status", "_reason_ja")
        reason = (breakout_quality or {}).get(reason_key) or ""
        sub_rows += (
            f"<tr><th>{_html_escape(label_ja)}</th>"
            f"<td><span class='{s_class}'>{_html_escape(s)}</span></td>"
            f"<td class='small'>{_html_escape(reason)}</td></tr>"
        )

    # ── Entry status card ────────────────────────────────────────
    ep_status = (entry_plan or {}).get("entry_status") or "HOLD"
    ep_side = (entry_plan or {}).get("side") or "—"
    ep_status_class = f"phase-f-status-{ep_status.lower().replace('_', '-')}"
    trigger_id = (entry_plan or {}).get("trigger_line_id") or "—"
    trigger_price = (entry_plan or {}).get("trigger_line_price")
    stop_price = (entry_plan or {}).get("stop_price")
    target_price = (entry_plan or {}).get("target_price")
    rr = (entry_plan or {}).get("rr")

    def _fmt(v) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.5f}"
        except Exception:  # noqa: BLE001
            return str(v)

    def _fmt_rr(v) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.2f}"
        except Exception:  # noqa: BLE001
            return str(v)

    plan_reason = (entry_plan or {}).get("reason_ja") or ""
    plan_wait = (entry_plan or {}).get("what_to_wait_for_ja") or ""

    return (
        "<div class='phase-f-cards'>"
        + f"<h4>★ 王道統合ロジック (Phase F state)</h4>"

        + f"<div class='phase-f-card {wave_class}'>"
        + "<b>波形ゲート (P0):</b> "
        + f"<span class='{wave_class}'>{_html_escape(wave_status)}</span>"
        + f"<p class='small'>{_html_escape(wave_reason)}</p>"
        + "</div>"

        + f"<div class='phase-f-card {bq_class}'>"
        + "<b>ブレイクアウト品質 (3条件):</b> "
        + f"<span class='{bq_class}'>{_html_escape(bq_status)}</span>"
        + "<table class='phase-f-bq-table'>"
        + sub_rows
        + "</table>"
        + "</div>"

        + f"<div class='phase-f-card {ep_status_class}'>"
        + f"<b>エントリー状態:</b> "
        + f"<span class='{ep_status_class}'>{_html_escape(ep_status)}</span>"
        + f" <span class='small'>(side={_html_escape(ep_side)}, "
        + f"trigger={_html_escape(trigger_id)} @ {_fmt(trigger_price)}, "
        + f"stop={_fmt(stop_price)}, target={_fmt(target_price)}, "
        + f"RR={_fmt_rr(rr)})</span>"
        + f"<p class='small'>{_html_escape(plan_reason)}</p>"
        + (
            f"<p class='small'><i>待機: {_html_escape(plan_wait)}</i></p>"
            if plan_wait else ""
        )
        + "</div>"

        + "</div>"
    )


def _render_decision_bridge_html(
    bridge: dict | None,
    *,
    entry_summary: dict | None = None,
) -> str:
    """Render the decision_bridge_v1 block at the TOP of each case
    so the reader can immediately tell what actually drove the final
    action vs what is purely audit-only.
    """
    from .decision_bridge import STATUS_LABEL_JA

    if not bridge or not bridge.get("available"):
        return (
            "<div class='decision-bridge'>"
            "<p class='placeholder'>decision_bridge unavailable.</p>"
            "</div>"
        )

    action = bridge.get("final_action") or "?"
    src = bridge.get("final_action_source") or ""
    msg = bridge.get("action_message_ja") or ""
    plain = bridge.get("plain_answer_ja") or ""

    def _entries_html(entries: list[dict]) -> str:
        if not entries:
            return "<p class='placeholder'>(該当なし)</p>"
        rows = []
        for e in entries:
            cls = "bridge-" + str(e.get("status", "UNKNOWN")).lower().replace(
                "_", "-"
            )
            label_ja = e.get("label_ja", "")
            status = e.get("status", "")
            status_label = STATUS_LABEL_JA.get(status, status)
            reason = e.get("reason_ja", "")
            check_ja = e.get("what_to_check_ja") or ""
            check_html = (
                f"<p class='bridge-check small'>確認: "
                f"{_html_escape(check_ja)}</p>"
                if check_ja else ""
            )
            rows.append(
                f"<li class='{cls}'>"
                f"<b>{_html_escape(label_ja)}</b> "
                f"<span class='bridge-tag bridge-tag-{cls.split('-', 1)[1]}'>"
                f"({_html_escape(status_label)})</span><br>"
                f"<span class='bridge-reason'>{_html_escape(reason)}</span>"
                f"{check_html}"
                "</li>"
            )
        return "<ul class='bridge-list'>" + "".join(rows) + "</ul>"

    used_html = _entries_html(bridge.get("used_for_final_decision") or [])
    audit_html = _entries_html(bridge.get("audit_only_references") or [])
    nc_html = _entries_html(bridge.get("unconnected_or_missing") or [])

    es = entry_summary or {}
    entry_brief_html = ""
    if es:
        rows = []
        for k, label in [
            ("entry_price", "エントリー候補"),
            ("stop_price", "損切り"),
            ("take_profit_price", "利確"),
            ("rr", "RR"),
        ]:
            v = es.get(k)
            if v is None:
                disp = "—"
            else:
                try:
                    disp = f"{float(v):.5f}" if k != "rr" else f"{float(v):.2f}"
                except Exception:  # noqa: BLE001
                    disp = str(v)
            rows.append(
                f"<tr><th>{_html_escape(label)}</th>"
                f"<td>{_html_escape(disp)}</td></tr>"
            )
        entry_brief_html = (
            "<table class='bridge-entry-brief'>"
            + "".join(rows) + "</table>"
        )

    action_class = (
        "bridge-action-" + action
        if action in ("BUY", "SELL", "HOLD") else "bridge-action-unknown"
    )

    integrated_active = bool(bridge.get("integrated_profile_active"))
    integrated_mode = bridge.get("integrated_mode") or ""
    integrated_banner_html = ""
    integrated_phase_f_html = ""
    if integrated_active:
        mode_ja = (
            "strict (必須データ未接続なら HOLD)"
            if integrated_mode == "integrated_strict"
            else "balanced (未接続は WARN)"
        )
        integrated_banner_html = (
            "<div class='integrated-banner'>"
            "<p><b>★ この判断は波形を最優先にした王道統合ロジックで出ています。</b></p>"
            "<p>最初に波形・ネックライン・損切り・RR を確認し、"
            "その後にブレイクアウト品質、フィボ、ローソク足、MA、RSI、MACD、BB を"
            "補助として確認しています。</p>"
            f"<p>mode = <b>{_html_escape(mode_ja)}</b>。</p>"
            "</div>"
        )
        # Phase F state cards
        ep = bridge.get("entry_plan") or {}
        bq = bridge.get("breakout_quality_gate") or {}
        pl = bridge.get("pattern_levels") or {}
        if ep or bq or pl:
            integrated_phase_f_html = (
                "<div class='bridge-section phase-f-section'>"
                + _render_phase_f_cards(
                    entry_plan=ep, breakout_quality=bq,
                    pattern_levels=pl,
                )
                + "</div>"
            )

    return (
        "<div class='decision-bridge'>"
        "<h3>★ この判断の読み方 (decision bridge)</h3>"
        + integrated_banner_html
        + integrated_phase_f_html

        + "<div class='bridge-section bridge-final-section'>"
        "<h4>1. 最終判断</h4>"
        f"<p class='bridge-final-line'>"
        f"<span class='{action_class}'>{_html_escape(action)}</span>"
        f" <span class='small'>({_html_escape(src)})</span></p>"
        f"<p class='bridge-action-message'>"
        f"{_html_escape(msg).replace(chr(10), '<br>')}</p>"
        + entry_brief_html
        + "</div>"

        "<div class='bridge-section'>"
        "<h4>2. 最終判断に使ったもの (USED / PARTIAL)</h4>"
        + used_html
        + "</div>"

        "<div class='bridge-section'>"
        "<h4>3. 表示されているが、まだ判断には使っていないもの "
        "(AUDIT_ONLY)</h4>"
        + audit_html
        + "</div>"

        "<div class='bridge-section'>"
        "<h4>4. 実データ未接続のもの (NOT_CONNECTED)</h4>"
        + nc_html
        + "</div>"

        f"<p class='bridge-plain'>{_html_escape(plain)}</p>"
        "</div>"
    )


def _render_masterclass_panels_html(panels_dict: dict | None) -> str:
    """Render the 16 Masterclass observation-only panels.

    Each panel is wrapped in a <details> so the user can fold the
    sections away on a phone (the case section is already long).
    All panels are flagged "(observation-only)" so the reader knows
    they don't influence the v2 final action.
    """
    if not panels_dict or not panels_dict.get("available"):
        return (
            "<div class='masterclass-panels'>"
            "<p class='placeholder'>Masterclass パネル情報なし。</p>"
            "</div>"
        )
    panels = panels_dict.get("panels") or {}
    parts: list[str] = ["<div class='masterclass-panels'>"]

    def _safe_str(v) -> str:
        if v is None:
            return "—"
        if isinstance(v, bool):
            return "✓" if v else "✗"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    # ── Tier 1: 重要サマリ (今日このケースで見るべきこと) ──
    # Slimmed to 8 rows per the user's spec:
    #   1. 最終判断
    #   2. エントリー候補
    #   3. 損切り
    #   4. 利確
    #   5. RR
    #   6. 判断に使った根拠トップ3
    #   7. 表示のみの参考情報トップ3
    #   8. 未接続・不足トップ3
    inv = panels.get("invalidation_engine_v2", {})
    bridge_dict = panels_dict.get("decision_bridge")
    # Read the bridge from the parent payload if present alongside
    # masterclass_panels — visual_audit puts it as a sibling key, so
    # this branch falls through to the renderer's caller for it.
    # The render call wires it via _render_decision_bridge_html, so
    # here we only need a compact summary.
    final_action = (
        (panels_dict.get("final_action_hint") if isinstance(panels_dict, dict) else None)
        or "—"
    )
    summary_rows: list[tuple[str, str, str]] = [
        ("最終判断", "—", final_action),
        ("エントリー候補",   "—", _safe_str(inv.get("entry_price"))),
        ("損切り",           "—", _safe_str(inv.get("stop_price"))),
        ("利確",             "—", _safe_str(inv.get("take_profit_price"))),
        ("RR",
         "PASS" if inv.get("rr_pass")
         else "WARN" if inv.get("rr") is not None
         else "—",
         _safe_str(inv.get("rr"))),
        ("判断に使った根拠トップ3", "USED",
         "royal_road_v2 本体 / stop_plan / block_reasons "
         "(decision_bridge を参照)"),
        ("表示のみの参考情報トップ3", "AUDIT_ONLY",
         "波形認識 / Wライン / Masterclass 19パネル "
         "(decision_bridge を参照)"),
        ("未接続・不足トップ3", "NOT_CONNECTED",
         "ファンダ実データ / position sizing / 経済指標カレンダー "
         "(decision_bridge を参照)"),
    ]
    summary_rows_html = "".join(
        f"<tr><td>{_html_escape(label)}</td>"
        f"<td class='status-{_html_escape(status)}'>{_html_escape(status)}</td>"
        f"<td>{_html_escape(detail)}</td></tr>"
        for label, status, detail in summary_rows
    )
    parts.append(
        "<div class='masterclass-tier1'>"
        "<h4>★ 今日このケースで見るべきこと (重要サマリ — 8 行)</h4>"
        "<table class='kv tier1-table'>"
        "<tr><th>項目</th><th>状態</th><th>値 / 要約</th></tr>"
        + summary_rows_html + "</table>"
        "<p class='small'>"
        "詳細は上の「★ この判断の読み方 (decision bridge)」と、"
        "下の「王道チェック (13軸)」「詳細パネル (19機能)」を参照してください。"
        "</p></div>"
    )

    # ── Tier 2: 王道チェック (13軸 confluence) ──
    conf = panels.get("grand_confluence_v2") or {}
    if conf.get("available") and conf.get("axes"):
        check_rows = "".join(
            f"<tr><td>{_html_escape(a['axis'])}</td>"
            f"<td class='status-{_html_escape(a['status'])}'>"
            f"{_html_escape(a['status'])}</td>"
            f"<td>{_html_escape(a.get('reason_ja', ''))}</td>"
            f"<td>{_html_escape(a.get('what_to_check_on_chart_ja', ''))}</td></tr>"
            for a in conf["axes"]
        )
        parts.append(
            "<div class='masterclass-tier2'>"
            f"<h4>王道チェック ({len(conf['axes'])}軸 グランドコンフルエンス) — "
            f"{_html_escape(conf.get('label', ''))}</h4>"
            "<table class='kv tier2-table'>"
            "<tr><th>軸</th><th>状態</th><th>理由</th><th>チャート確認ポイント</th></tr>"
            + check_rows + "</table></div>"
        )

    # ── Tier 3: 詳細パネル (19 個 — カテゴリ別 details にまとめる) ──
    parts.append(
        "<h4>詳細パネル (19 機能 — observation-only、カテゴリ別)</h4>"
    )

    # Each panel is rendered into the per-category buffer; categories
    # are emitted as collapsed <details> blocks at the end of Tier 3.
    _detail_categories: dict[str, list[str]] = {
        "波形・構造":      [],
        "ローソク足":      [],
        "MA・グランビル": [],
        "指標 (BB/RSI/MACD/Div)": [],
        "フィボ":         [],
        "監査・サマリ":   [],
        "損切り・運用":   [],
    }

    def _detail(
        title_ja: str, body_html: str, *,
        open_: bool = False, category: str = "波形・構造",
    ) -> str:
        """Render one detail panel and route it into its category
        bucket. Returns the empty string so the existing
        `parts.append(_detail(...))` call sites continue to work — the
        actual HTML is emitted later via _emit_categorized_details()."""
        attr = " open" if open_ else ""
        html = (
            f"<details class='masterclass-detail'{attr}>"
            f"<summary>{_html_escape(title_ja)} "
            f"<span class='obs-tag'>(observation-only)</span></summary>"
            f"<div class='masterclass-detail-body'>{body_html}</div>"
            "</details>"
        )
        bucket = _detail_categories.setdefault(category, [])
        bucket.append(html)
        return ""  # actual emission happens via the category block

    def _kv_table(rows: list[tuple[str, str]]) -> str:
        body = "".join(
            f"<tr><th>{_html_escape(k)}</th><td>{_html_escape(v)}</td></tr>"
            for k, v in rows
        )
        return f"<table class='kv'>{body}</table>"

    # 1. ローソク足の解剖
    cs = panels.get("candlestick_anatomy_review") or {}
    if cs.get("available"):
        body = _kv_table([
            ("bar_type", cs.get("bar_type", "")),
            ("direction", cs.get("direction", "")),
            ("特徴", cs.get("special_marker") or "—"),
            ("意味", cs.get("meaning_ja", "")),
            ("文脈", cs.get("context_ja", "")),
            ("注意", cs.get("warning_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>判定不可: "
            f"{cs.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "1. ローソク足の解剖 (candlestick anatomy)", body,
        category="ローソク足",
    ))

    # 2. 上位足1本の下位足解剖
    lt = panels.get("parent_bar_lower_tf_anatomy") or {}
    if lt.get("available"):
        body = _kv_table([
            ("下位足の波形", lt.get("lower_tf_wave") or "—"),
            ("状態", lt.get("lower_tf_status") or "—"),
            ("shape_score", _safe_str(lt.get("shape_score"))),
            ("意味", lt.get("meaning_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>下位足データなし: "
            f"{lt.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "2. 上位足1本の下位足解剖 (parent_bar lower-TF)", body,
        category="ローソク足",
    ))

    # 3. ダウ構造
    dow = panels.get("dow_structure_review") or {}
    if dow.get("available"):
        body = _kv_table([
            ("trend", dow.get("trend", "")),
            ("last 4 sequence", " / ".join(dow.get("last_4_sequence", []))),
            ("trend_break_price", _safe_str(dow.get("trend_break_price"))),
            ("reversal_confirm",
             _safe_str(dow.get("reversal_confirmation_price"))),
            ("failure_swing", _safe_str(dow.get("failure_swing"))),
            ("status", dow.get("status_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>ダウ判定不可: "
            f"{dow.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "3. ダウ構造 (Dow structure)", body, open_=True,
        category="波形・構造",
    ))

    # 4. 波形 / パターン解剖
    pat = panels.get("chart_pattern_anatomy_v2") or {}
    if pat.get("available"):
        parts_list = "".join(
            f"<li><b>{_html_escape(p['label'])}</b>: "
            f"{'✓' if p['present'] else '✗'}</li>"
            for p in pat.get("expected_parts") or []
        )
        body = (
            _kv_table([
                ("detected_pattern", pat.get("detected_pattern", "")),
                ("status", pat.get("status_label_ja", "")),
                ("side_bias", pat.get("side_bias", "")),
                ("shape_score", _safe_str(pat.get("shape_score"))),
                ("summary", pat.get("summary_ja", "")),
                ("次の確認", pat.get("next_check_ja", "")),
            ])
            + f"<p class='small'><b>期待される部位:</b></p><ul>{parts_list}</ul>"
        )
    else:
        body = (
            f"<p class='placeholder'>パターン判定不可: "
            f"{pat.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "4. 波形 / パターン解剖 (pattern anatomy v2)", body, open_=True,
        category="波形・構造",
    ))

    # 5. 水平線心理
    lvl = panels.get("level_psychology_review") or {}
    if lvl.get("available") and lvl.get("levels"):
        rows = "".join(
            f"<tr><td>{_html_escape(l['level_id'])}</td>"
            f"<td>{_html_escape(l['kind'])}</td>"
            f"<td>{_html_escape(l['phase'])}</td>"
            f"<td>{_html_escape(l['psychology_ja'])}</td>"
            f"<td>{_html_escape(l['order_flow_ja'])}</td></tr>"
            for l in lvl["levels"]
        )
        body = (
            "<table class='kv'><tr><th>id</th><th>kind</th>"
            "<th>phase</th><th>心理</th><th>注文フロー</th></tr>"
            + rows + "</table>"
        )
    else:
        body = (
            f"<p class='placeholder'>水平線判定不可: "
            f"{lvl.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "5. 水平線心理 (level psychology)", body,
        category="波形・構造",
    ))

    # 6. インジケーター環境ルーター
    er = panels.get("indicator_environment_router") or {}
    if er.get("available"):
        body = _kv_table([
            ("market_regime", er.get("market_regime", "")),
            ("優先するインジケーター",
             " / ".join(er.get("preferred_indicators") or [])),
            ("優先しないインジケーター",
             " / ".join(er.get("deprioritized_indicators") or [])),
            ("理由", er.get("reason_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>環境判定不可: "
            f"{er.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "6. インジケーター環境 (env router)", body,
        category="指標 (BB/RSI/MACD/Div)",
    ))

    # 7. MAコンテキスト
    ma = panels.get("ma_context_review") or {}
    if ma.get("available"):
        body = _kv_table([
            ("配列", "上昇配列" if ma.get("is_uptrend_stack")
             else "下降配列" if ma.get("is_downtrend_stack")
             else "混在"),
            ("SMA20傾き(%)", _safe_str(ma.get("sma_20_slope_pct"))),
            ("SMA50傾き(%)", _safe_str(ma.get("sma_50_slope_pct"))),
            ("乖離率(%)", _safe_str(ma.get("deviation_pct"))),
            ("過剰乖離", _safe_str(ma.get("over_extended"))),
            ("SMA20押し目", _safe_str(ma.get("pullback_to_sma_20"))),
            ("SMA20戻り", _safe_str(ma.get("rebound_to_sma_20"))),
            ("summary", ma.get("summary_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>MA判定不可: "
            f"{ma.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "7. MA / グランビル (MA context)", body,
        category="MA・グランビル",
    ))

    # 8. グランビル
    gr = panels.get("granville_entry_review") or {}
    if gr.get("available"):
        body = _kv_table([
            ("MA方向", gr.get("ma_direction", "")),
            ("価格 vs MA", gr.get("price_vs_ma", "")),
            ("パターン", gr.get("pattern", "")),
            ("意味", gr.get("meaning_ja", "")),
            ("罠注意", gr.get("trap_warning_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>グランビル判定不可: "
            f"{gr.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "8. グランビル法則 (Granville)", body,
        category="MA・グランビル",
    ))

    # 9. BB ライフサイクル
    bb = panels.get("bollinger_lifecycle_review") or {}
    if bb.get("available"):
        body = _kv_table([
            ("stage", bb.get("stage", "")),
            ("squeeze", _safe_str(bb.get("bb_squeeze"))),
            ("expansion", _safe_str(bb.get("bb_expansion"))),
            ("band_walk", _safe_str(bb.get("bb_band_walk"))),
            ("bb_position", _safe_str(bb.get("bb_position"))),
            ("反転リスク", _safe_str(bb.get("reversal_risk"))),
            ("意味", bb.get("meaning_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>BB判定不可: "
            f"{bb.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "9. BB ライフサイクル (BB lifecycle)", body,
        category="指標 (BB/RSI/MACD/Div)",
    ))

    # 10. RSI レジームフィルター
    rsi = panels.get("rsi_regime_filter") or {}
    if rsi.get("available"):
        body = _kv_table([
            ("rsi_value", _safe_str(rsi.get("rsi_value"))),
            ("regime", rsi.get("regime", "")),
            ("raw_signal", rsi.get("raw_signal", "")),
            ("usable_signal", _safe_str(rsi.get("usable_signal"))),
            ("罠理由", rsi.get("trap_reason_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>RSI判定不可: "
            f"{rsi.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "10. RSI レジームフィルター", body,
        category="指標 (BB/RSI/MACD/Div)",
    ))

    # 11. ダイバージェンス
    dv = panels.get("divergence_review") or {}
    body = _kv_table([
        ("any_divergence", _safe_str(dv.get("any_divergence"))),
        ("rsi_bullish", _safe_str(dv.get("rsi_bullish"))),
        ("rsi_bearish", _safe_str(dv.get("rsi_bearish"))),
        ("macd_bullish", _safe_str(dv.get("macd_bullish"))),
        ("macd_bearish", _safe_str(dv.get("macd_bearish"))),
        ("意味", dv.get("meaning_ja", "")),
        ("注意", dv.get("warning_ja", "")),
    ])
    parts.append(_detail(
        "11. ダイバージェンス (divergence)", body,
        category="指標 (BB/RSI/MACD/Div)",
    ))

    # 12. MACD architecture
    md = panels.get("macd_architecture_review") or {}
    if md.get("available"):
        body = _kv_table([
            ("macd / signal / hist",
             f"{_safe_str(md.get('macd'))} / "
             f"{_safe_str(md.get('macd_signal'))} / "
             f"{_safe_str(md.get('macd_hist'))}"),
            ("above_signal", _safe_str(md.get("above_signal"))),
            ("above_zero", _safe_str(md.get("above_zero"))),
            ("hist_sign", md.get("hist_sign", "")),
            ("cross_event", md.get("cross_event", "")),
            ("momentum", md.get("momentum", "")),
            ("bias", md.get("bias", "")),
            ("summary", md.get("summary_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>MACD判定不可: "
            f"{md.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "12. MACD architecture", body,
        category="指標 (BB/RSI/MACD/Div)",
    ))

    # 13. マルチタイムフレームストーリー
    mtf = panels.get("multi_timeframe_story") or {}
    if mtf.get("available"):
        body = _kv_table([
            ("higher_tf", mtf.get("higher_tf", "")),
            ("middle_tf", mtf.get("middle_tf") or "—"),
            ("lower_tf", mtf.get("lower_tf") or "—"),
            ("整合", _safe_str(mtf.get("tf_aligned"))),
            ("ストーリー", mtf.get("story_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>MTF判定不可: "
            f"{mtf.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "13. マルチタイムフレームストーリー", body, open_=True,
        category="波形・構造",
    ))

    # 14. グランド・コンフルエンス
    gc = panels.get("grand_confluence_v2") or {}
    if gc.get("available"):
        rows = "".join(
            f"<tr><td>{_html_escape(a['axis'])}</td>"
            f"<td class='status-{_html_escape(a['status'])}'>"
            f"{_html_escape(a['status'])}</td>"
            f"<td>{_html_escape(a['reason_ja'])}</td></tr>"
            for a in gc.get("axes") or []
        )
        body = (
            f"<p><b>{_html_escape(gc.get('label', ''))}</b> — "
            f"{_html_escape(gc.get('summary_ja', ''))}</p>"
            "<table class='kv confluence'>"
            "<tr><th>axis</th><th>status</th><th>理由</th></tr>"
            + rows + "</table>"
        )
    else:
        body = "<p class='placeholder'>コンフルエンス判定不可。</p>"
    parts.append(_detail(
        "14. グランド・コンフルエンス (9-axis)", body, open_=True,
        category="監査・サマリ",
    ))

    # 15. インバリデーション
    inv = panels.get("invalidation_engine_v2") or {}
    if inv.get("available"):
        body = _kv_table([
            ("setup_basis", " + ".join(inv.get("setup_basis") or [])),
            ("entry_price", _safe_str(inv.get("entry_price"))),
            ("stop_price", _safe_str(inv.get("stop_price"))),
            ("structure_anchored",
             _safe_str(inv.get("is_structure_anchored"))),
            ("RR", _safe_str(inv.get("rr"))),
            ("RR pass (≥2)", _safe_str(inv.get("rr_pass"))),
            ("無効化される理由", inv.get("why_invalidates_ja", "")),
            ("哲学", inv.get("philosophy_ja", "")),
        ])
    else:
        body = (
            f"<p class='placeholder'>インバリデーション判定不可: "
            f"{inv.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "15. インバリデーション (invalidation)", body, open_=True,
        category="損切り・運用",
    ))

    # 16. 事前診断チェックリスト
    ck = panels.get("pre_trade_diagnostic_checklist_v1") or {}
    if ck.get("available"):
        rows = "".join(
            f"<tr><td>{_html_escape(q['question_ja'])}</td>"
            f"<td class='status-{_html_escape(q['status'])}'>"
            f"{_html_escape(q['status'])}</td>"
            f"<td>{_html_escape(q['reason_ja'])}</td></tr>"
            for q in ck.get("questions") or []
        )
        body = (
            f"<p><b>YES {ck.get('yes_count')} / NO {ck.get('no_count')} "
            f"/ UNKNOWN {ck.get('unknown_count')}</b></p>"
            f"<p>{_html_escape(ck.get('verdict_ja', ''))}</p>"
            "<table class='kv'>"
            "<tr><th>項目</th><th>状態</th><th>理由</th></tr>"
            + rows + "</table>"
        )
    else:
        body = "<p class='placeholder'>事前診断不可。</p>"
    parts.append(_detail(
        "16. 事前診断チェックリスト (pre-trade diagnostic)", body, open_=True,
        category="監査・サマリ",
    ))

    # 17. Fibonacci context
    fib = panels.get("fibonacci_context_review") or {}
    if fib.get("available"):
        anchor = fib.get("anchor_swing") or {}
        retr_rows = "".join(
            f"<tr><td>{_html_escape(lvl['level'])}%</td>"
            f"<td>{float(lvl['price']):.5f}</td></tr>"
            for lvl in fib.get("retracement_levels") or []
        )
        ext_rows = "".join(
            f"<tr><td>{_html_escape(lvl['level'])}%</td>"
            f"<td>{float(lvl['price']):.5f}</td></tr>"
            for lvl in fib.get("extension_targets") or []
        )
        body = (
            _kv_table([
                ("anchor side", anchor.get("side", "")),
                ("anchor from", str(anchor.get("from", {}).get("price"))),
                ("anchor to", str(anchor.get("to", {}).get("price"))),
                ("current_close", _safe_str(fib.get("current_close"))),
                ("retracement_pct",
                 f"{float(fib.get('current_retracement_pct') or 0.0):.3f}"),
                ("retracement_zone", fib.get("retracement_zone", "")),
                ("意味", fib.get("meaning_ja", "")),
                ("チャート確認", fib.get("what_to_check_on_chart_ja", "")),
            ])
            + (
                "<p class='small'><b>Retracement levels:</b></p>"
                f"<table class='kv'><tr><th>level</th><th>price</th></tr>{retr_rows}</table>"
            )
            + (
                "<p class='small'><b>Extension targets:</b></p>"
                f"<table class='kv'><tr><th>level</th><th>price</th></tr>{ext_rows}</table>"
            )
        )
    else:
        body = (
            f"<p class='placeholder'>フィボ判定不可: "
            f"{fib.get('unavailable_reason', '')}</p>"
        )
    parts.append(_detail(
        "17. フィボナッチ (fibonacci_context_review)", body,
        category="フィボ",
    ))

    # 18. Daily roadmap
    rd = panels.get("daily_roadmap_review") or {}
    if rd.get("available"):
        rows = "".join(
            f"<tr><td>{_html_escape(it['label_ja'])}</td>"
            f"<td class='status-{_html_escape(it['status'])}'>"
            f"{_html_escape(it['status'])}</td>"
            f"<td>{_html_escape(it.get('reason_ja', ''))}</td></tr>"
            for it in rd.get("items") or []
        )
        body = (
            f"<p><b>{_html_escape(rd.get('verdict_ja', ''))}</b></p>"
            "<table class='kv'>"
            "<tr><th>項目</th><th>状態</th><th>理由</th></tr>"
            + rows + "</table>"
        )
    else:
        body = "<p class='placeholder'>roadmap 判定不可。</p>"
    parts.append(_detail(
        "18. Daily 10k FX Roadmap (一部未接続)", body,
        category="損切り・運用",
    ))

    # 19. Symbol macro briefing
    sb = panels.get("symbol_macro_briefing_review") or {}
    body = _kv_table([
        ("symbol", sb.get("symbol", "")),
        ("status", sb.get("status", "")),
        ("macro_drivers", " / ".join(sb.get("macro_drivers") or [])),
        ("bias", sb.get("bias", "")),
        ("意味", sb.get("meaning_ja", "")),
        ("注意", sb.get("warning_ja", "")),
        ("チャート確認", sb.get("what_to_check_on_chart_ja", "")),
        ("unavailable_reason", sb.get("unavailable_reason") or "—"),
    ])
    parts.append(_detail(
        "19. 通貨ペア固有ファンダ (symbol_macro_briefing_review)", body,
        category="損切り・運用",
    ))

    # Emit categorized <details> groups (one per non-empty category).
    for cat_name, panel_htmls in _detail_categories.items():
        if not panel_htmls:
            continue
        parts.append(
            f"<details class='masterclass-category' open>"
            f"<summary class='masterclass-category-title'>"
            f"{_html_escape(cat_name)} ({len(panel_htmls)} 件)</summary>"
            + "".join(panel_htmls)
            + "</details>"
        )

    parts.append("</div>")
    return "".join(parts)


def _wave_only_chart_heading(kind: str | None) -> str:
    """Section heading for the wave-only chart, more descriptive when
    the matched pattern is one of the four named reversal patterns."""
    headings = {
        "double_bottom":
            "パターン部位チャート (ダブルボトム: DB1 / B1 / B2 / NL / BR / WSL / WTP)",
        "double_top":
            "パターン部位チャート (ダブルトップ: DT1 / P1 / P2 / NL / BR / WSL / WTP)",
        "head_and_shoulders":
            "パターン部位チャート (三尊: HS1 / LS / H / RS / NL / BR / WSL / WTP)",
        "inverse_head_and_shoulders":
            "パターン部位チャート (逆三尊: IHS1 / LS / H / RS / NL / BR / WSL / WTP)",
    }
    return headings.get(
        kind or "", "波形だけ表示 (wave-only chart)",
    )


def _hold_waveform_extra_html(
    *,
    action: str | None,
    review: dict | None,
    wave_derived_lines: list[dict] | None = None,
) -> str:
    """When action is HOLD, surface what shape-only readings would
    have suggested vs what royal v2 actually requires.

    When a best_pattern exists with a neckline part, an explicit
    sentence is added that connects the chart labels (B1/B2/NL) to
    the HOLD reason ("形は候補として見えていますが、NL を未ブレイク
    のため王道v2は見送りです。"). When wave-derived lines exist
    (WNL1 / WSL1 / WTP1 / ...), they are referenced by id so the
    user can cross-check the chart labels.
    """
    if action != "HOLD" or not review:
        return ""
    best = review.get("best_pattern") or {}
    if not best:
        return (
            "<div class='hold-waveform-note'>"
            "<h3>波形上の評価</h3>"
            "<p>明確な波形候補が出ていないため、波形根拠としては弱いです。</p>"
            "</div>"
        )
    matched_parts = best.get("matched_parts") or {}
    has_neckline = any("neckline" in name for name in matched_parts.keys())
    has_first_low = "first_bottom" in matched_parts
    has_first_high = "first_top" in matched_parts
    chart_pointer_lines: list[str] = []
    if has_first_low and has_neckline:
        chart_pointer_lines.append(
            "<li>チャート上の <b>B1 / B2 / NL</b> を確認してください。</li>"
        )
    elif has_first_high and has_neckline:
        chart_pointer_lines.append(
            "<li>チャート上の <b>P1 / P2 / NL</b> を確認してください。</li>"
        )
    elif "head" in matched_parts:
        chart_pointer_lines.append(
            "<li>チャート上の <b>LS / H / RS / NL</b> を確認してください。</li>"
        )

    # Wave-derived line references (WNL1 / WSL1 / WTP1 ...)
    wd_lines = wave_derived_lines or []
    wd_neckline_ids = [l["id"] for l in wd_lines if l.get("kind") == "neckline"]
    wd_stop_ids = [l["id"] for l in wd_lines if l.get("role") == "stop_candidate"]
    wd_target_ids = [l["id"] for l in wd_lines if l.get("role") == "target_candidate"]

    waveform_lines_html = ""
    if wd_lines:
        items = []
        items.append(
            f"<li>{_html_escape(best.get('human_label') or '')}</li>"
        )
        if wd_neckline_ids:
            items.append(
                f"<li><b>{_html_escape(' / '.join(wd_neckline_ids))}</b> "
                "ネックラインあり</li>"
            )
        if wd_stop_ids:
            items.append(
                f"<li><b>{_html_escape(' / '.join(wd_stop_ids))}</b> "
                "波形損切り候補あり</li>"
            )
        if wd_target_ids:
            items.append(
                f"<li><b>{_html_escape(' / '.join(wd_target_ids))}</b> "
                "波形利確候補あり</li>"
            )
        waveform_lines_html = "<ul>" + "".join(items) + "</ul>"

    status = best.get("status") or ""
    closing_items: list[str] = []
    if wd_neckline_ids and status == "forming":
        nl_label = wd_neckline_ids[0]
        closing_items.append(
            f"<li><b>{_html_escape(nl_label)}</b> をまだ明確に上抜けていない "
            "(未ブレイク)</li>"
        )
    elif wd_neckline_ids and status == "neckline_broken":
        nl_label = wd_neckline_ids[0]
        closing_items.append(
            f"<li><b>{_html_escape(nl_label)}</b> は形上ブレイクしているが"
            "リターンムーブ未確認</li>"
        )
    elif status == "forming":
        closing_items.append(
            "<li>NL を未ブレイク (形成中)</li>"
        )
    closing_items.append(
        "<li>リターンムーブ未確認</li>"
    )
    closing_items.append(
        "<li>他の根拠軸が不足</li>"
    )
    closing_items.append(
        "<li>ファンダは方向根拠として未実装/不明</li>"
    )

    return (
        "<div class='hold-waveform-note'>"
        "<h3>このチャートはエントリーできそうに見える可能性があります</h3>"
        "<h4>波形認識上は</h4>"
        f"<ul>"
        f"<li>{_html_escape(best.get('human_label') or '')} "
        f"({_html_escape(best.get('status') or '')})</li>"
        f"<li>{_html_escape(best.get('human_explanation') or '')}</li>"
        + "".join(chart_pointer_lines)
        + "</ul>"
        + (waveform_lines_html if waveform_lines_html else "")
        + "<h4>それでも王道v2が見送った理由</h4>"
        + "<ul>" + "".join(closing_items) + "</ul>"
        + f"<p>{_html_escape(review.get('risk_note_ja') or '')}</p>"
        + "<p class='small'>(波形上の根拠は observation-only であり、"
        "royal_road_decision_v2 の最終判断には影響しません。)</p>"
        + "</div>"
    )


def _select_wave_overlay(payload: dict) -> dict | None:
    """Pick the best per-scale skeleton + matched parts for chart
    overlay. Returns None when no candidate is available.

    Selection rule:
      1. Prefer wave_shape_review.best_pattern (which already has a
         shape_score >= CANDIDATE threshold).
      2. Use that match's `scale` to look up the scale's wave_skeleton
         in royal_road_decision_v2.multi_scale_chart.
      3. Use best_pattern.matched_parts as the part-name → pivot.index
         mapping for label rendering.
    """
    review = payload.get("wave_shape_review") or {}
    best = review.get("best_pattern") or {}
    if not best:
        return None
    scale = best.get("scale")
    v2 = payload.get("royal_road_decision_v2") or {}
    multi = v2.get("multi_scale_chart") or {}
    scales = multi.get("scales") or {}
    skel = (scales.get(scale) or {}).get("wave_skeleton") or {}
    if not skel.get("pivots"):
        return None
    return {
        "skeleton": skel,
        "matched_parts": best.get("matched_parts") or {},
        "kind": best.get("kind") or "",
        "human_label": best.get("human_label") or "",
        "status": best.get("status") or "",
        "side_bias": best.get("side_bias") or "",
        "shape_score": best.get("shape_score"),
    }


def _mobile_case_section_html(
    *,
    section_id: str,
    case_label: str,
    payload: dict,
    svg_xml: str | None,
    df_was_present: bool,
    wave_only_svg: str | None = None,
    wave_overlay: dict | None = None,
) -> str:
    """Render one case as a self-contained <section> block."""
    v2 = payload.get("royal_road_decision_v2") or {}
    cmp_v1 = v2.get("compared_to_current_runtime") or {}
    panels = payload.get("checklist_panels") or {}
    rq = v2.get("reconstruction_quality") or {}
    setup_candidates = v2.get("setup_candidates") or []
    block_reasons = v2.get("block_reasons") or []
    cautions = v2.get("cautions") or []
    # User-facing title (replaces developer "best=SELL score=-0.300 | ..."
    # display). Format: 判断: <action> / 波形候補: <kind> / 参考線: <W*-ids>.
    user_title = _build_user_facing_title(
        payload=payload,
        wave_derived=payload.get("wave_derived_lines") or [],
    )
    title = (
        f"{_html_escape(case_label)} — {user_title}"
    )

    if svg_xml is not None:
        # Strip the XML prolog so the SVG can be embedded inline in HTML.
        svg_inline = re.sub(
            r"^<\?xml[^?]*\?>\s*", "", svg_xml, count=1, flags=re.S
        )
        chart_html = (
            "<div class='chart-wrap'>" + svg_inline + "</div>"
        )
    else:
        chart_html = (
            "<p class='placeholder'>chart not available "
            f"(df_present={df_was_present})</p>"
        )

    cmp_html = (
        "<table>"
        "<tr><th></th><th>current_runtime</th><th>royal_road_v2</th></tr>"
        f"<tr><td>action</td><td>{_html_escape(cmp_v1.get('current_action', ''))}</td>"
        f"<td>{_html_escape(cmp_v1.get('royal_road_action', ''))}</td></tr>"
        f"<tr><td>same?</td><td colspan='2'>{cmp_v1.get('same_action')}</td></tr>"
        f"<tr><td>diff</td><td colspan='2'>{_html_escape(cmp_v1.get('difference_type', ''))}</td></tr>"
        "</table>"
    )

    panels_html = _render_checklist_panels_html(panels)

    setup_html = "".join(
        f"<details><summary>{_setup_candidate_summary_ja(i, c)}</summary>"
        f"<pre>{_html_escape(json.dumps(c, indent=2, default=str))}</pre>"
        "</details>"
        for i, c in enumerate(setup_candidates)
    ) or "<p class='placeholder'>セットアップ候補なし</p>"

    block_html = (
        "<ul>" + "".join(
            f"<li><b>{_html_escape(r)}</b></li>" for r in block_reasons
        ) + "</ul>"
        if block_reasons else "<p>(none)</p>"
    )
    cautions_html = (
        "<ul>" + "".join(
            f"<li>{_html_escape(c)}</li>" for c in cautions
        ) + "</ul>"
        if cautions else "<p>(none)</p>"
    )

    rq_html = f"<pre>{_html_escape(json.dumps(rq, indent=2, default=str))}</pre>"

    raw_excerpt = {
        "case_id": payload.get("case_id"),
        "symbol": payload.get("symbol"),
        "parent_bar_ts": payload.get("parent_bar_ts"),
        "render_window_end_ts": payload.get("render_window_end_ts"),
        "bars_used_in_render": payload.get("bars_used_in_render"),
        "action": v2.get("action"),
        "mode": v2.get("mode"),
        "block_reasons": block_reasons,
        "cautions": cautions,
        "reconstruction_quality_total":
            rq.get("total_reconstruction_score"),
        "compared_to_current_runtime": cmp_v1,
    }

    wave_review = payload.get("wave_shape_review") or {}
    entry_summary = payload.get("entry_summary") or {}
    wave_derived = payload.get("wave_derived_lines") or []
    wave_review_html = _render_wave_shape_review_html(wave_review)
    pattern_dissection_html = _render_pattern_dissection_html(wave_review)
    entry_summary_html = _render_entry_summary_html(entry_summary)
    line_count_html = _render_line_count_summary_html(
        overlays=payload.get("overlays") or {},
        wave_derived_lines=wave_derived,
    )
    wave_derived_table_html = _render_wave_derived_lines_table_html(
        wave_derived,
    )
    hold_waveform_html = _hold_waveform_extra_html(
        action=v2.get("action"),
        review=wave_review,
        wave_derived_lines=wave_derived,
    )
    wave_only_html = (
        f"<div class='wave-only-wrap'>{wave_only_svg}</div>"
        if wave_only_svg else ""
    )
    legend_html = _render_wave_line_legend_html(wave_derived)
    wave_only_kind = (wave_overlay or {}).get("kind", "") if wave_overlay else ""
    wave_only_heading = _wave_only_chart_heading(wave_only_kind)
    # Inject final_action_hint so Tier 1 can show the action without
    # requiring a parent payload reference inside the renderer.
    mc_for_render = dict(payload.get("masterclass_panels") or {})
    mc_for_render["final_action_hint"] = (
        (payload.get("decision_bridge") or {}).get("final_action")
        or v2.get("action") or "—"
    )
    masterclass_panels_html = _render_masterclass_panels_html(
        mc_for_render,
    )
    decision_bridge_html = _render_decision_bridge_html(
        payload.get("decision_bridge"),
        entry_summary=entry_summary,
    )

    return (
        f"<section class='case-section' id='{_html_escape(section_id)}'>"
        f"<h2>{title}</h2>"
        # Decision bridge first — answers "what drove this action?"
        + decision_bridge_html
        + (legend_html if legend_html else "")
        + "<h3>chart</h3>"
        + chart_html
        + (
            f"<h3>{_html_escape(wave_only_heading)}</h3>"
            + wave_only_html
            if wave_only_html else ""
        )
        + "<h3>線カウントサマリ (S/R + T/X + 波形由来)</h3>"
        + line_count_html
        + "<h3>波形由来の線 (wave-derived lines)</h3>"
        + wave_derived_table_html
        + "<h3>結論カード (entry / stop / RR)</h3>"
        + entry_summary_html
        + "<h3>波形レビュー (waveform shape review — observation only)</h3>"
        + wave_review_html
        + (
            "<h3>パターン解剖 (pattern parts)</h3>"
            + pattern_dissection_html
            if pattern_dissection_html else ""
        )
        + (hold_waveform_html if hold_waveform_html else "")
        + "<h3>Masterclass パネル (16機能 — observation only)</h3>"
        + masterclass_panels_html
        + "<h3>Royal Road Checklist Panels</h3>"
        + panels_html
        + "<h3>current_runtime vs royal_road_v2</h3>"
        + cmp_html
        + "<h3>setup_candidates</h3>"
        + setup_html
        + "<h3>block_reasons</h3>"
        + block_html
        + "<h3>cautions</h3>"
        + cautions_html
        + "<h3>reconstruction_quality</h3>"
        + rq_html
        + "<h3>audit summary (raw)</h3>"
        + f"<pre>{_html_escape(json.dumps(raw_excerpt, indent=2, default=str))}</pre>"
        "<p><a href='#case-list'>↑ back to case list</a></p>"
        "</section>"
    )


def render_visual_audit_mobile_single_file(
    *,
    traces: Iterable[Any],
    df_by_symbol: dict[str, pd.DataFrame],
    df_lower_by_symbol: dict[str, pd.DataFrame] | None = None,
    out_path: Path | str,
    max_cases: int = 5,
    title: str = "visual_audit_mobile_v1",
    demo_fixture_banner: str | None = None,
) -> dict:
    """Generate ONE self-contained HTML file with up to `max_cases`
    cases. Designed for sharing audit results to a phone where relative
    links and multi-file layouts break easily.

    Output is a single HTML document with:
      - <meta name='viewport' content='width=device-width, initial-scale=1'>
      - inline <style> (full _DEFAULT_CSS, including the @media block)
      - top: case list with in-page anchors
      - per case: inline <svg> chart + checklist panels + comparison +
        setup_candidates + block_reasons + cautions + raw audit excerpt
      - demo_fixture_banner emitted as a prominent banner when set

    Future-leak: the chart SVG is built via `_build_candle_svg_xml`
    which only reads `df.index <= parent_bar_ts`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = select_important_cases(traces, max_cases=max_cases)

    sections: list[str] = []
    list_items: list[str] = []
    cases_meta: list[dict] = []

    for idx, case in enumerate(selected, start=1):
        sym = case["symbol"]
        df = df_by_symbol.get(sym)
        payload = build_visual_audit_payload(
            trace=case["trace"],
            df=df if df is not None else pd.DataFrame(),
        )
        if payload is None:
            continue
        section_id = f"case-{idx}"
        v2 = payload.get("royal_road_decision_v2") or {}
        action = v2.get("action") or "?"
        ts = case.get("ts", "")
        case_label = f"{sym} @ {ts} ({action})"
        # Pick the wave overlay (best_pattern's scale) for chart drawing.
        wave_overlay = _select_wave_overlay(payload)
        wd_lines = list(payload.get("wave_derived_lines") or [])
        # Append WFIB* lines from the fibonacci_context_review so the
        # chart overlay renders them alongside the wave-derived lines.
        # observation-only — used_in_decision is preserved (False).
        mc_panels = (payload.get("masterclass_panels") or {}).get("panels") or {}
        fib_panel = mc_panels.get("fibonacci_context_review") or {}
        for fib_line in (fib_panel.get("fib_wave_lines") or []):
            wd_lines.append(fib_line)
        # User-facing chart title (replaces dev "best=... | quality=..."
        # text drawn inside the SVG). Format:
        # 判断: <action> / 波形候補: <kind> / 参考線: <W*-ids>.
        svg_user_title = _build_user_facing_title(
            payload=payload, wave_derived=wd_lines,
        )
        # Build the inline SVG chart string (or None on failure).
        svg_xml: str | None = None
        if df is not None and len(df) > 0:
            try:
                end_ts = pd.Timestamp(payload["parent_bar_ts"])
                svg_xml = _build_candle_svg_xml(
                    df=df, end_ts=end_ts,
                    n_bars=int(payload.get("bars_used_in_render") or 0),
                    overlays=payload.get("overlays") or {},
                    title=svg_user_title,
                    wave_overlay=wave_overlay,
                    wave_derived_lines=wd_lines,
                )
            except Exception:  # noqa: BLE001
                svg_xml = None
        # Inject the case_id / symbol fields the section template uses.
        payload["case_id"] = f"{sym}_{_safe_ts_for_path(ts)}"
        payload["symbol"] = sym
        # Build the standalone "波形だけ" SVG (when a skeleton exists).
        wave_only_svg = (
            _build_wave_only_svg(
                skeleton=wave_overlay.get("skeleton"),
                matched_parts=wave_overlay.get("matched_parts"),
                kind=wave_overlay.get("kind"),
                human_label=wave_overlay.get("human_label"),
                status=wave_overlay.get("status"),
                side_bias=wave_overlay.get("side_bias"),
                entry_summary=payload.get("entry_summary"),
                wave_derived_lines=wd_lines,
            )
            if wave_overlay else None
        )
        sections.append(_mobile_case_section_html(
            section_id=section_id,
            case_label=case_label,
            payload=payload,
            svg_xml=svg_xml,
            df_was_present=df is not None and len(df) > 0,
            wave_only_svg=wave_only_svg,
            wave_overlay=wave_overlay,
        ))
        list_items.append(
            f"<li><a href='#{section_id}'>"
            f"{_html_escape(case_label)} "
            f"<span class='priority-tag'>"
            f"[{_html_escape(case.get('priority', ''))}]</span>"
            "</a></li>"
        )
        cases_meta.append({
            "case_id": payload["case_id"],
            "symbol": sym, "ts": ts,
            "priority": case.get("priority"),
            "action": action,
            "section_id": section_id,
        })

    banner_html = (
        f"<div class='demo-banner'><b>demo_fixture_not_backtest_result</b>: "
        f"{_html_escape(demo_fixture_banner)}</div>"
        if demo_fixture_banner else ""
    )
    case_list_html = (
        "<section id='case-list'><h2>Case list</h2>"
        + (f"<ul>{''.join(list_items)}</ul>" if list_items
           else "<p class='placeholder'>(no cases)</p>")
        + "</section>"
    )

    body = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{_html_escape(title)}</title>"
        f"<style>{_DEFAULT_CSS}</style>"
        "</head><body>"
        f"<h1>{_html_escape(title)}</h1>"
        + banner_html
        + f"<p>schema={MOBILE_SCHEMA_VERSION} cases={len(sections)}</p>"
        + case_list_html
        + "".join(sections)
        + "</body></html>"
    )
    out_path.write_text(body)
    return {
        "schema_version": MOBILE_SCHEMA_VERSION,
        "out_path": str(out_path),
        "n_cases": len(sections),
        "cases": cases_meta,
        "size_bytes": out_path.stat().st_size,
        "demo_fixture_not_backtest_result": bool(demo_fixture_banner),
    }
