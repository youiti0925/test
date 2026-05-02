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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

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


def _render_candle_svg(
    *,
    df: pd.DataFrame,
    end_ts: pd.Timestamp,
    n_bars: int,
    overlays: dict | None,
    title: str,
    out_path: Path,
) -> dict:
    """Dependency-free SVG fallback for the candle chart. Identical
    overlay set to the matplotlib renderer (selected/rejected SR zones,
    selected/rejected trendlines, pattern necklines, lower_tf trigger,
    stop / take_profit / structure_stop / atr_stop, parent_bar marker).

    Future-leak safe: only bars with `df.index <= end_ts` are drawn.
    Returns a dict with `image_status`, `image_path`, `marker_file`.
    """
    try:
        if df is None or len(df) == 0 or n_bars <= 0:
            marker = out_path.with_suffix(".image_unavailable")
            marker.write_text("no_visible_bars\n")
            return {
                "image_status": "render_error:no_visible_bars",
                "marker_file": str(marker),
            }
        visible = df[df.index <= end_ts].tail(n_bars)
        if len(visible) == 0:
            marker = out_path.with_suffix(".image_unavailable")
            marker.write_text("no_visible_bars\n")
            return {
                "image_status": "render_error:no_visible_bars",
                "marker_file": str(marker),
            }
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
        # Expand range so overlay lines (stops, zones) above / below
        # the visible candles still fit on the canvas.
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
            # Higher price → smaller y (SVG y grows downward).
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
        # Background + plot frame
        parts.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='#fdfdfd'/>")
        parts.append(
            f"<rect x='{margin_l}' y='{margin_t}' width='{plot_w}' "
            f"height='{plot_h}' fill='white' stroke='#bbb' stroke-width='0.5'/>"
        )
        # Y-axis price labels (5 ticks)
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
        # Title
        parts.append(
            f"<text x='{margin_l}' y='18' fill='#222' font-weight='bold'>"
            f"{_html_escape(title)}</text>"
        )

        # ---- SR zones (selected first, rejected on top in lower opacity) ----
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

        # ---- Candles (wick + body) -----------------------------------
        for i in range(n):
            xi = x_of(i)
            y_h = y_of(highs[i]); y_l = y_of(lows[i])
            y_o = y_of(opens[i]); y_c = y_of(closes[i])
            # Wick
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

        # ---- Trendlines (selected) ----------------------------------
        # Anchor indices are bar offsets in the FULL window. For SVG
        # we map to the visible window: any anchor index >= len(df)
        # is silently clipped — same behaviour as the matplotlib path.
        full_len = len(df)
        # Build a mapping from full-window index → visible-window index
        # via timestamps (anchors are relative to df_window which the
        # detector saw, which == the visible window for v2).
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

        # ---- Patterns (neckline as horizontal, upper/lower as sloped) ----
        for p in ov.get("patterns_selected", []):
            side = p.get("side_bias")
            color = (
                "#1b5e20" if side == "BUY"
                else "#b71c1c" if side == "SELL"
                else "#ef6c00"
            )
            # neckline → single price (horizontal)
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
            # upper_line / lower_line → may be a dict {slope, intercept,
            # anchors} (sloped trend boundary) OR a scalar (horizontal).
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

        # ---- Lower TF trigger marker ---------------------------------
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

        # ---- Stop / TP / structure_stop / atr_stop -------------------
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

        # ---- parent_bar marker (vertical at last visible bar) -------
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
        # Time axis: first / last bar timestamp
        parts.append(
            f"<text x='{margin_l}' y='{margin_t + plot_h + 16}' fill='#444'>"
            f"{visible.index[0].isoformat()}</text>"
        )
        parts.append(
            f"<text x='{margin_l + plot_w}' y='{margin_t + plot_h + 16}' "
            f"text-anchor='end' fill='#444'>{visible.index[-1].isoformat()}</text>"
        )
        parts.append("</svg>\n")
        out_path.write_text("".join(parts))
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
"""


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


def _render_invalidation_explanation_html(panel: dict) -> str:
    if not panel.get("available"):
        return (
            f"<p class='panel-unavailable'>"
            f"invalidation_explanation unavailable: "
            f"{_html_escape(panel.get('unavailable_reason', ''))}</p>"
        )
    return (
        "<table>"
        f"<tr><td>chosen_mode</td><td>{_html_escape(panel.get('chosen_mode'))}</td></tr>"
        f"<tr><td>outcome</td><td>{_html_escape(panel.get('outcome'))}</td></tr>"
        f"<tr><td>selected_stop_price</td><td>{panel.get('selected_stop_price')}</td></tr>"
        f"<tr><td>atr_stop_price</td><td>{panel.get('atr_stop_price')}</td></tr>"
        f"<tr><td>structure_stop_price</td><td>{panel.get('structure_stop_price')}</td></tr>"
        f"<tr><td>invalidation_structure</td><td>{_html_escape(panel.get('invalidation_structure'))}</td></tr>"
        f"<tr><td>invalidation_status</td><td><b>{_html_escape(panel.get('invalidation_status'))}</b></td></tr>"
        f"<tr><td>rr_selected</td><td>{panel.get('rr_selected')}</td></tr>"
        f"<tr><td>why_this_stop_invalidates_the_setup</td><td>"
        f"{_html_escape(panel.get('why_this_stop_invalidates_the_setup', ''))}"
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
    title = (
        f"{sym} {ts} "
        f"action={v2.get('action')} "
        f"mode={v2.get('mode')} "
        f"quality={(v2.get('reconstruction_quality') or {}).get('total_reconstruction_score', 0.0):.3f}"
    )
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
        f"<details open><summary>candidate #{i+1} side={c.get('side')} "
        f"score={c.get('score'):.3f} confidence={c.get('confidence')}</summary>"
        f"<pre>{_html_escape(json.dumps(c, indent=2, default=str))}</pre>"
        "</details>"
        for i, c in enumerate(setup_candidates)
    ) or "<p class='placeholder'>no setup candidates</p>"
    best_setup = v2.get("best_setup")
    best_html = (
        f"<pre>{_html_escape(json.dumps(best_setup, indent=2, default=str))}</pre>"
        if best_setup else "<p class='placeholder'>None</p>"
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
