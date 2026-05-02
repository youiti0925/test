"""Aggregator for the Masterclass observation-only audit panels.

Single entry point `build_masterclass_panels(...)` returns a dict
that aggregates all 16 Masterclass features into one
`masterclass_panels` block on the visual_audit payload.

The aggregator wires together:

  candlestick_anatomy_review
  parent_bar_lower_tf_anatomy
  dow_structure_review
  chart_pattern_anatomy_v2
  level_psychology_review
  indicator_environment_router
  ma_context_review
  granville_entry_review
  bollinger_lifecycle_review
  rsi_regime_filter
  divergence_review
  macd_architecture_review
  multi_timeframe_story
  grand_confluence_v2
  invalidation_engine_v2
  pre_trade_diagnostic_checklist_v1

Strict invariants
-----------------
- Every panel emitted carries `observation_only=True` and
  `used_in_decision=False`.
- The aggregator never reads beyond `parent_bar_ts` (each
  underlying panel is responsible for its own future-leak guard).
- Returns `{"available": False}` when no v2 audit is being
  produced (so default current_runtime traces never receive a
  Masterclass payload).
"""
from __future__ import annotations

from typing import Final

import pandas as pd

from .masterclass_candlestick import (
    build_candlestick_anatomy_review,
    build_parent_bar_lower_tf_anatomy,
)
from .masterclass_checklist import build_pre_trade_diagnostic_checklist
from .masterclass_confluence import build_grand_confluence_v2
from .masterclass_dow import build_dow_structure_review
from .masterclass_indicators import (
    build_bollinger_lifecycle_review,
    build_divergence_review,
    build_granville_entry_review,
    build_indicator_environment_router,
    build_ma_context_review,
    build_macd_architecture_review,
    build_rsi_regime_filter,
)
from .masterclass_invalidation import build_invalidation_engine_v2
from .masterclass_levels import build_level_psychology_review
from .masterclass_mtf import build_multi_timeframe_story
from .masterclass_pattern import build_chart_pattern_anatomy_v2


SCHEMA_VERSION: Final[str] = "masterclass_panels_v1"


def _empty() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "panels": {},
    }


def build_masterclass_panels(
    *,
    visible_df: pd.DataFrame | None,
    parent_bar_ts: pd.Timestamp | str | None,
    technical_dict: dict | None,
    technical_confluence: dict | None,
    overlays: dict | None,
    wave_shape_review: dict | None,
    wave_derived_lines: list[dict] | None,
    entry_summary: dict | None,
    invalidation_explanation: dict | None,
    df_lower_tf: pd.DataFrame | None = None,
    higher_tf_interval_minutes: int = 60,
    higher_tf_trend: str | None = None,
    macro_score: float | None = None,
    technical_prev: dict | None = None,
) -> dict:
    """Build the masterclass_panels block.

    Returns `{"available": False}` when `visible_df` is None or
    empty (no v2 audit context).
    """
    if visible_df is None or len(visible_df) == 0:
        return _empty()

    tc = technical_confluence or {}
    tech = technical_dict or {}
    tech_prev = technical_prev or {}

    atr_value = tech.get("atr_14")
    last_close = (
        float(visible_df["close"].iloc[-1])
        if len(visible_df) > 0 else None
    )

    # Inputs from the existing technical_confluence slice
    sr_ctx = (tc.get("support_resistance") or {})
    near_support = bool(sr_ctx.get("near_support"))
    near_resistance = bool(sr_ctx.get("near_resistance"))
    indicator_ctx = (tc.get("indicator_context") or {})
    market_regime = tc.get("market_regime")
    rsi_value = indicator_ctx.get("rsi_value")
    bb_squeeze = indicator_ctx.get("bb_squeeze")
    bb_expansion = indicator_ctx.get("bb_expansion")
    bb_band_walk = indicator_ctx.get("bb_band_walk")

    # Build each panel.
    candlestick = build_candlestick_anatomy_review(
        visible_df=visible_df,
        atr_value=atr_value,
        near_support=near_support,
        near_resistance=near_resistance,
    )
    parent_lower_tf = build_parent_bar_lower_tf_anatomy(
        parent_bar_ts=parent_bar_ts,
        df_lower_tf=df_lower_tf,
        higher_tf_interval_minutes=higher_tf_interval_minutes,
    )
    dow = build_dow_structure_review(visible_df=visible_df)
    pattern = build_chart_pattern_anatomy_v2(
        wave_shape_review=wave_shape_review,
        wave_derived_lines=wave_derived_lines,
    )
    levels = build_level_psychology_review(
        overlays=overlays, last_close=last_close,
    )
    env_router = build_indicator_environment_router(
        market_regime=market_regime,
        bb_squeeze=bool(bb_squeeze),
        bb_expansion=bool(bb_expansion),
    )
    ma_review = build_ma_context_review(
        last_close=last_close,
        sma_20=tech.get("sma_20"),
        sma_50=tech.get("sma_50"),
        ema_12=tech.get("ema_12"),
        sma_20_prev=tech_prev.get("sma_20"),
        sma_50_prev=tech_prev.get("sma_50"),
        atr_value=atr_value,
    )
    granville = build_granville_entry_review(
        last_close=last_close,
        sma_20=tech.get("sma_20"),
        sma_20_prev=tech_prev.get("sma_20"),
        atr_value=atr_value,
    )
    bb_review = build_bollinger_lifecycle_review(
        bb_squeeze=bb_squeeze,
        bb_expansion=bb_expansion,
        bb_band_walk=bb_band_walk,
        bb_position=tech.get("bb_position"),
    )
    rsi_review = build_rsi_regime_filter(
        rsi_value=rsi_value, market_regime=market_regime,
    )
    divergence = build_divergence_review(
        rsi_bearish_divergence=indicator_ctx.get("rsi_bearish_divergence"),
        rsi_bullish_divergence=indicator_ctx.get("rsi_bullish_divergence"),
        macd_bearish_divergence=indicator_ctx.get(
            "macd_bearish_divergence"
        ),
        macd_bullish_divergence=indicator_ctx.get(
            "macd_bullish_divergence"
        ),
    )
    macd_review = build_macd_architecture_review(
        macd=tech.get("macd"),
        macd_signal=tech.get("macd_signal"),
        macd_hist=tech.get("macd_hist"),
        macd_prev=tech_prev.get("macd"),
        macd_signal_prev=tech_prev.get("macd_signal"),
        macd_hist_prev=tech_prev.get("macd_hist"),
    )

    # Multi-timeframe story
    middle_tf_state = None
    if dow.get("trend") == "UP":
        middle_tf_state = "pullback"
    elif dow.get("trend") == "DOWN":
        middle_tf_state = "rebound"
    elif dow.get("trend") in ("RANGE", "MIXED"):
        middle_tf_state = "range"
    lower_tf_kind = (
        parent_lower_tf.get("lower_tf_wave")
        if parent_lower_tf.get("available") else None
    )
    lower_tf_status = (
        parent_lower_tf.get("lower_tf_status")
        if parent_lower_tf.get("available") else None
    )
    lower_tf_side = (
        parent_lower_tf.get("side_bias")
        if parent_lower_tf.get("available") else None
    )
    mtf = build_multi_timeframe_story(
        higher_tf_trend=higher_tf_trend or dow.get("trend"),
        middle_tf_state=middle_tf_state,
        lower_tf_kind=lower_tf_kind,
        lower_tf_status=lower_tf_status,
        lower_tf_side_bias=lower_tf_side,
    )

    confluence = build_grand_confluence_v2(
        dow_review=dow,
        overlays=overlays,
        wave_derived_lines=wave_derived_lines,
        ma_review=ma_review,
        rsi_review=rsi_review,
        macd_review=macd_review,
        candlestick_review=candlestick,
        pattern_review=pattern,
        entry_summary=entry_summary,
        macro_score=macro_score,
    )
    invalidation = build_invalidation_engine_v2(
        entry_summary=entry_summary,
        pattern_review=pattern,
        overlays=overlays,
        dow_review=dow,
        ma_review=ma_review,
        invalidation_explanation=invalidation_explanation,
    )
    checklist = build_pre_trade_diagnostic_checklist(
        dow_review=dow,
        overlays=overlays,
        wave_derived_lines=wave_derived_lines,
        mtf_story=mtf,
        ma_review=ma_review,
        bb_review=bb_review,
        rsi_review=rsi_review,
        macd_review=macd_review,
        invalidation_review=invalidation,
        entry_summary=entry_summary,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "panels": {
            "candlestick_anatomy_review": candlestick,
            "parent_bar_lower_tf_anatomy": parent_lower_tf,
            "dow_structure_review": dow,
            "chart_pattern_anatomy_v2": pattern,
            "level_psychology_review": levels,
            "indicator_environment_router": env_router,
            "ma_context_review": ma_review,
            "granville_entry_review": granville,
            "bollinger_lifecycle_review": bb_review,
            "rsi_regime_filter": rsi_review,
            "divergence_review": divergence,
            "macd_architecture_review": macd_review,
            "multi_timeframe_story": mtf,
            "grand_confluence_v2": confluence,
            "invalidation_engine_v2": invalidation,
            "pre_trade_diagnostic_checklist_v1": checklist,
        },
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_masterclass_panels",
]
