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
from .source_pack_daily_roadmap import build_daily_roadmap_review
from .source_pack_fibonacci import (
    build_fib_wave_lines,
    build_fibonacci_context_review,
)
from .source_pack_symbol_briefing import build_symbol_macro_briefing_review


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
    symbol: str | None = None,
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
        higher_tf_trend=higher_tf_trend,
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

    invalidation = build_invalidation_engine_v2(
        entry_summary=entry_summary,
        pattern_review=pattern,
        overlays=overlays,
        dow_review=dow,
        ma_review=ma_review,
        invalidation_explanation=invalidation_explanation,
    )

    # Source-pack additions: Fibonacci context + WFIB lines, Daily
    # 10k roadmap, per-symbol macro briefing.
    # Encode a skeleton from the visible df for fib anchor detection.
    from .waveform_encoder import encode_wave_skeleton
    fib_skeleton = None
    if visible_df is not None and len(visible_df) >= 50:
        fib_skeleton = encode_wave_skeleton(
            visible_df, scale="fib_anchor", atr_value=atr_value,
        ).to_dict()
    fib_review = build_fibonacci_context_review(
        skeleton=fib_skeleton, last_close=last_close,
    )
    fib_lines = build_fib_wave_lines(fibonacci_review=fib_review)
    # NOTE: fib_lines are observation-only and do NOT modify the
    # `wave_derived_lines` already in the payload (they're surfaced
    # separately under `fibonacci_context_review.fib_wave_lines` for
    # the audit reader and will be added to the chart overlay by a
    # later visual_audit edit).
    fib_review["fib_wave_lines"] = fib_lines

    roadmap = build_daily_roadmap_review(
        macro_score=macro_score,
        higher_tf_trend=higher_tf_trend,
        invalidation_review=invalidation,
        entry_summary=entry_summary,
    )

    symbol_briefing = build_symbol_macro_briefing_review(
        symbol=symbol, data_available=False,
    )

    # Build confluence AFTER source-pack panels so it sees them too.
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
        fibonacci_review=fib_review,
        mtf_story=mtf,
        roadmap_review=roadmap,
        symbol_briefing_review=symbol_briefing,
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

    raw_panels = {
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
        # Source-pack additions (observation-only)
        "fibonacci_context_review": fib_review,
        "daily_roadmap_review": roadmap,
        "symbol_macro_briefing_review": symbol_briefing,
    }
    enriched_panels = {
        name: _enrich_panel_contract(name, p)
        for name, p in raw_panels.items()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "panels": enriched_panels,
    }


# ---------------------------------------------------------------------------
# Per-panel contract enrichment
#
# Every panel must carry the following audit-contract keys when emitted
# from the aggregator (in addition to the panel's own fields):
#
#   status                   PASS / WARN / BLOCK / UNKNOWN
#   inputs_used              list of input field names consulted
#   what_to_check_on_chart_ja  JA hint pointing the reader at the
#                            chart spot to verify
#
# Panels whose own builder already emits any of these are not
# overwritten — the enricher only fills missing keys.
# ---------------------------------------------------------------------------


_PANEL_INPUTS_USED: Final[dict[str, list[str]]] = {
    "candlestick_anatomy_review": [
        "visible_df.parent_bar (open/high/low/close)",
        "visible_df.prev_bar (for engulfing/harami)",
        "technical.atr_14",
        "technical_confluence.support_resistance.near_support",
        "technical_confluence.support_resistance.near_resistance",
    ],
    "parent_bar_lower_tf_anatomy": [
        "df_lower_tf (5m / 15m bars)",
        "parent_bar_ts",
        "higher_tf_interval_minutes",
    ],
    "dow_structure_review": [
        "visible_df.high",
        "visible_df.low",
        "patterns.detect_swings (lookback=3)",
    ],
    "chart_pattern_anatomy_v2": [
        "wave_shape_review.best_pattern",
        "wave_derived_lines (WSL/WTP)",
    ],
    "level_psychology_review": [
        "overlays.level_zones_selected",
        "visible_df.close (last)",
    ],
    "indicator_environment_router": [
        "technical_confluence.market_regime",
        "technical_confluence.indicator_context.bb_squeeze",
        "technical_confluence.indicator_context.bb_expansion",
    ],
    "ma_context_review": [
        "technical.sma_20", "technical.sma_50", "technical.ema_12",
        "technical_prev.sma_20 (slope)",
        "technical_prev.sma_50 (slope)",
        "technical.atr_14",
        "visible_df.close (last)",
    ],
    "granville_entry_review": [
        "visible_df.close (last)",
        "technical.sma_20",
        "technical_prev.sma_20",
        "technical.atr_14",
    ],
    "bollinger_lifecycle_review": [
        "technical_confluence.indicator_context.bb_squeeze",
        "technical_confluence.indicator_context.bb_expansion",
        "technical_confluence.indicator_context.bb_band_walk",
        "technical.bb_position",
    ],
    "rsi_regime_filter": [
        "technical_confluence.indicator_context.rsi_value",
        "technical_confluence.market_regime",
    ],
    "divergence_review": [
        "technical_confluence.indicator_context.rsi_*_divergence",
        "technical_confluence.indicator_context.macd_*_divergence",
    ],
    "macd_architecture_review": [
        "technical.macd", "technical.macd_signal", "technical.macd_hist",
        "technical_prev.macd", "technical_prev.macd_signal",
        "technical_prev.macd_hist",
    ],
    "multi_timeframe_story": [
        "higher_tf_trend (input)",
        "dow_structure_review.trend (middle)",
        "parent_bar_lower_tf_anatomy.lower_tf_wave",
        "parent_bar_lower_tf_anatomy.lower_tf_status",
        "parent_bar_lower_tf_anatomy.side_bias",
    ],
    "grand_confluence_v2": [
        "dow_structure_review", "overlays", "wave_derived_lines",
        "ma_context_review", "rsi_regime_filter",
        "macd_architecture_review", "candlestick_anatomy_review",
        "chart_pattern_anatomy_v2", "entry_summary", "macro_score",
    ],
    "invalidation_engine_v2": [
        "entry_summary.entry_price", "entry_summary.stop_price",
        "entry_summary.structure_stop_price",
        "entry_summary.atr_stop_price",
        "entry_summary.take_profit_price",
        "chart_pattern_anatomy_v2.detected_pattern",
        "overlays.level_zones_selected",
        "dow_structure_review.trend",
    ],
    "pre_trade_diagnostic_checklist_v1": [
        "dow_structure_review", "overlays", "wave_derived_lines",
        "multi_timeframe_story", "ma_context_review",
        "bollinger_lifecycle_review", "rsi_regime_filter",
        "macd_architecture_review", "invalidation_engine_v2",
        "entry_summary",
    ],
    # Source-pack additions
    "fibonacci_context_review": [
        "wave_skeleton (encoded from visible_df)",
        "visible_df.close (last)",
    ],
    "daily_roadmap_review": [
        "macro_score (input)",
        "higher_tf_trend (input)",
        "invalidation_engine_v2",
        "entry_summary",
    ],
    "symbol_macro_briefing_review": [
        "symbol (input)",
        "data_available (currently always False)",
    ],
}


_PANEL_CHECK_HINT_JA: Final[dict[str, str]] = {
    "candlestick_anatomy_review":
        "判定された bar_type (ピンバー / 包み足 / 大陽線 / 等) のヒゲと"
        "実体が、サポレジ / トレンド方向と整合するかをチャート上で確認"
        "してください。",
    "parent_bar_lower_tf_anatomy":
        "上位足の最後の足 (parent_bar) の中身を、下位足チャートで覗くと"
        "どんな波形になっているかを確認してください。"
        "下位足データ未接続の場合はそのまま「未接続」と表示されます。",
    "dow_structure_review":
        "チャートのスイング高安が HH / HL (上昇) または LH / LL (下降) で"
        "並んでいるか、押し安値 (上昇) または戻り高値 (下降) を割っていないか"
        "を確認してください。",
    "chart_pattern_anatomy_v2":
        "「波形だけ表示」チャートに描かれた DB1 / DT1 / HS1 / IHS1 の"
        "ファミリーマーカーと、B1 / B2 / NL / BR などの部位ラベルが"
        "整合しているかを確認してください。",
    "level_psychology_review":
        "S1 / R1 などの帯と現在価格の関係 (initial_test / pullback / "
        "breakout など) が、自分の目で見ても同じ phase になっているかを"
        "確認してください。",
    "indicator_environment_router":
        "今のチャートが TREND か RANGE かをまず判定し、優先するインジケータが"
        "MA / MACD (TREND) なのか RSI / Stoch (RANGE) なのかを意識して"
        "ください。",
    "ma_context_review":
        "SMA20 / SMA50 が右上がり (上昇配列) か右下がり (下降配列) か、"
        "現在価格が SMA20 にタッチしているか乖離しているかを確認してください。",
    "granville_entry_review":
        "MA の傾きと、現在価格が MA に「触れている / 上 / 下」のどれかを"
        "確認してください。下向き MA への戻りで買おうとしていないか (罠) を"
        "特に注意してください。",
    "bollinger_lifecycle_review":
        "BB の幅が縮小 (Squeeze) しているか、拡大 (Expansion) しているか、"
        "片側のバンドに沿って歩いているか (Band Walk) を確認してください。",
    "rsi_regime_filter":
        "RSI の値そのものよりも、TREND 環境では「買われすぎ / 売られすぎ」"
        "を逆張り根拠にしないことを確認してください。",
    "divergence_review":
        "ダイバージェンスは「単独で入る根拠にしない」原則を守り、"
        "形 / 構造 / サポレジでの確認と組み合わせてください。",
    "macd_architecture_review":
        "MACD 線がシグナルを抜けたか (golden / dead cross)、ゼロラインの"
        "上下どちらにあるか、ヒストグラムの符号と勢い (強まる / 弱まる) を"
        "確認してください。",
    "multi_timeframe_story":
        "上位足のトレンド方向と、下位足の波形候補の方向が一致しているかを"
        "確認してください。一致していない場合は王道としては慎重な局面です。",
    "grand_confluence_v2":
        "9 軸 (Dow / Line / MA / Oscillator / Price Action / Pattern / "
        "RR / Invalidation / Macro) のうち PASS が 6 以上 / BLOCK 0 の"
        "場合のみ Strong Confluence として扱うのが王道です。",
    "invalidation_engine_v2":
        "損切り価格が「波形 / サポレジが完全に崩れる場所」に置かれているか、"
        "RR が最低 1:2 以上あるかを必ず確認してください。",
    "pre_trade_diagnostic_checklist_v1":
        "7 つの問いに YES が 6 以上 / NO が 0 のときだけ「事前診断クリア」と"
        "判定します。NO が 1 つでもあれば見送り推奨です。",
    "fibonacci_context_review":
        "波形の最後の 2 ピボットからフィボ 38.2 / 50 / 61.8 / 127.2 / 161.8 を"
        "計算し、現在価格がどのゾーンにいるかを確認してください。",
    "daily_roadmap_review":
        "未接続 (UNKNOWN) 項目はポジションサイジング・経済指標カレンダー・"
        "ジャーナルなど。手動で確認してください。",
    "symbol_macro_briefing_review":
        "通貨ペア固有のファンダ要因 (USDJPY なら米金利・DXY・リスク選好) が"
        "現在の方向と整合しているかを別チャートで確認してください。"
        "実データは現状未接続です。",
}


def _infer_status_from_panel(name: str, panel: dict) -> str:
    """Heuristic status (PASS / WARN / BLOCK / UNKNOWN) when the panel
    does not emit one itself."""
    if not panel or not panel.get("available"):
        return "UNKNOWN"
    # Per-panel rules
    if name == "dow_structure_review":
        t = panel.get("trend", "UNKNOWN")
        if t in ("UP", "DOWN"):
            return "PASS"
        if t in ("RANGE", "MIXED"):
            return "WARN"
        return "UNKNOWN"
    if name == "chart_pattern_anatomy_v2":
        st = panel.get("status", "")
        if st == "neckline_broken":
            return "PASS"
        if st in ("forming", "neckline_not_broken", "retested"):
            return "WARN"
        if st == "invalidated":
            return "BLOCK"
        return "UNKNOWN"
    if name == "candlestick_anatomy_review":
        bar = panel.get("bar_type", "")
        if bar in ("bullish_pinbar", "bearish_pinbar",
                   "bullish_engulfing", "bearish_engulfing",
                   "large_bull", "large_bear"):
            return "PASS"
        if bar in ("bullish_harami", "bearish_harami",
                   "neutral_doji"):
            return "WARN"
        return "UNKNOWN"
    if name == "ma_context_review":
        if panel.get("is_uptrend_stack") or panel.get("is_downtrend_stack"):
            return "PASS"
        if panel.get("over_extended"):
            return "WARN"
        return "WARN"
    if name == "rsi_regime_filter":
        if panel.get("usable_signal"):
            return "PASS"
        if panel.get("raw_signal") in ("overbought", "oversold"):
            return "WARN"
        return "UNKNOWN"
    if name == "macd_architecture_review":
        bias = panel.get("bias", "")
        if bias in ("BUY", "SELL"):
            return "PASS"
        return "UNKNOWN"
    if name == "bollinger_lifecycle_review":
        stage = panel.get("stage", "")
        if stage in ("squeeze", "expansion", "band_walk"):
            return "PASS" if not panel.get("reversal_risk") else "WARN"
        return "UNKNOWN"
    if name == "divergence_review":
        if panel.get("any_divergence"):
            return "WARN"  # never PASS — masterclass: divergence alone is not entry
        return "UNKNOWN"
    if name == "indicator_environment_router":
        if panel.get("is_trend_regime") or panel.get("is_range_regime"):
            return "PASS"
        return "UNKNOWN"
    if name == "granville_entry_review":
        if panel.get("pattern") in (
            "trend_pullback_buy", "trend_pullback_sell",
        ):
            return "PASS"
        if panel.get("pattern") == "range_no_bias":
            return "WARN"
        return "UNKNOWN"
    if name == "level_psychology_review":
        return "PASS" if (panel.get("levels") or []) else "UNKNOWN"
    if name == "multi_timeframe_story":
        return "PASS" if panel.get("tf_aligned") else "WARN"
    if name == "invalidation_engine_v2":
        if panel.get("is_structure_anchored") and panel.get("rr_pass"):
            return "PASS"
        if panel.get("rr_pass") is False:
            return "BLOCK"
        return "WARN"
    if name == "pre_trade_diagnostic_checklist_v1":
        if (panel.get("no_count") or 0) > 0:
            return "BLOCK"
        if (panel.get("yes_count") or 0) >= 6:
            return "PASS"
        return "WARN"
    if name == "grand_confluence_v2":
        block = panel.get("block_count", 0)
        if block:
            return "BLOCK"
        if (panel.get("pass_count") or 0) >= 6:
            return "PASS"
        return "WARN"
    if name == "parent_bar_lower_tf_anatomy":
        if panel.get("lower_tf_wave"):
            return "PASS"
        return "UNKNOWN"
    if name == "fibonacci_context_review":
        # Status already set by builder
        return panel.get("status", "UNKNOWN")
    if name == "daily_roadmap_review":
        return panel.get("status", "UNKNOWN")
    if name == "symbol_macro_briefing_review":
        return panel.get("status", "UNKNOWN")
    return "UNKNOWN"


def _enrich_panel_contract(name: str, panel: dict | None) -> dict | None:
    """Inject status / inputs_used / what_to_check_on_chart_ja into a
    panel without overwriting builder-supplied values."""
    if panel is None:
        return panel
    if "status" not in panel:
        panel["status"] = _infer_status_from_panel(name, panel)
    if "inputs_used" not in panel:
        panel["inputs_used"] = list(_PANEL_INPUTS_USED.get(name, []))
    if "what_to_check_on_chart_ja" not in panel:
        panel["what_to_check_on_chart_ja"] = (
            _PANEL_CHECK_HINT_JA.get(name, "")
        )
    return panel


__all__ = [
    "SCHEMA_VERSION",
    "build_masterclass_panels",
]
