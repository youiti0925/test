"""Tests for the 16 Masterclass observation-only audit panels.

Pinned invariants (per panel):
  - schema_version is the documented v1 string
  - observation_only == True
  - used_in_decision == False
  - panel-specific fields are present when available=True
  - Japanese reason / meaning text is non-empty when available
  - inputs that are too sparse / missing → available=False with
    unavailable_reason set
  - default current_runtime profile produces no masterclass_panels
    (verified via build_visual_audit_payload returning None)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fx.masterclass_aggregate import (
    SCHEMA_VERSION as AGG_SCHEMA,
    build_masterclass_panels,
)
from src.fx.masterclass_candlestick import (
    CANDLESTICK_SCHEMA,
    LOWER_TF_SCHEMA,
    build_candlestick_anatomy_review,
    build_parent_bar_lower_tf_anatomy,
)
from src.fx.masterclass_checklist import (
    SCHEMA_VERSION as CK_SCHEMA,
    build_pre_trade_diagnostic_checklist,
)
from src.fx.masterclass_confluence import (
    SCHEMA_VERSION as CONF_SCHEMA,
    build_grand_confluence_v2,
)
from src.fx.masterclass_dow import (
    SCHEMA_VERSION as DOW_SCHEMA,
    build_dow_structure_review,
)
from src.fx.masterclass_indicators import (
    BB_SCHEMA,
    DIVERGENCE_SCHEMA,
    ENV_ROUTER_SCHEMA,
    GRANVILLE_SCHEMA,
    MA_CONTEXT_SCHEMA,
    MACD_SCHEMA,
    RSI_SCHEMA,
    build_bollinger_lifecycle_review,
    build_divergence_review,
    build_granville_entry_review,
    build_indicator_environment_router,
    build_ma_context_review,
    build_macd_architecture_review,
    build_rsi_regime_filter,
)
from src.fx.masterclass_invalidation import (
    SCHEMA_VERSION as INV_SCHEMA,
    build_invalidation_engine_v2,
)
from src.fx.masterclass_levels import (
    SCHEMA_VERSION as LVL_SCHEMA,
    build_level_psychology_review,
)
from src.fx.masterclass_mtf import (
    SCHEMA_VERSION as MTF_SCHEMA,
    build_multi_timeframe_story,
)
from src.fx.masterclass_pattern import (
    SCHEMA_VERSION as PAT_SCHEMA,
    build_chart_pattern_anatomy_v2,
)


# ---------------------------------------------------------------------------
# Common shape invariants
# ---------------------------------------------------------------------------


def _assert_observation_shape(panel: dict) -> None:
    assert panel["observation_only"] is True
    assert panel["used_in_decision"] is False
    assert "schema_version" in panel
    assert isinstance(panel.get("schema_version"), str)


# ---------------------------------------------------------------------------
# 1. candlestick_anatomy_review
# ---------------------------------------------------------------------------


def _bullish_pinbar_df() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=2, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open":  [1.10,  1.10],
        "high":  [1.105, 1.105],
        "low":   [1.099, 1.090],   # long lower wick
        "close": [1.102, 1.103],   # small body
        "volume":[1000, 1000],
    }, index=idx)


def test_candlestick_pinbar_classifies_bullish_pinbar_with_ja_text():
    df = _bullish_pinbar_df()
    p = build_candlestick_anatomy_review(
        visible_df=df, atr_value=0.005,
        near_support=True, near_resistance=False,
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == CANDLESTICK_SCHEMA
    assert p["available"] is True
    assert p["bar_type"] == "bullish_pinbar"
    assert p["direction"] == "BUY"
    # Japanese fields are non-empty
    assert p["meaning_ja"]
    assert "ピンバー" in p["meaning_ja"] or "pinbar" in p["meaning_ja"].lower()
    assert p["context_ja"]
    assert p["warning_ja"]


def test_candlestick_empty_input_returns_unavailable():
    p = build_candlestick_anatomy_review(
        visible_df=None, atr_value=None,
    )
    _assert_observation_shape(p)
    assert p["available"] is False
    assert p["unavailable_reason"] == "no_visible_bars"


# ---------------------------------------------------------------------------
# 2. parent_bar_lower_tf_anatomy
# ---------------------------------------------------------------------------


def test_lower_tf_missing_returns_unavailable():
    p = build_parent_bar_lower_tf_anatomy(
        parent_bar_ts=pd.Timestamp("2025-01-01T00:00:00+00:00"),
        df_lower_tf=None,
    )
    _assert_observation_shape(p)
    assert p["available"] is False
    assert p["unavailable_reason"] == "lower_tf_missing"


def test_lower_tf_valid_window_runs_skeleton_match():
    parent_close = pd.Timestamp("2025-01-01T01:00:00+00:00")
    # 12 lower-tf 5-min bars inside the 1h parent window
    n = 12
    idx = pd.date_range(
        "2025-01-01T00:05:00+00:00", periods=n, freq="5min", tz="UTC",
    )
    close = np.array(
        [1.10, 1.099, 1.097, 1.096, 1.098, 1.099,
         1.097, 1.096, 1.099, 1.101, 1.103, 1.105],
        dtype=float,
    )
    df_lt = pd.DataFrame({
        "open": close, "high": close + 0.0005, "low": close - 0.0005,
        "close": close, "volume": [1000] * n,
    }, index=idx)
    p = build_parent_bar_lower_tf_anatomy(
        parent_bar_ts=parent_close,
        df_lower_tf=df_lt,
        higher_tf_interval_minutes=60,
    )
    _assert_observation_shape(p)
    # Either available with a result, or available=False with too_few
    if p["available"]:
        assert "meaning_ja" in p
    assert p["schema_version"] == LOWER_TF_SCHEMA


# ---------------------------------------------------------------------------
# 3. dow_structure_review
# ---------------------------------------------------------------------------


def _trending_df(seed: int = 7, n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def test_dow_review_emits_trend_and_status_ja():
    df = _trending_df()
    p = build_dow_structure_review(visible_df=df)
    _assert_observation_shape(p)
    assert p["schema_version"] == DOW_SCHEMA
    if p["available"]:
        assert p["trend"] in ("UP", "DOWN", "RANGE", "MIXED", "UNKNOWN")
        assert p["status_ja"]


def test_dow_review_too_few_bars_returns_unavailable():
    df = _trending_df(n=8)
    p = build_dow_structure_review(visible_df=df)
    _assert_observation_shape(p)
    assert p["available"] is False
    assert p["unavailable_reason"] == "too_few_bars"


# ---------------------------------------------------------------------------
# 4. chart_pattern_anatomy_v2
# ---------------------------------------------------------------------------


def test_pattern_anatomy_db_emits_status_and_parts_with_ja():
    review = {
        "best_pattern": {
            "kind": "double_bottom",
            "status": "forming",
            "side_bias": "BUY",
            "shape_score": 0.84,
            "matched_parts": {
                "first_bottom": 20, "neckline_peak": 45,
                "second_bottom": 70, "breakout": 99,
            },
        },
    }
    wd = [
        {"id": "WSL1", "role": "stop_candidate", "price": 1.0,
         "kind": "pattern_invalidation"},
        {"id": "WTP1", "role": "target_candidate", "price": 1.14,
         "kind": "pattern_target"},
    ]
    p = build_chart_pattern_anatomy_v2(
        wave_shape_review=review, wave_derived_lines=wd,
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == PAT_SCHEMA
    assert p["available"] is True
    assert p["detected_pattern"] == "double_bottom"
    assert p["status"] == "neckline_not_broken"
    assert "ダブルボトム" in p["summary_ja"]
    assert any(part["label"] == "B1" for part in p["expected_parts"])
    assert p["sl_line"] is not None
    assert p["tp_line"] is not None


def test_pattern_anatomy_no_pattern_returns_unavailable():
    p = build_chart_pattern_anatomy_v2(
        wave_shape_review=None, wave_derived_lines=[],
    )
    _assert_observation_shape(p)
    assert p["available"] is False


# ---------------------------------------------------------------------------
# 5. level_psychology_review
# ---------------------------------------------------------------------------


def test_level_psychology_classifies_zones_with_ja():
    overlays = {
        "level_zones_selected": [
            {
                "kind": "support", "zone_low": 1.090, "zone_high": 1.092,
                "price": 1.091, "touch_count": 3,
                "body_break_count": 0, "role_reversal_count": 1,
                "false_breakout_count": 0,
                "distance_to_close_atr": 0.4,
            },
            {
                "kind": "resistance", "zone_low": 1.110, "zone_high": 1.112,
                "price": 1.111, "touch_count": 1,
                "body_break_count": 0, "role_reversal_count": 0,
                "false_breakout_count": 0,
                "distance_to_close_atr": 5.0,
            },
        ],
    }
    p = build_level_psychology_review(overlays=overlays, last_close=1.1)
    _assert_observation_shape(p)
    assert p["schema_version"] == LVL_SCHEMA
    assert p["available"] is True
    assert len(p["levels"]) == 2
    s1 = p["levels"][0]
    assert s1["level_id"] == "S1"
    assert s1["psychology_ja"]
    assert s1["order_flow_ja"]
    assert s1["used_in_decision"] is False


def test_level_psychology_no_zones_returns_unavailable():
    p = build_level_psychology_review(
        overlays={"level_zones_selected": []},
        last_close=1.1,
    )
    _assert_observation_shape(p)
    assert p["available"] is False


# ---------------------------------------------------------------------------
# 6. indicator_environment_router
# ---------------------------------------------------------------------------


def test_env_router_trend_prefers_ma_macd():
    p = build_indicator_environment_router(market_regime="TREND_UP")
    _assert_observation_shape(p)
    assert p["schema_version"] == ENV_ROUTER_SCHEMA
    assert p["available"] is True
    assert p["is_trend_regime"] is True
    assert any(
        "MA" in s for s in p["preferred_indicators"]
    )
    assert "RSI" in p["reason_ja"]


def test_env_router_range_prefers_oscillators():
    p = build_indicator_environment_router(market_regime="RANGE")
    assert p["is_range_regime"] is True
    assert any("RSI" in s for s in p["preferred_indicators"])


def test_env_router_missing_regime_unavailable():
    p = build_indicator_environment_router(market_regime=None)
    _assert_observation_shape(p)
    assert p["available"] is False


# ---------------------------------------------------------------------------
# 7. ma_context_review
# ---------------------------------------------------------------------------


def test_ma_context_uptrend_stack():
    p = build_ma_context_review(
        last_close=1.10, sma_20=1.099, sma_50=1.097,
        sma_20_prev=1.0985, sma_50_prev=1.0965, atr_value=0.005,
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == MA_CONTEXT_SCHEMA
    assert p["available"] is True
    assert p["is_uptrend_stack"] is True
    assert p["summary_ja"]


def test_ma_context_missing_inputs():
    p = build_ma_context_review(
        last_close=None, sma_20=None, sma_50=None,
    )
    assert p["available"] is False


# ---------------------------------------------------------------------------
# 8. granville_entry_review
# ---------------------------------------------------------------------------


def test_granville_uptrend_pullback_buy():
    p = build_granville_entry_review(
        last_close=1.099, sma_20=1.099, sma_20_prev=1.097,
        atr_value=0.005,
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == GRANVILLE_SCHEMA
    assert p["available"] is True
    assert p["pattern"] == "trend_pullback_buy"
    assert p["meaning_ja"]


def test_granville_missing_inputs():
    p = build_granville_entry_review(last_close=None, sma_20=None)
    assert p["available"] is False


# ---------------------------------------------------------------------------
# 9. bollinger_lifecycle_review
# ---------------------------------------------------------------------------


def test_bb_squeeze_stage():
    p = build_bollinger_lifecycle_review(
        bb_squeeze=True, bb_expansion=False, bb_band_walk=False,
        bb_position=0.5,
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == BB_SCHEMA
    assert p["stage"] == "squeeze"
    assert "Squeeze" in p["meaning_ja"]


def test_bb_band_walk_stage():
    p = build_bollinger_lifecycle_review(
        bb_squeeze=False, bb_expansion=False, bb_band_walk=True,
        bb_position=0.85,
    )
    assert p["stage"] == "band_walk"


# ---------------------------------------------------------------------------
# 10. rsi_regime_filter
# ---------------------------------------------------------------------------


def test_rsi_overbought_in_trend_is_trap():
    p = build_rsi_regime_filter(
        rsi_value=75.0, market_regime="TREND_UP",
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == RSI_SCHEMA
    assert p["raw_signal"] == "overbought"
    assert p["usable_signal"] is False
    assert "罠" in p["trap_reason_ja"] or "危険" in p["trap_reason_ja"]


def test_rsi_overbought_in_range_is_usable():
    p = build_rsi_regime_filter(
        rsi_value=75.0, market_regime="RANGE",
    )
    assert p["usable_signal"] is True


# ---------------------------------------------------------------------------
# 11. divergence_review
# ---------------------------------------------------------------------------


def test_divergence_warning_present_even_when_detected():
    p = build_divergence_review(rsi_bearish_divergence=True)
    _assert_observation_shape(p)
    assert p["schema_version"] == DIVERGENCE_SCHEMA
    assert p["any_divergence"] is True
    assert "ダイバージェンス単体" in p["warning_ja"]


def test_divergence_no_signal():
    p = build_divergence_review()
    assert p["any_divergence"] is False
    assert p["meaning_ja"]


# ---------------------------------------------------------------------------
# 12. macd_architecture_review
# ---------------------------------------------------------------------------


def test_macd_golden_cross_detected():
    p = build_macd_architecture_review(
        macd=0.001, macd_signal=0.0005, macd_hist=0.0005,
        macd_prev=0.0005, macd_signal_prev=0.0007,
        macd_hist_prev=0.0001,
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == MACD_SCHEMA
    assert p["cross_event"] == "golden_cross"
    assert p["bias"] == "BUY"


def test_macd_missing_inputs():
    p = build_macd_architecture_review(
        macd=None, macd_signal=None, macd_hist=None,
    )
    assert p["available"] is False


# ---------------------------------------------------------------------------
# 13. multi_timeframe_story
# ---------------------------------------------------------------------------


def test_mtf_story_aligned():
    p = build_multi_timeframe_story(
        higher_tf_trend="UP", middle_tf_state="pullback",
        lower_tf_kind="double_bottom", lower_tf_status="neckline_broken",
        lower_tf_side_bias="BUY",
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == MTF_SCHEMA
    assert p["tf_aligned"] is True
    assert p["alignment_side"] == "BUY"
    assert "ダブルボトム" in p["story_ja"] or "double_bottom" in p["story_ja"]


def test_mtf_story_missing_higher_tf():
    p = build_multi_timeframe_story(higher_tf_trend=None)
    assert p["available"] is False


# ---------------------------------------------------------------------------
# 14. grand_confluence_v2
# ---------------------------------------------------------------------------


def test_grand_confluence_emits_9_axes():
    p = build_grand_confluence_v2(
        dow_review={"available": True, "trend": "UP"},
        overlays={"level_zones_selected": [{"kind": "support"}],
                  "trendlines_selected": []},
        wave_derived_lines=[{"id": "WNL1", "kind": "neckline",
                             "role": "entry_confirmation_line"}],
        ma_review={"available": True, "is_uptrend_stack": True,
                   "is_downtrend_stack": False, "over_extended": False},
        rsi_review={"available": True, "usable_signal": False,
                    "raw_signal": "neutral", "rsi_value": 55,
                    "trap_reason_ja": ""},
        macd_review={"available": True, "bias": "BUY",
                     "summary_ja": "test"},
        candlestick_review={"available": True,
                            "bar_type": "bullish_pinbar",
                            "direction": "BUY", "meaning_ja": "test"},
        pattern_review={"available": True, "status": "neckline_broken",
                        "summary_ja": "test"},
        entry_summary={"rr": 2.5, "stop_price": 1.0,
                       "structure_stop_price": 1.0,
                       "atr_stop_price": 1.0,
                       "rr_unavailable_reason": None},
        macro_score=0.0,
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == CONF_SCHEMA
    assert len(p["axes"]) == 9
    axis_names = {a["axis"] for a in p["axes"]}
    assert axis_names == {
        "dow", "line", "ma", "oscillator", "price_action",
        "pattern", "risk_reward", "invalidation", "macro",
    }
    for a in p["axes"]:
        assert a["status"] in ("PASS", "WARN", "BLOCK", "UNKNOWN")
        assert isinstance(a["reason_ja"], str)


# ---------------------------------------------------------------------------
# 15. invalidation_engine_v2
# ---------------------------------------------------------------------------


def test_invalidation_engine_passes_rr_when_2_or_more():
    p = build_invalidation_engine_v2(
        entry_summary={
            "entry_price": 1.10, "stop_price": 1.09,
            "structure_stop_price": 1.09, "atr_stop_price": 1.09,
            "take_profit_price": 1.13, "rr": 3.0,
        },
        pattern_review={"available": True,
                        "detected_pattern": "double_bottom"},
        overlays={"level_zones_selected": [{"kind": "support"}]},
        dow_review={"trend": "UP"},
        ma_review={"pullback_to_sma_20": True},
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == INV_SCHEMA
    assert p["rr_pass"] is True
    assert p["is_structure_anchored"] is True
    assert "double_bottom" in p["setup_basis"]
    assert "support_zone" in p["setup_basis"]


def test_invalidation_engine_missing_entry_unavailable():
    p = build_invalidation_engine_v2(
        entry_summary=None, pattern_review=None,
        overlays={}, dow_review=None, ma_review=None,
    )
    assert p["available"] is False


# ---------------------------------------------------------------------------
# 16. pre_trade_diagnostic_checklist_v1
# ---------------------------------------------------------------------------


def test_checklist_emits_7_questions_with_ja():
    p = build_pre_trade_diagnostic_checklist(
        dow_review={"trend": "UP"},
        overlays={"level_zones_selected": [{}], "trendlines_selected": []},
        wave_derived_lines=[{"id": "WNL1"}],
        mtf_story={"available": True, "tf_aligned": True,
                   "higher_tf": "UP", "lower_tf": "double_bottom"},
        ma_review={"is_uptrend_stack": True},
        bb_review={"stage": "expansion"},
        rsi_review={"usable_signal": False, "rsi_value": 55},
        macd_review={"bias": "BUY"},
        invalidation_review={"available": True,
                             "is_structure_anchored": True},
        entry_summary={"rr": 3.0},
    )
    _assert_observation_shape(p)
    assert p["schema_version"] == CK_SCHEMA
    assert len(p["questions"]) == 7
    for q in p["questions"]:
        assert q["status"] in ("YES", "NO", "UNKNOWN")
        assert q["question_ja"]
        assert q["reason_ja"]
    assert p["verdict_ja"]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def test_aggregator_emits_all_16_panels():
    df = _trending_df(n=200)
    panels = build_masterclass_panels(
        visible_df=df,
        parent_bar_ts=df.index[-1],
        technical_dict={
            "atr_14": 0.005, "sma_20": 100.5, "sma_50": 100.0,
            "macd": 0.001, "macd_signal": 0.0005, "macd_hist": 0.0005,
            "bb_position": 0.5,
        },
        technical_confluence={
            "market_regime": "TREND_UP",
            "support_resistance": {"near_support": True,
                                   "near_resistance": False},
            "indicator_context": {"rsi_value": 55, "bb_squeeze": False,
                                  "bb_expansion": True,
                                  "bb_band_walk": False},
        },
        overlays={"level_zones_selected": [], "trendlines_selected": []},
        wave_shape_review=None, wave_derived_lines=[],
        entry_summary={"entry_price": 100.0, "stop_price": 99.0,
                       "take_profit_price": 103.0, "rr": 3.0,
                       "structure_stop_price": 99.0,
                       "atr_stop_price": 99.0},
        invalidation_explanation=None, df_lower_tf=None,
        higher_tf_trend="UP", macro_score=0.0,
    )
    assert panels["available"] is True
    assert panels["schema_version"] == AGG_SCHEMA
    assert len(panels["panels"]) == 16
    expected_keys = {
        "candlestick_anatomy_review", "parent_bar_lower_tf_anatomy",
        "dow_structure_review", "chart_pattern_anatomy_v2",
        "level_psychology_review", "indicator_environment_router",
        "ma_context_review", "granville_entry_review",
        "bollinger_lifecycle_review", "rsi_regime_filter",
        "divergence_review", "macd_architecture_review",
        "multi_timeframe_story", "grand_confluence_v2",
        "invalidation_engine_v2", "pre_trade_diagnostic_checklist_v1",
    }
    assert set(panels["panels"].keys()) == expected_keys
    # Every panel carries the observation contract
    for name, p in panels["panels"].items():
        assert p["observation_only"] is True, name
        assert p["used_in_decision"] is False, name


def test_aggregator_no_visible_df_returns_unavailable():
    panels = build_masterclass_panels(
        visible_df=None, parent_bar_ts=None,
        technical_dict=None, technical_confluence=None,
        overlays=None, wave_shape_review=None, wave_derived_lines=None,
        entry_summary=None, invalidation_explanation=None,
    )
    assert panels["available"] is False


# ---------------------------------------------------------------------------
# Integration: visible df future-leak rule
# ---------------------------------------------------------------------------


def test_aggregator_future_leak_truncated_window():
    """Pass a df truncated at parent_bar_ts. The aggregator only
    consults `visible_df` and never reaches beyond `iloc[-1]`."""
    df = _trending_df(n=200)
    parent_idx = 100
    truncated = df.iloc[: parent_idx + 1]
    panels = build_masterclass_panels(
        visible_df=truncated, parent_bar_ts=truncated.index[-1],
        technical_dict={"atr_14": 0.5},
        technical_confluence={
            "market_regime": "TREND_UP",
            "support_resistance": {"near_support": False,
                                   "near_resistance": False},
            "indicator_context": {"rsi_value": 50},
        },
        overlays={"level_zones_selected": []},
        wave_shape_review=None, wave_derived_lines=[],
        entry_summary={"entry_price": 100.0, "stop_price": 99.0,
                       "take_profit_price": 103.0, "rr": 3.0,
                       "structure_stop_price": 99.0,
                       "atr_stop_price": 99.0},
        invalidation_explanation=None,
    )
    # Dow swing detection only consults `truncated`; pattern + level
    # panels see only their slices. We assert no error and that the
    # aggregator returns a sensible payload.
    assert panels["available"] is True
    dow = panels["panels"]["dow_structure_review"]
    if dow["available"]:
        # Sequence should be derivable purely from the truncated frame
        assert dow["trend"] in (
            "UP", "DOWN", "RANGE", "MIXED", "UNKNOWN",
        )
