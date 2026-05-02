"""Tests for source-pack panels (Fibonacci / Daily Roadmap / Symbol
Macro Briefing) and the 13-axis grand_confluence_v2 expansion.

Pinned invariants:
  - fibonacci_context_review emits 38.2 / 50 / 61.8 retracement
    levels and 127.2 / 161.8 extension targets
  - WFIB lines all carry used_in_decision=False
  - daily_roadmap_review marks position_sizing as 未接続 (UNKNOWN)
  - symbol_macro_briefing_review for USDJPY lists US_yield / DXY /
    JPY_risk_sentiment as drivers; for unsupported symbols emits
    no_briefing_for_symbol
  - grand_confluence_v2 returns 13 axes including fibonacci /
    mtf / roadmap / symbol_macro
  - aggregator emits all 19 panels with observation_only=True /
    used_in_decision=False
  - default current_runtime profile produces no Masterclass payload
    (covered by existing test_default_profile_produces_no_visual_audit)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fx.masterclass_aggregate import build_masterclass_panels
from src.fx.masterclass_confluence import build_grand_confluence_v2
from src.fx.source_pack_daily_roadmap import (
    SCHEMA_VERSION as ROADMAP_SCHEMA,
    build_daily_roadmap_review,
)
from src.fx.source_pack_fibonacci import (
    EXTENSION_LEVELS,
    RETRACEMENT_LEVELS,
    SCHEMA_VERSION as FIB_SCHEMA,
    build_fib_wave_lines,
    build_fibonacci_context_review,
)
from src.fx.source_pack_symbol_briefing import (
    SCHEMA_VERSION as SBR_SCHEMA,
    build_symbol_macro_briefing_review,
)


# ---------------------------------------------------------------------------
# Fibonacci
# ---------------------------------------------------------------------------


def _up_swing_skeleton() -> dict:
    return {
        "pivots": [
            {"index": 0,  "ts": "2025-01-01T00:00:00+00:00",
             "price": 1.08, "kind": "L"},
            {"index": 100, "ts": "2025-01-05T04:00:00+00:00",
             "price": 1.10, "kind": "H"},
        ],
    }


def _down_swing_skeleton() -> dict:
    return {
        "pivots": [
            {"index": 0,  "price": 1.10, "kind": "H"},
            {"index": 100, "price": 1.08, "kind": "L"},
        ],
    }


def test_fib_emits_382_500_618_retracement_and_1272_1618_extension():
    skel = _up_swing_skeleton()
    fib = build_fibonacci_context_review(
        skeleton=skel, last_close=1.0876,  # roughly 0.62 retr
    )
    assert fib["available"] is True
    assert fib["schema_version"] == FIB_SCHEMA
    levels = {l["level"] for l in fib["retracement_levels"]}
    # Stored as "38.2" / "50.0" / "61.8"
    assert {"38.2", "50.0", "61.8"} == levels
    ext_levels = {l["level"] for l in fib["extension_targets"]}
    assert {"127.2", "161.8"} == ext_levels
    # observation contract
    assert fib["observation_only"] is True
    assert fib["used_in_decision"] is False


def test_fib_default_constants_are_documented():
    assert RETRACEMENT_LEVELS == (0.382, 0.500, 0.618)
    assert EXTENSION_LEVELS == (1.272, 1.618)


def test_fib_zone_at_50_618_is_pass():
    skel = _up_swing_skeleton()
    # 0.02 swing × 0.55 retr = 0.011 → close = 1.10 - 0.011 = 1.089
    fib = build_fibonacci_context_review(skeleton=skel, last_close=1.089)
    assert fib["retracement_zone"] == "50.0-61.8"
    assert fib["status"] == "PASS"


def test_fib_no_anchor_returns_unavailable():
    fib = build_fibonacci_context_review(skeleton=None, last_close=1.10)
    assert fib["available"] is False


def test_fib_wave_lines_all_used_in_decision_false():
    skel = _up_swing_skeleton()
    fib = build_fibonacci_context_review(skeleton=skel, last_close=1.089)
    lines = build_fib_wave_lines(fibonacci_review=fib)
    assert lines, "expected WFIB lines"
    ids = {l["id"] for l in lines}
    assert {"WFIB382", "WFIB500", "WFIB618", "WFIB1272", "WFIB1618"} == ids
    for l in lines:
        assert l["used_in_decision"] is False


def test_fib_down_swing_extends_below_anchor():
    skel = _down_swing_skeleton()
    # In a DOWN swing from 1.10 to 1.08, an extension target sits
    # BELOW 1.08.
    fib = build_fibonacci_context_review(skeleton=skel, last_close=1.085)
    assert fib["available"] is True
    extensions = {float(l["price"]) for l in fib["extension_targets"]}
    # All extension prices should be below the swing low (1.08)
    for p in extensions:
        assert p < 1.08


# ---------------------------------------------------------------------------
# Daily Roadmap
# ---------------------------------------------------------------------------


def test_daily_roadmap_marks_position_sizing_as_unconnected():
    rd = build_daily_roadmap_review(
        macro_score=0.0, higher_tf_trend="UP",
        invalidation_review={"available": True,
                              "is_structure_anchored": True},
        entry_summary={"rr": 3.0},
    )
    assert rd["available"] is True
    assert rd["schema_version"] == ROADMAP_SCHEMA
    risk_item = next(
        i for i in rd["items"] if i["id"] == "risk_per_trade_defined"
    )
    assert risk_item["status"] == "UNKNOWN"
    assert "未接続" in risk_item["reason_ja"]
    # Observation contract
    assert rd["observation_only"] is True
    assert rd["used_in_decision"] is False


def test_daily_roadmap_news_calendar_unconnected():
    rd = build_daily_roadmap_review(
        macro_score=None, higher_tf_trend="UP",
        invalidation_review=None, entry_summary=None,
    )
    news_item = next(
        i for i in rd["items"] if i["id"] == "news_calendar_checked"
    )
    assert news_item["status"] == "UNKNOWN"
    assert "未接続" in news_item["reason_ja"]


def test_daily_roadmap_rr_pass_and_block():
    rd_pass = build_daily_roadmap_review(
        macro_score=0.5, higher_tf_trend="UP",
        invalidation_review={"available": True,
                              "is_structure_anchored": True},
        entry_summary={"rr": 3.0},
    )
    rr_pass_item = next(
        i for i in rd_pass["items"] if i["id"] == "rr_planned"
    )
    assert rr_pass_item["status"] == "PASS"
    rd_low = build_daily_roadmap_review(
        macro_score=0.5, higher_tf_trend="UP",
        invalidation_review={"available": True,
                              "is_structure_anchored": True},
        entry_summary={"rr": 1.2},
    )
    rr_low_item = next(
        i for i in rd_low["items"] if i["id"] == "rr_planned"
    )
    assert rr_low_item["status"] in ("NO", "BLOCK")


# ---------------------------------------------------------------------------
# Symbol Macro Briefing
# ---------------------------------------------------------------------------


def test_symbol_briefing_usdjpy_lists_drivers():
    sb = build_symbol_macro_briefing_review(
        symbol="USDJPY=X", data_available=False,
    )
    assert sb["schema_version"] == SBR_SCHEMA
    # Even with data_available=False, drivers list is exposed so the
    # user can see what to check manually.
    assert sb["macro_drivers"] == ["US_yield", "DXY", "JPY_risk_sentiment"]
    assert sb["unavailable_reason"] == "macro_briefing_data_missing"
    assert "未接続" in sb["meaning_ja"]
    assert sb["observation_only"] is True
    assert sb["used_in_decision"] is False


def test_symbol_briefing_usdjpy_data_available():
    sb = build_symbol_macro_briefing_review(
        symbol="USDJPY=X", data_available=True,
    )
    assert sb["available"] is True
    assert "US_yield" in sb["macro_drivers"]


def test_symbol_briefing_unsupported_symbol_generic():
    sb = build_symbol_macro_briefing_review(
        symbol="AUDCAD=X", data_available=False,
    )
    assert sb["unavailable_reason"] == "no_briefing_for_symbol"
    assert sb["observation_only"] is True
    assert sb["used_in_decision"] is False


def test_symbol_briefing_eurusd_lists_drivers():
    sb = build_symbol_macro_briefing_review(
        symbol="EURUSD=X", data_available=False,
    )
    assert "DXY" in sb["macro_drivers"]


# ---------------------------------------------------------------------------
# 13-axis grand_confluence_v2
# ---------------------------------------------------------------------------


def test_grand_confluence_v2_emits_13_axes_with_source_pack_additions():
    fib = build_fibonacci_context_review(
        skeleton=_up_swing_skeleton(), last_close=1.089,
    )
    rd = build_daily_roadmap_review(
        macro_score=0.0, higher_tf_trend="UP",
        invalidation_review={"available": True,
                              "is_structure_anchored": True},
        entry_summary={"rr": 3.0},
    )
    sb = build_symbol_macro_briefing_review(
        symbol="USDJPY=X", data_available=False,
    )
    conf = build_grand_confluence_v2(
        dow_review={"available": True, "trend": "UP"},
        overlays={"level_zones_selected": [{"kind": "support"}],
                  "trendlines_selected": []},
        wave_derived_lines=[],
        ma_review={"available": True, "is_uptrend_stack": True,
                   "over_extended": False},
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
        fibonacci_review=fib,
        mtf_story={"available": True, "tf_aligned": True,
                   "higher_tf": "UP"},
        roadmap_review=rd,
        symbol_briefing_review=sb,
    )
    assert len(conf["axes"]) == 13
    axis_names = {a["axis"] for a in conf["axes"]}
    expected = {
        "dow", "line", "ma", "oscillator", "price_action",
        "pattern", "risk_reward", "invalidation", "macro",
        "fibonacci", "mtf", "roadmap", "symbol_macro",
    }
    assert axis_names == expected
    # Each axis carries a what_to_check_on_chart_ja key
    for a in conf["axes"]:
        assert "what_to_check_on_chart_ja" in a


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 1.10 + np.cumsum(rng.standard_normal(n) * 0.0008)
    return pd.DataFrame({
        "open": close, "high": close + 0.0008, "low": close - 0.0008,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def test_aggregator_emits_19_panels_with_source_pack_additions():
    df = _df()
    panels = build_masterclass_panels(
        visible_df=df, parent_bar_ts=df.index[-1],
        technical_dict={"atr_14": 0.005, "sma_20": 1.099, "sma_50": 1.097,
                        "macd": 0.001, "macd_signal": 0.0005, "macd_hist": 0.0005},
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
        entry_summary={"entry_price": 1.10, "stop_price": 1.09,
                       "take_profit_price": 1.13, "rr": 3.0,
                       "structure_stop_price": 1.09,
                       "atr_stop_price": 1.09},
        invalidation_explanation=None, df_lower_tf=None,
        symbol="USDJPY=X",
    )
    assert panels["available"] is True
    assert len(panels["panels"]) == 19
    expected_keys = {
        "candlestick_anatomy_review", "parent_bar_lower_tf_anatomy",
        "dow_structure_review", "chart_pattern_anatomy_v2",
        "level_psychology_review", "indicator_environment_router",
        "ma_context_review", "granville_entry_review",
        "bollinger_lifecycle_review", "rsi_regime_filter",
        "divergence_review", "macd_architecture_review",
        "multi_timeframe_story", "grand_confluence_v2",
        "invalidation_engine_v2", "pre_trade_diagnostic_checklist_v1",
        # Source-pack additions
        "fibonacci_context_review",
        "daily_roadmap_review",
        "symbol_macro_briefing_review",
    }
    assert set(panels["panels"].keys()) == expected_keys
    for name, p in panels["panels"].items():
        assert p["observation_only"] is True, name
        assert p["used_in_decision"] is False, name
        # status / inputs_used / what_to_check_on_chart_ja contract
        assert p.get("status") in (
            "PASS", "WARN", "BLOCK", "UNKNOWN",
        ), f"{name}: {p.get('status')}"
        assert isinstance(p.get("inputs_used"), list), name
        assert isinstance(
            p.get("what_to_check_on_chart_ja"), str,
        ), name


def test_aggregator_confluence_has_13_axes_when_aggregated():
    df = _df()
    panels = build_masterclass_panels(
        visible_df=df, parent_bar_ts=df.index[-1],
        technical_dict={"atr_14": 0.005, "sma_20": 1.099, "sma_50": 1.097,
                        "macd": 0.001, "macd_signal": 0.0005, "macd_hist": 0.0005},
        technical_confluence={
            "market_regime": "TREND_UP",
            "support_resistance": {"near_support": True,
                                    "near_resistance": False},
            "indicator_context": {"rsi_value": 55, "bb_squeeze": False,
                                  "bb_expansion": True,
                                  "bb_band_walk": False},
        },
        overlays={"level_zones_selected": []},
        wave_shape_review=None, wave_derived_lines=[],
        entry_summary={"entry_price": 1.10, "stop_price": 1.09,
                       "take_profit_price": 1.13, "rr": 3.0,
                       "structure_stop_price": 1.09,
                       "atr_stop_price": 1.09},
        invalidation_explanation=None, df_lower_tf=None,
        symbol="USDJPY=X",
    )
    conf = panels["panels"]["grand_confluence_v2"]
    assert len(conf["axes"]) == 13


def test_aggregator_no_visible_df_returns_unavailable():
    panels = build_masterclass_panels(
        visible_df=None, parent_bar_ts=None,
        technical_dict=None, technical_confluence=None,
        overlays=None, wave_shape_review=None, wave_derived_lines=None,
        entry_summary=None, invalidation_explanation=None,
    )
    assert panels["available"] is False
