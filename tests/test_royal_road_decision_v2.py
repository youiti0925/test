"""Unit tests for royal_road_decision_v2.decide_royal_road_v2.

Tests are DETERMINISTIC: each test monkeypatches the v2 sub-detectors
(detect_levels / detect_trendlines / detect_patterns / lower-TF
trigger / macro alignment) to return controlled snapshots. This
eliminates ambiguity ("BUY or HOLD" assertions are forbidden) and
makes the rule chain fully observable.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx import royal_road_decision_v2 as mod
from src.fx.chart_patterns import ChartPatternSnapshot, empty_snapshot as cp_empty
from src.fx.lower_timeframe_trigger import (
    LowerTimeframeTrigger,
    empty_trigger as ltf_empty,
)
from src.fx.macro_alignment import MacroAlignmentSnapshot, empty_alignment
from src.fx.risk import atr as compute_atr
from src.fx.risk_gate import RiskState
from src.fx.royal_road_decision_v2 import (
    PROFILE_NAME_V2,
    compare_v2_vs_current,
    compare_v2_vs_v1,
    decide_royal_road_v2,
)
from src.fx.support_resistance import Level, SRSnapshot, empty_snapshot as sr_empty
from src.fx.trendlines import TrendlineContext, empty_context as tl_empty


# ─────────── helpers ──────────────────────────────────────────────


def _ohlcv(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def _state() -> RiskState:
    return RiskState(
        df=_ohlcv(200),
        events=(),
        spread_pct=None,
        sentiment_snapshot=None,
        now=datetime(2025, 1, 5, 12, 0, tzinfo=timezone.utc),
    )


def _confluence(
    *,
    label: str = "STRONG_BUY_SETUP",
    score: float = 0.6,
    invalidation_clear: bool = True,
    structure_stop: float | None = 99.0,
    near_support: bool = True,
    near_resistance: bool = False,
    candle_bull: bool = True,
    market_regime: str = "TREND_UP",
    structure_code: str = "HL",
    avoid_reasons: list[str] | None = None,
) -> dict:
    return {
        "policy_version": "technical_confluence_v1",
        "market_regime": market_regime,
        "dow_structure": {
            "structure_code": structure_code,
            "last_swing_high": 101.0, "last_swing_low": 99.0,
            "bos_up": False, "bos_down": False,
        },
        "support_resistance": {
            "near_support": near_support,
            "near_resistance": near_resistance,
            "breakout": False, "pullback": False,
            "role_reversal": False, "fake_breakout": False,
        },
        "candlestick_signal": {
            "bullish_pinbar": candle_bull,
            "bearish_pinbar": False,
            "bullish_engulfing": False, "bearish_engulfing": False,
            "harami": False,
            "strong_bull_body": False, "strong_bear_body": False,
            "rejection_wick": False,
        },
        "chart_pattern": {},
        "indicator_context": {},
        "risk_plan_obs": {
            "atr_stop_distance_atr": 2.0,
            "structure_stop_price": structure_stop,
            "structure_stop_distance_atr": 1.2 if structure_stop else None,
            "rr_atr_based": 1.5,
            "rr_structure_based": 2.5 if structure_stop else None,
            "invalidation_clear": invalidation_clear,
        },
        "vote_breakdown": {},
        "final_confluence": {
            "label": label, "score": score,
            "bullish_reasons": [], "bearish_reasons": [],
            "avoid_reasons": avoid_reasons or [],
        },
    }


def _patch_detectors(
    monkeypatch,
    *,
    sr: SRSnapshot,
    tl: TrendlineContext,
    cp: ChartPatternSnapshot,
    ltf: LowerTimeframeTrigger | None = None,
    macro: MacroAlignmentSnapshot | None = None,
):
    """Replace the v2 detector calls with constants."""
    monkeypatch.setattr(mod, "detect_levels", lambda *a, **k: sr)
    monkeypatch.setattr(mod, "detect_trendlines", lambda *a, **k: tl)
    monkeypatch.setattr(mod, "detect_patterns", lambda *a, **k: cp)
    if ltf is not None:
        monkeypatch.setattr(mod, "detect_lower_tf_trigger", lambda *a, **k: ltf)
    if macro is not None:
        monkeypatch.setattr(
            mod, "compute_macro_alignment", lambda *a, **k: macro,
        )


def _strong_bullish_sr(near_support: bool = True, near_resistance: bool = False) -> SRSnapshot:
    """An SR snapshot with strong bullish bias (near strong support).

    confidence=0.85 + selected_level_zones_top5 populated so the v2
    reconstruction_quality computation rates this snapshot as strong
    enough to clear the _MIN_RECONSTRUCTION_SCORE gate.
    """
    lvl_sup = Level(
        price=99.0, kind="support", touch_count=5,
        first_touch_ts=None, last_touch_ts=None,
        first_touch_index=10, last_touch_index=80,
        broken_count=0, role_reversal_count=0, false_breakout_count=0,
        strength_score=4.0, recency_score=0.9,
        distance_to_close_atr=0.3,
        zone_low=98.7, zone_high=99.3, zone_width_atr=0.6,
        wick_touch_count=4, close_touch_count=5, body_break_count=0,
        wick_fakeout_count=1, rejection_count=3, confidence=0.85,
        reasons=["strong_multi_touch", "multiple_rejections"],
    )
    return SRSnapshot(
        levels=(lvl_sup,),
        nearest_support=lvl_sup if near_support else None,
        nearest_resistance=None,
        near_strong_support=near_support,
        near_strong_resistance=near_resistance,
        breakout=False, pullback=False, role_reversal=False,
        fake_breakout=False, reason="ok",
        selected_level_zones_top5=(lvl_sup,) if near_support else (),
        rejected_level_zones=(),
    )


def _strong_bearish_sr() -> SRSnapshot:
    lvl_res = Level(
        price=101.0, kind="resistance", touch_count=5,
        first_touch_ts=None, last_touch_ts=None,
        first_touch_index=10, last_touch_index=80,
        broken_count=0, role_reversal_count=0, false_breakout_count=0,
        strength_score=4.0, recency_score=0.9,
        distance_to_close_atr=0.3,
        zone_low=100.7, zone_high=101.3, zone_width_atr=0.6,
        wick_touch_count=4, close_touch_count=5, body_break_count=0,
        wick_fakeout_count=1, rejection_count=3, confidence=0.85,
        reasons=["strong_multi_touch"],
    )
    return SRSnapshot(
        levels=(lvl_res,),
        nearest_support=None,
        nearest_resistance=lvl_res,
        near_strong_support=False,
        near_strong_resistance=True,
        breakout=False, pullback=False, role_reversal=False,
        fake_breakout=False, reason="ok",
        selected_level_zones_top5=(lvl_res,),
        rejected_level_zones=(),
    )


def _decide(tc: dict, **overrides):
    df = _ohlcv()
    kwargs = dict(
        df_window=df,
        technical_confluence=tc,
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=1.5,
        risk_state=_state(),
        atr_value=float(compute_atr(df, 14).iloc[-1] or 0.5),
        last_close=float(df["close"].iloc[-1]),
        symbol="EURUSD=X",
        macro_context=None,
        df_lower_tf=None,
        lower_tf_interval=None,
        base_bar_close_ts=df.index[-1],
        mode="balanced",
    )
    kwargs.update(overrides)
    return decide_royal_road_v2(**kwargs)


# ─────────── advisory shape ───────────────────────────────────────


def test_v2_advisory_carries_required_keys(monkeypatch):
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(_confluence(label="STRONG_BUY_SETUP"))
    adv = d.advisory
    for key in (
        "profile", "mode", "mode_status", "mode_needs_validation",
        "score", "reasons", "block_reasons", "cautions",
        "evidence_axes", "evidence_axes_count",
        "min_evidence_axes_required",
        "support_resistance_v2", "trendline_context",
        "chart_pattern_v2", "lower_tf_trigger", "macro_alignment",
        "structure_stop_plan", "source",
    ):
        assert key in adv, f"v2 advisory missing key: {key}"
    assert adv["profile"] == PROFILE_NAME_V2


# ─────────── deterministic BUY ─────────────────────────────────────


def test_v2_strong_buy_with_strong_support_buys(monkeypatch):
    """Strong bullish SR + bullish HTF + STRONG_BUY_SETUP → BUY."""
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(_confluence(label="STRONG_BUY_SETUP"))
    assert d.action == "BUY", (
        f"expected BUY; got {d.action} with block_reasons={d.advisory['block_reasons']}"
    )


# ─────────── deterministic SELL ───────────────────────────────────


def test_v2_strong_sell_with_strong_resistance_sells(monkeypatch):
    _patch_detectors(
        monkeypatch,
        sr=_strong_bearish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(
        _confluence(
            label="STRONG_SELL_SETUP", score=-0.6,
            near_support=False, near_resistance=True,
            market_regime="TREND_DOWN", structure_code="LH",
            candle_bull=False,
        ),
        higher_timeframe_trend="DOWNTREND",
    )
    assert d.action == "SELL", (
        f"expected SELL; got {d.action} with block_reasons={d.advisory['block_reasons']}"
    )


# ─────────── deterministic HOLD paths ─────────────────────────────


def test_v2_avoid_trade_label_holds(monkeypatch):
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(_confluence(label="AVOID_TRADE"))
    assert d.action == "HOLD"
    assert any("label:AVOID_TRADE" in r for r in d.advisory["block_reasons"])


def test_v2_no_trade_label_holds(monkeypatch):
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(_confluence(label="NO_TRADE"))
    assert d.action == "HOLD"
    assert any("label:NO_TRADE" in r for r in d.advisory["block_reasons"])


def test_v2_unknown_label_holds(monkeypatch):
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(_confluence(label="UNKNOWN"))
    assert d.action == "HOLD"


def test_v2_near_strong_resistance_blocks_buy(monkeypatch):
    """Even with STRONG_BUY_SETUP, near_strong_resistance blocks."""
    _patch_detectors(
        monkeypatch,
        sr=_strong_bearish_sr(),     # near_strong_resistance = True
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(_confluence(label="STRONG_BUY_SETUP"))
    assert d.action == "HOLD"
    assert "near_strong_resistance_for_buy" in d.advisory["block_reasons"]


def test_v2_near_strong_support_blocks_sell(monkeypatch):
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),      # near_strong_support = True
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(
        _confluence(
            label="STRONG_SELL_SETUP",
            near_support=False, near_resistance=False,
            candle_bull=False,
        ),
        higher_timeframe_trend="DOWNTREND",
    )
    assert d.action == "HOLD"
    assert "near_strong_support_for_sell" in d.advisory["block_reasons"]


def test_v2_macro_strong_against_blocks_buy(monkeypatch):
    """USDJPY VERY_HIGH VIX → macro_strong_against=BUY → block."""
    macro_ctx = {
        "dxy_trend_5d_bucket": "FLAT",
        "us10y_change_24h_bp": 0,
        "vix": 35.0,
    }
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP"),
        symbol="USDJPY=X",
        macro_context=macro_ctx,
    )
    assert d.action == "HOLD"
    assert any(
        "vix_very_high" in r for r in d.advisory["block_reasons"]
    )


def test_v2_invalid_stop_plan_blocks_when_stop_mode_not_atr(monkeypatch):
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP", structure_stop=None),
        stop_mode="structure",
    )
    assert d.action == "HOLD"
    assert any(
        "stop_plan_invalid" in r or "structure_stop_missing" in r
        for r in d.advisory["block_reasons"]
    )


def test_v2_default_stop_mode_atr_does_not_invalidate_on_missing_structure(monkeypatch):
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP", structure_stop=None),
        stop_mode="atr",
    )
    assert all(
        "stop_plan_invalid" not in r for r in d.advisory["block_reasons"]
    )


def test_v2_htf_counter_trend_blocks(monkeypatch):
    _patch_detectors(
        monkeypatch,
        sr=_strong_bullish_sr(),
        tl=tl_empty("ok"),
        cp=cp_empty("ok"),
    )
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP"),
        higher_timeframe_trend="DOWNTREND",
    )
    assert d.action == "HOLD"
    assert "htf_counter_trend_for_buy" in d.advisory["block_reasons"]


def test_v2_compare_taxonomy_is_closed():
    class Dummy:
        def __init__(self, action): self.action = action
    out = compare_v2_vs_current(
        decision_current=Dummy("BUY"),
        decision_v2=Dummy("HOLD"),
    )
    assert out["difference_type"] == "current_buy_royal_hold"


def test_v2_vs_v1_taxonomy():
    class Dummy:
        def __init__(self, action): self.action = action
    out = compare_v2_vs_v1(
        decision_v1=Dummy("HOLD"),
        decision_v2=Dummy("BUY"),
    )
    assert out["difference_type"] == "v1_hold_v2_buy"
