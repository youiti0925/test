"""Tests for v2 setup_candidates / best_setup / reconstruction_quality
+ HOLD on insufficient quality + lower_tf trigger metadata."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx import royal_road_decision_v2 as mod
from src.fx.chart_patterns import empty_snapshot as cp_empty
from src.fx.lower_timeframe_trigger import (
    LowerTimeframeTrigger,
    detect_lower_tf_trigger,
)
from src.fx.macro_alignment import empty_alignment
from src.fx.risk_gate import RiskState
from src.fx.royal_road_decision_v2 import (
    decide_royal_road_v2,
)
from src.fx.support_resistance import Level, SRSnapshot
from src.fx.trendlines import empty_context as tl_empty


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
        df=_ohlcv(200), events=(),
        spread_pct=None, sentiment_snapshot=None,
        now=datetime(2025, 1, 5, 12, 0, tzinfo=timezone.utc),
    )


def _confluence(label: str = "STRONG_BUY_SETUP", structure_stop: float | None = 99.0) -> dict:
    return {
        "policy_version": "technical_confluence_v1",
        "market_regime": "TREND_UP",
        "dow_structure": {"structure_code": "HL"},
        "support_resistance": {
            "near_support": True, "near_resistance": False,
            "role_reversal": False, "fake_breakout": False,
        },
        "candlestick_signal": {"bullish_pinbar": True},
        "chart_pattern": {},
        "indicator_context": {},
        "risk_plan_obs": {
            "atr_stop_distance_atr": 2.0,
            "structure_stop_price": structure_stop,
            "structure_stop_distance_atr": 1.2 if structure_stop else None,
            "rr_atr_based": 1.5,
            "rr_structure_based": 2.5 if structure_stop else None,
            "invalidation_clear": True,
        },
        "vote_breakdown": {},
        "final_confluence": {
            "label": label, "score": 0.6,
            "bullish_reasons": [], "bearish_reasons": [], "avoid_reasons": [],
        },
    }


def _strong_bullish_sr() -> SRSnapshot:
    lvl = Level(
        price=99.0, kind="support", touch_count=5,
        first_touch_ts=None, last_touch_ts=None,
        first_touch_index=10, last_touch_index=80,
        broken_count=0, role_reversal_count=1, false_breakout_count=0,
        strength_score=4.0, recency_score=0.9,
        distance_to_close_atr=0.3,
        zone_low=98.7, zone_high=99.3, zone_width_atr=0.6,
        wick_touch_count=4, close_touch_count=5, body_break_count=0,
        wick_fakeout_count=1, rejection_count=3, confidence=0.85,
        reasons=["strong_multi_touch", "multiple_rejections"],
    )
    return SRSnapshot(
        levels=(lvl,),
        nearest_support=lvl, nearest_resistance=None,
        near_strong_support=True, near_strong_resistance=False,
        breakout=False, pullback=False, role_reversal=False,
        fake_breakout=False, reason="ok",
        selected_level_zones_top5=(lvl,),
        rejected_level_zones=(),
    )


def _decide(tc: dict, *, monkeypatch=None, **overrides):
    df = _ohlcv()
    kwargs = dict(
        df_window=df,
        technical_confluence=tc,
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=1.5,
        risk_state=_state(),
        atr_value=0.5,
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


def test_v2_advisory_carries_setup_candidates_and_quality(monkeypatch):
    monkeypatch.setattr(mod, "detect_levels", lambda *a, **k: _strong_bullish_sr())
    monkeypatch.setattr(mod, "detect_trendlines", lambda *a, **k: tl_empty("ok"))
    monkeypatch.setattr(mod, "detect_patterns", lambda *a, **k: cp_empty("ok"))
    d = _decide(_confluence(label="STRONG_BUY_SETUP"))
    adv = d.advisory
    assert "setup_candidates" in adv
    assert "best_setup" in adv
    assert "reconstruction_quality" in adv
    assert "multi_scale_chart" in adv
    rq = adv["reconstruction_quality"]
    for key in (
        "level_quality_score", "trendline_quality_score",
        "pattern_quality_score", "lower_tf_quality_score",
        "macro_quality_score", "stop_plan_quality_score",
        "total_reconstruction_score", "weights",
    ):
        assert key in rq


def test_best_setup_selected_when_quality_passes(monkeypatch):
    monkeypatch.setattr(mod, "detect_levels", lambda *a, **k: _strong_bullish_sr())
    monkeypatch.setattr(mod, "detect_trendlines", lambda *a, **k: tl_empty("ok"))
    monkeypatch.setattr(mod, "detect_patterns", lambda *a, **k: cp_empty("ok"))
    d = _decide(_confluence(label="STRONG_BUY_SETUP"))
    adv = d.advisory
    assert d.action == "BUY"
    assert adv["best_setup"] is not None
    assert adv["best_setup"]["side"] == "BUY"
    assert adv["best_setup"]["reject_reasons"] == []


def test_low_reconstruction_quality_holds(monkeypatch):
    """All sub-detectors return empty / weak → total_reconstruction_score
    falls below threshold → HOLD with the documented block reason."""
    monkeypatch.setattr(
        mod, "detect_levels",
        lambda *a, **k: SRSnapshot(
            levels=(), nearest_support=None, nearest_resistance=None,
            near_strong_support=False, near_strong_resistance=False,
            breakout=False, pullback=False, role_reversal=False,
            fake_breakout=False, reason="empty",
            selected_level_zones_top5=(), rejected_level_zones=(),
        )
    )
    monkeypatch.setattr(mod, "detect_trendlines", lambda *a, **k: tl_empty("empty"))
    monkeypatch.setattr(mod, "detect_patterns", lambda *a, **k: cp_empty("empty"))
    # No structure stop → stop_plan invalid in non-atr mode; or atr mode
    # gives 1.0 quality. Force structure stop missing AND macro empty.
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP", structure_stop=None),
        stop_mode="atr",     # so the stop_plan still emits valid output
    )
    adv = d.advisory
    # Total quality should be low (level=0, trendline=0, pattern=0,
    # lower_tf=0, macro=0, stop=1.0 weighted only). Threshold = 0.40.
    assert adv["reconstruction_quality"]["total_reconstruction_score"] < 0.40
    assert d.action == "HOLD"
    assert "insufficient_royal_road_reconstruction_quality" in adv["block_reasons"]


def test_lower_tf_trigger_ts_does_not_exceed_parent_bar_ts():
    """lower_tf trigger metadata must report parent_bar_ts and
    trigger_ts, and trigger_ts <= parent_bar_ts (no future leak)."""
    rng = np.random.default_rng(7)
    n = 120
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    closes = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
    df_lower = pd.DataFrame({
        "open": closes - 0.02, "high": closes + 0.1,
        "low": closes - 0.1, "close": closes, "volume": [100] * n,
    }, index=idx)
    parent_bar = idx[80]   # cut off mid-history; future bars must NOT influence
    out = detect_lower_tf_trigger(
        df_lower_tf=df_lower, lower_tf_interval="15m",
        base_bar_close_ts=parent_bar,
    )
    d = out.to_dict()
    # New metadata fields must be present.
    for key in (
        "parent_bar_ts", "used_bars_start_ts", "used_bars_end_ts",
        "trigger_ts", "trigger_price", "trigger_type",
        "trigger_strength", "confidence", "reason",
    ):
        assert key in d
    # parent_bar_ts equals the supplied cutoff.
    assert d["parent_bar_ts"] == parent_bar.isoformat()
    # used_bars_end_ts is at or before the cutoff (no future leak).
    if d["used_bars_end_ts"] is not None:
        assert d["used_bars_end_ts"] <= d["parent_bar_ts"]
    # trigger_ts is at or before the cutoff (no future leak).
    if d["trigger_ts"] is not None:
        assert d["trigger_ts"] <= d["parent_bar_ts"]


def test_setup_candidates_carries_per_axis_context(monkeypatch):
    monkeypatch.setattr(mod, "detect_levels", lambda *a, **k: _strong_bullish_sr())
    monkeypatch.setattr(mod, "detect_trendlines", lambda *a, **k: tl_empty("ok"))
    monkeypatch.setattr(mod, "detect_patterns", lambda *a, **k: cp_empty("ok"))
    d = _decide(_confluence(label="STRONG_BUY_SETUP"))
    cands = d.advisory["setup_candidates"]
    assert len(cands) >= 1
    c = cands[0]
    for key in (
        "side", "score", "confidence", "label",
        "level_context", "trendline_context", "pattern_context",
        "lower_tf_context", "macro_context", "stop_plan",
        "rr", "reasons", "reject_reasons",
    ):
        assert key in c
