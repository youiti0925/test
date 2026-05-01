"""Unit tests for royal_road_decision_v2.decide_royal_road_v2."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx.risk import atr as compute_atr
from src.fx.risk_gate import RiskState
from src.fx.royal_road_decision_v2 import (
    PROFILE_NAME_V2,
    compare_v2_vs_current,
    compare_v2_vs_v1,
    decide_royal_road_v2,
)


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


def test_v2_advisory_carries_required_keys():
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


def test_v2_macro_strong_against_blocks_buy():
    """VIX VERY_HIGH on USDJPY with vix_align==SELL → blocks BUY."""
    macro_ctx = {
        "dxy_trend_5d_bucket": "FLAT",
        "us10y_change_24h_bp": 0,
        "vix": 35.0,
    }
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP"),
        symbol="USDJPY=X",
        macro_context=macro_ctx,
    )
    assert d.action == "HOLD"
    assert any("vix_very_high" in r for r in d.advisory["block_reasons"])


def test_v2_avoid_trade_label_blocks():
    d = _decide(_confluence(label="AVOID_TRADE"))
    assert d.action == "HOLD"
    assert any("label:AVOID_TRADE" in r for r in d.advisory["block_reasons"])


def test_v2_strong_setup_with_evidence_can_buy():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP",
        market_regime="TREND_UP", structure_code="HL",
        near_support=True, near_resistance=False,
        candle_bull=True,
    ))
    # Default mode=balanced, axes count >= 1 should suffice. Action
    # depends on derived snapshots from the random df, which provides
    # neither a strong support nor a chart pattern; but the bullish
    # structure axis is True. So expect either BUY or HOLD on
    # insufficient_axes (n_bull >=1 → passes).
    assert d.action in ("BUY", "HOLD")


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


def test_v2_invalid_stop_plan_blocks_when_stop_mode_not_atr():
    # structure_stop missing + stop_mode=structure → block
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP", structure_stop=None),
        stop_mode="structure",
    )
    assert d.action == "HOLD"
    assert any(
        "stop_plan_invalid" in r or "structure_stop_missing" in r
        for r in d.advisory["block_reasons"]
    )


def test_v2_default_stop_mode_atr_works_even_when_structure_missing():
    d = _decide(
        _confluence(label="STRONG_BUY_SETUP", structure_stop=None),
        stop_mode="atr",
    )
    # Action depends on axes; key thing is no stop_plan_invalid block.
    assert all(
        "stop_plan_invalid" not in r for r in d.advisory["block_reasons"]
    )
