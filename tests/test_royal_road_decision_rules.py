"""Unit tests for `royal_road_decision.decide_royal_road` rule chain.

Each test feeds a hand-built `technical_confluence_v1` dict, an empty
df-based RiskState (so the gate passes), and asserts the rule output
matches the royal-road v1 spec (see `docs/royal_road_decision_v1.md`).

These tests do NOT exercise the engine wiring — that belongs in
`test_royal_road_decision_engine_invariant.py` and
`test_royal_road_decision_compare_trace.py`.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.fx.risk_gate import RiskState
from src.fx.royal_road_decision import (
    PROFILE_NAME,
    SUPPORTED_DECISION_PROFILES,
    compare_decisions,
    decide_royal_road,
    validate_decision_profile,
)


# ──────────── helpers ──────────────────────────────────────────────


def _ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    return pd.DataFrame({
        "open": close, "high": close + 0.3, "low": close - 0.3,
        "close": close, "volume": [1000] * n,
    }, index=idx)


def _state() -> RiskState:
    """Minimal RiskState whose gate passes (no events, no spread, no
    recent loss streak)."""
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
    candle_bull: bool = False,
    candle_bear: bool = False,
    market_regime: str = "TREND_UP",
    structure_code: str = "HL",
    avoid_reasons: list[str] | None = None,
    bullish_reasons: list[str] | None = None,
    bearish_reasons: list[str] | None = None,
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
            "bearish_pinbar": candle_bear,
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
            "bullish_reasons": bullish_reasons or [],
            "bearish_reasons": bearish_reasons or [],
            "avoid_reasons":   avoid_reasons or [],
        },
    }


def _decide(tc: dict, **overrides):
    kwargs = dict(
        technical_signal="HOLD",
        pattern=None,
        higher_timeframe_trend="UPTREND",
        risk_reward=1.5,
        risk_state=_state(),
        technical_confluence=tc,
    )
    kwargs.update(overrides)
    return decide_royal_road(**kwargs)


# ─────────────── happy paths ───────────────────────────────────────


def test_strong_buy_with_full_evidence_returns_buy():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP", near_support=True,
        market_regime="TREND_UP", structure_code="HL",
        candle_bull=True, score=0.7,
    ))
    assert d.action == "BUY"
    assert d.advisory["profile"] == PROFILE_NAME
    assert d.advisory["score"] == 0.7
    assert "STRONG_BUY_SETUP_confirmed" in d.advisory["reasons"]
    assert d.advisory["block_reasons"] == []
    assert 0.55 <= d.confidence <= 0.95


def test_strong_sell_with_full_evidence_returns_sell():
    d = _decide(
        _confluence(
            label="STRONG_SELL_SETUP",
            near_support=False, near_resistance=True,
            market_regime="TREND_DOWN", structure_code="LH",
            candle_bull=False, candle_bear=True, score=-0.6,
        ),
        higher_timeframe_trend="DOWNTREND",
    )
    assert d.action == "SELL"
    assert d.advisory["profile"] == PROFILE_NAME


# ─────────────── strict v1 rejections (HOLD) ───────────────────────


def test_weak_buy_setup_returns_hold():
    d = _decide(_confluence(label="WEAK_BUY_SETUP"))
    assert d.action == "HOLD"
    assert any("weak_setup:WEAK_BUY_SETUP" in r for r in d.advisory["block_reasons"])


def test_weak_sell_setup_returns_hold():
    d = _decide(_confluence(label="WEAK_SELL_SETUP"))
    assert d.action == "HOLD"
    assert any("weak_setup:WEAK_SELL_SETUP" in r for r in d.advisory["block_reasons"])


def test_no_trade_label_returns_hold():
    d = _decide(_confluence(label="NO_TRADE"))
    assert d.action == "HOLD"


def test_avoid_trade_label_returns_hold():
    d = _decide(_confluence(label="AVOID_TRADE"))
    assert d.action == "HOLD"


def test_unknown_label_returns_hold():
    d = _decide(_confluence(label="UNKNOWN"))
    assert d.action == "HOLD"


def test_invalidation_unclear_blocks_buy():
    d = _decide(_confluence(label="STRONG_BUY_SETUP", invalidation_clear=False))
    assert d.action == "HOLD"
    assert "invalidation_unclear" in d.advisory["block_reasons"]


def test_structure_stop_missing_blocks_buy():
    d = _decide(_confluence(label="STRONG_BUY_SETUP", structure_stop=None))
    assert d.action == "HOLD"
    assert "structure_stop_missing" in d.advisory["block_reasons"]


def test_near_resistance_blocks_strong_buy():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP",
        near_support=False, near_resistance=True,
        market_regime="TREND_UP", structure_code="HL",
    ))
    assert d.action == "HOLD"
    assert "near_resistance_for_buy" in d.advisory["block_reasons"]


def test_near_support_blocks_strong_sell():
    d = _decide(
        _confluence(
            label="STRONG_SELL_SETUP",
            near_support=True, near_resistance=False,
            market_regime="TREND_DOWN", structure_code="LH",
            candle_bear=True, score=-0.6,
        ),
        higher_timeframe_trend="DOWNTREND",
    )
    assert d.action == "HOLD"
    assert "near_support_for_sell" in d.advisory["block_reasons"]


def test_no_bullish_evidence_blocks_buy():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP",
        near_support=False, near_resistance=False,
        market_regime="RANGE", structure_code="MIXED",
        candle_bull=False,
    ))
    assert d.action == "HOLD"
    assert "no_bullish_evidence" in d.advisory["block_reasons"]


def test_htf_counter_trend_blocks_buy():
    d = _decide(
        _confluence(
            label="STRONG_BUY_SETUP",
            near_support=True, market_regime="TREND_UP",
            structure_code="HL", candle_bull=True,
        ),
        higher_timeframe_trend="DOWNTREND",
    )
    assert d.action == "HOLD"
    assert "htf_counter_trend_for_buy" in d.advisory["block_reasons"]


def test_serious_avoid_blocks_strong_buy():
    d = _decide(_confluence(
        label="STRONG_BUY_SETUP",
        near_support=True, market_regime="TREND_UP",
        structure_code="HL", candle_bull=True,
        avoid_reasons=["fake_breakout"],
    ))
    assert d.action == "HOLD"
    assert any("fake_breakout" in r for r in d.advisory["block_reasons"])


def test_rr_below_floor_blocks():
    d = _decide(
        _confluence(
            label="STRONG_BUY_SETUP", near_support=True,
            market_regime="TREND_UP", structure_code="HL",
            candle_bull=True,
        ),
        risk_reward=0.5,
    )
    assert d.action == "HOLD"
    assert any("rr<" in r for r in d.advisory["block_reasons"])


# ─────────────── compare_decisions taxonomy ───────────────────────


class _Dummy:
    def __init__(self, action: str) -> None:
        self.action = action


def test_compare_same_action_same():
    out = compare_decisions(decision_current=_Dummy("BUY"), decision_royal=_Dummy("BUY"))
    assert out["difference_type"] == "same"
    assert out["same_action"] is True


def test_compare_current_buy_royal_hold():
    out = compare_decisions(
        decision_current=_Dummy("BUY"), decision_royal=_Dummy("HOLD"),
    )
    assert out["difference_type"] == "current_buy_royal_hold"


def test_compare_current_hold_royal_sell():
    out = compare_decisions(
        decision_current=_Dummy("HOLD"), decision_royal=_Dummy("SELL"),
    )
    assert out["difference_type"] == "current_hold_royal_sell"


def test_compare_opposite_direction():
    out = compare_decisions(
        decision_current=_Dummy("BUY"), decision_royal=_Dummy("SELL"),
    )
    assert out["difference_type"] == "opposite_direction"


# ─────────────── profile validation ────────────────────────────────


def test_supported_profiles_includes_default_and_royal():
    assert "current_runtime" in SUPPORTED_DECISION_PROFILES
    assert PROFILE_NAME in SUPPORTED_DECISION_PROFILES


def test_validate_decision_profile_rejects_unknown():
    with pytest.raises(ValueError):
        validate_decision_profile("nonexistent_profile")


def test_validate_decision_profile_returns_canonical_name():
    assert validate_decision_profile("current_runtime") == "current_runtime"
    assert validate_decision_profile("royal_road_decision_v1") == PROFILE_NAME
