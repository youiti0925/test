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
from src.fx.royal_road_decision_modes import (
    DEFAULT_ROYAL_ROAD_MODE,
    get_royal_road_mode_config,
    supported_royal_road_modes,
)


def _ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.3)
    return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": [1000] * n}, index=idx)


def _state() -> RiskState:
    return RiskState(df=_ohlcv(), events=(), spread_pct=None, sentiment_snapshot=None, now=datetime(2025, 1, 5, 12, 0, tzinfo=timezone.utc))


def _tc(label="STRONG_BUY_SETUP", *, side="BUY", invalidation_clear=True, structure_stop=99.0, near_support=True, near_resistance=False, avoid=None):
    bull = side == "BUY"
    return {
        "market_regime": "TREND_UP" if bull else "TREND_DOWN",
        "dow_structure": {"structure_code": "HL" if bull else "LH", "bos_up": False, "bos_down": False},
        "support_resistance": {"near_support": near_support, "near_resistance": near_resistance, "role_reversal": False},
        "candlestick_signal": {"bullish_pinbar": bull, "bullish_engulfing": False, "strong_bull_body": False, "bearish_pinbar": not bull, "bearish_engulfing": False, "strong_bear_body": False},
        "chart_pattern": {"double_top": False, "double_bottom": False, "triple_top": False, "triple_bottom": False, "neckline_broken": False},
        "risk_plan_obs": {"invalidation_clear": invalidation_clear, "structure_stop_price": structure_stop},
        "final_confluence": {"label": label, "score": 0.6 if bull else -0.6, "bullish_reasons": [], "bearish_reasons": [], "avoid_reasons": avoid or []},
    }


def _decide(tc, **kw):
    args = dict(technical_signal="HOLD", pattern=None, higher_timeframe_trend="UPTREND", risk_reward=1.5, risk_state=_state(), technical_confluence=tc)
    args.update(kw)
    return decide_royal_road(**args)


def test_default_mode_is_balanced():
    assert DEFAULT_ROYAL_ROAD_MODE == "balanced"
    assert get_royal_road_mode_config().name == "balanced"
    assert supported_royal_road_modes() == ("strict", "balanced", "exploratory")


def test_balanced_strong_buy_returns_buy():
    d = _decide(_tc("STRONG_BUY_SETUP"))
    assert d.action == "BUY"
    assert d.advisory["profile"] == PROFILE_NAME
    assert d.advisory["mode"] == "balanced"
    assert d.advisory["mode_needs_validation"] is True


def test_balanced_strong_sell_returns_sell():
    d = _decide(_tc("STRONG_SELL_SETUP", side="SELL", near_support=False, near_resistance=True), higher_timeframe_trend="DOWNTREND")
    assert d.action == "SELL"
    assert d.advisory["mode"] == "balanced"


def test_strict_weak_buy_stays_hold():
    d = _decide(_tc("WEAK_BUY_SETUP"), mode="strict")
    assert d.action == "HOLD"
    assert "weak_setup:WEAK_BUY_SETUP" in d.advisory["block_reasons"]


def test_balanced_weak_buy_can_enter_with_multiple_axes():
    d = _decide(_tc("WEAK_BUY_SETUP"))
    assert d.action == "BUY"


def test_balanced_weak_sell_can_enter_with_multiple_axes():
    d = _decide(_tc("WEAK_SELL_SETUP", side="SELL", near_support=False, near_resistance=True), higher_timeframe_trend="DOWNTREND")
    assert d.action == "SELL"


def test_balanced_near_resistance_blocks_buy():
    d = _decide(_tc("STRONG_BUY_SETUP", near_support=False, near_resistance=True))
    assert d.action == "HOLD"
    assert "near_resistance_for_buy" in d.advisory["block_reasons"]


def test_balanced_near_support_blocks_sell():
    d = _decide(_tc("STRONG_SELL_SETUP", side="SELL", near_support=True, near_resistance=False), higher_timeframe_trend="DOWNTREND")
    assert d.action == "HOLD"
    assert "near_support_for_sell" in d.advisory["block_reasons"]


def test_balanced_soft_missing_structure_stop():
    d = _decide(_tc("STRONG_BUY_SETUP", structure_stop=None))
    assert d.action == "BUY"
    assert "structure_stop_missing_soft" in d.advisory["cautions"]


def test_balanced_soft_invalidation_unclear():
    d = _decide(_tc("STRONG_BUY_SETUP", invalidation_clear=False))
    assert d.action == "BUY"
    assert "invalidation_unclear_soft" in d.advisory["cautions"]


def test_strict_still_blocks_missing_structure_stop():
    d = _decide(_tc("STRONG_BUY_SETUP", structure_stop=None), mode="strict")
    assert d.action == "HOLD"
    assert "structure_stop_missing" in d.advisory["block_reasons"]


def test_serious_avoid_blocks():
    d = _decide(_tc("STRONG_BUY_SETUP", avoid=["fake_breakout"]))
    assert d.action == "HOLD"
    assert "avoid:fake_breakout" in d.advisory["block_reasons"]


@pytest.mark.parametrize("label", ["NO_TRADE", "AVOID_TRADE", "UNKNOWN"])
def test_non_directional_labels_hold(label):
    d = _decide(_tc(label))
    assert d.action == "HOLD"
    assert f"label:{label}" in d.advisory["block_reasons"]


class _D:
    def __init__(self, action):
        self.action = action


def test_compare_decisions_taxonomy():
    assert compare_decisions(decision_current=_D("BUY"), decision_royal=_D("HOLD"))["difference_type"] == "current_buy_royal_hold"
    assert compare_decisions(decision_current=_D("HOLD"), decision_royal=_D("SELL"))["difference_type"] == "current_hold_royal_sell"
    assert compare_decisions(decision_current=_D("BUY"), decision_royal=_D("SELL"))["difference_type"] == "opposite_direction"


def test_profile_validation():
    assert "current_runtime" in SUPPORTED_DECISION_PROFILES
    assert PROFILE_NAME in SUPPORTED_DECISION_PROFILES
    assert validate_decision_profile(PROFILE_NAME) == PROFILE_NAME
    with pytest.raises(ValueError):
        validate_decision_profile("bad")
    with pytest.raises(ValueError):
        get_royal_road_mode_config("bad")
