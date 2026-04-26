"""Tests for risk management math."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.fx.risk import (
    atr,
    atr_stop_loss,
    atr_take_profit,
    fractional_kelly,
    kelly_fraction,
    plan_trade,
    position_size,
    true_range,
)


def _df():
    data = {
        "high": [10, 11, 12, 11, 12],
        "low": [9, 10, 10, 9, 11],
        "close": [9.5, 10.5, 11.5, 10, 11.8],
        "open": [9, 10, 11, 10.5, 10.2],
        "volume": [100] * 5,
    }
    return pd.DataFrame(data)


def test_true_range_first_bar_is_high_minus_low():
    tr = true_range(_df())
    assert tr.iloc[0] == 1.0  # 10 - 9


def test_atr_is_non_negative():
    result = atr(_df(), period=2).dropna()
    assert (result >= 0).all()


def test_atr_stop_loss_buy_below_entry():
    assert atr_stop_loss(100.0, 2.0, "BUY", 2.0) == 96.0


def test_atr_stop_loss_sell_above_entry():
    assert atr_stop_loss(100.0, 2.0, "SELL", 2.0) == 104.0


def test_atr_stop_loss_rejects_zero_atr():
    with pytest.raises(ValueError):
        atr_stop_loss(100.0, 0.0, "BUY")


def test_atr_stop_loss_rejects_hold():
    with pytest.raises(ValueError):
        atr_stop_loss(100.0, 2.0, "HOLD")


def test_atr_take_profit_buy_above_entry():
    assert atr_take_profit(100.0, 2.0, "BUY", 3.0) == 106.0


def test_kelly_positive_edge():
    # 60% win rate, 1:1 payoff => k = 0.6 - 0.4/1 = 0.2
    k = kelly_fraction(0.6, 1.0, 1.0)
    assert k == pytest.approx(0.2)


def test_kelly_negative_edge_returns_zero():
    # 40% win rate with 1:1 payoff is negative expectancy
    k = kelly_fraction(0.4, 1.0, 1.0)
    assert k == 0.0


def test_kelly_uneven_payoff():
    # 50% win rate, 2:1 payoff => k = 0.5 - 0.5/2 = 0.25
    k = kelly_fraction(0.5, 2.0, 1.0)
    assert k == pytest.approx(0.25)


def test_fractional_kelly_caps_properly():
    # Very edgy bet; cap should kick in
    f = fractional_kelly(0.95, 10.0, 1.0, fraction=0.5, cap=0.1)
    assert f <= 0.1


def test_position_size_risk_equals_risk_pct():
    # $10k capital, 1% risk, $2 stop distance => 50 units, $100 at risk
    size = position_size(10_000, entry=100, stop=98, risk_pct=0.01)
    assert size == 50.0
    # Verify the actual dollar loss matches
    assert size * (100 - 98) == 100.0


def test_position_size_rejects_zero_distance():
    with pytest.raises(ValueError):
        position_size(10_000, entry=100, stop=100)


def test_plan_trade_buy_produces_coherent_plan():
    plan = plan_trade(
        side="BUY",
        entry=100.0,
        atr_value=2.0,
        capital=10_000,
        stop_mult=2.0,
        tp_mult=3.0,
        risk_pct=0.01,
    )
    assert plan.side == "BUY"
    assert plan.stop < plan.entry < plan.take_profit
    assert plan.reward_to_risk == pytest.approx(1.5)
    # Size * stop distance should equal risk_dollars
    assert plan.size * (plan.entry - plan.stop) == pytest.approx(plan.risk_dollars)


def test_plan_trade_sell_inverts_stops():
    plan = plan_trade(
        side="SELL", entry=100.0, atr_value=2.0, capital=10_000,
    )
    assert plan.take_profit < plan.entry < plan.stop


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
