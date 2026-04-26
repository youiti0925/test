"""Tests for the broker abstraction and PaperBroker.

OANDA safety guards are tested without instantiating the live client.
"""
from __future__ import annotations

import os

import pytest

from src.fx.broker import PaperBroker
from src.fx.oanda import (
    LiveTradingBlocked,
    OANDABroker,
    OANDAConfig,
    _to_oanda_instrument,
)


# --- PaperBroker ---


def test_paper_broker_starts_with_initial_cash():
    b = PaperBroker(initial_cash=5000)
    assert b.balance() == 5000


def test_place_order_creates_open_position():
    b = PaperBroker()
    pos = b.place_order("USDJPY=X", "BUY", size=1000, price=150.0)
    assert pos.id == 1
    assert len(b.open_positions()) == 1


def test_place_order_rejects_invalid_side():
    b = PaperBroker()
    with pytest.raises(ValueError):
        b.place_order("USDJPY=X", "HOLD", 1000, 150.0)


def test_place_order_rejects_non_positive_size():
    b = PaperBroker()
    with pytest.raises(ValueError):
        b.place_order("USDJPY=X", "BUY", 0, 150.0)


def test_close_position_realizes_pnl_buy():
    b = PaperBroker(initial_cash=10_000)
    pos = b.place_order("USDJPY=X", "BUY", size=100, price=150.0)
    closed = b.close_position(pos.id, price=151.0)
    # 100 units * 1.0 move = 100 PnL
    assert closed.pnl == pytest.approx(100.0)
    assert b.balance() == pytest.approx(10_100.0)


def test_close_position_realizes_pnl_sell():
    b = PaperBroker(initial_cash=10_000)
    pos = b.place_order("USDJPY=X", "SELL", size=100, price=150.0)
    closed = b.close_position(pos.id, price=149.0)
    assert closed.pnl == pytest.approx(100.0)


def test_close_unknown_position_raises():
    b = PaperBroker()
    with pytest.raises(KeyError):
        b.close_position(999, 100.0)


def test_mark_to_market_triggers_stop_buy():
    b = PaperBroker()
    b.place_order("X", "BUY", 100, price=100.0, stop=98.0)
    b.mark_to_market({"X": 97.5})
    assert len(b.open_positions()) == 0
    assert len(b.closed_positions()) == 1


def test_mark_to_market_triggers_take_profit_sell():
    b = PaperBroker()
    b.place_order("X", "SELL", 100, price=100.0, take_profit=95.0)
    b.mark_to_market({"X": 94.5})
    assert len(b.open_positions()) == 0


def test_mark_to_market_no_trigger_leaves_position_open():
    b = PaperBroker()
    b.place_order("X", "BUY", 100, price=100.0, stop=98.0, take_profit=105.0)
    b.mark_to_market({"X": 101.0})
    assert len(b.open_positions()) == 1


# --- OANDA safety guards ---


def test_oanda_broker_refuses_without_confirm():
    with pytest.raises(LiveTradingBlocked):
        OANDABroker(
            OANDAConfig(api_key="fake", account_id="fake", env="practice"),
            confirm_demo=False,
        )


def test_oanda_config_from_env_rejects_live_env(monkeypatch):
    monkeypatch.setenv("OANDA_API_KEY", "k")
    monkeypatch.setenv("OANDA_ACCOUNT_ID", "a")
    monkeypatch.setenv("OANDA_ENV", "live")
    with pytest.raises(LiveTradingBlocked):
        OANDAConfig.from_env()


def test_oanda_config_from_env_missing_credentials(monkeypatch):
    monkeypatch.delenv("OANDA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OANDAConfig.from_env()


def test_oanda_broker_requires_allow_live_orders_env(monkeypatch):
    monkeypatch.delenv("OANDA_ALLOW_LIVE_ORDERS", raising=False)
    with pytest.raises(LiveTradingBlocked):
        OANDABroker(
            OANDAConfig(api_key="k", account_id="a", env="practice"),
            confirm_demo=True,
        )


def test_to_oanda_instrument_conversions():
    assert _to_oanda_instrument("USDJPY=X") == "USD_JPY"
    assert _to_oanda_instrument("BTC-USD") == "BTC_USD"
    assert _to_oanda_instrument("EUR_USD") == "EUR_USD"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
