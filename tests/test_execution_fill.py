"""Tests for the execution-fill audit (spec §12 / §16).

Pinned guarantees:
  * PaperBroker.place_order populates `last_fill` with the synthetic
    zero-spread / zero-slippage / zero-latency record.
  * `save_trade` persists every audit field round-trip.
  * Slippage and latency derive correctly from request/response/fill.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.fx.broker import ExecutionFill, PaperBroker
from src.fx.storage import Storage


def test_paperbroker_populates_last_fill():
    broker = PaperBroker(initial_cash=10_000.0)
    pos = broker.place_order(
        symbol="USDJPY=X", side="BUY", size=100.0, price=150.0,
    )
    fill = broker.last_execution_fill()
    assert fill is not None
    assert fill.symbol == "USDJPY=X"
    assert fill.side == "BUY"
    assert fill.expected_entry_price == 150.0
    assert fill.actual_fill_price == 150.0
    assert fill.slippage == 0.0
    assert fill.spread_pct == 0.0
    assert fill.broker_order_id is None  # paper has no real id
    # Sanity on the position too
    assert pos.entry == 150.0


def test_execution_fill_slippage_signed_by_side():
    base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    buy = ExecutionFill(
        symbol="X", side="BUY",
        expected_entry_price=100.0, actual_fill_price=100.05,
        size=1.0, bid=100.0, ask=100.05, spread_pct=0.05,
        broker_order_id="A1",
        order_request_time=base, order_response_time=base, fill_time=base,
    )
    sell = ExecutionFill(
        symbol="X", side="SELL",
        expected_entry_price=100.0, actual_fill_price=99.95,
        size=1.0, bid=99.95, ask=100.0, spread_pct=0.05,
        broker_order_id="A2",
        order_request_time=base, order_response_time=base, fill_time=base,
    )
    # BUY paid above expected → positive slippage
    assert buy.slippage == pytest.approx(0.05, abs=1e-9)
    # SELL filled below expected → also positive slippage (we got LESS than ask)
    assert sell.slippage == pytest.approx(0.05, abs=1e-9)


def test_execution_fill_latency_ms_from_request_response():
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    fill = ExecutionFill(
        symbol="X", side="BUY",
        expected_entry_price=1.0, actual_fill_price=1.0,
        size=1.0, bid=1.0, ask=1.0, spread_pct=0.0,
        broker_order_id="x",
        order_request_time=base,
        order_response_time=base + timedelta(milliseconds=83),
        fill_time=base + timedelta(milliseconds=85),
    )
    assert fill.execution_latency_ms == pytest.approx(83.0, abs=1e-3)


def test_save_trade_persists_full_fill_audit(tmp_path: Path):
    storage = Storage(tmp_path / "fx.db")
    base = datetime(2025, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    tid = storage.save_trade(
        symbol="USDJPY=X", side="BUY",
        entry=150.05, size=1000.0, opened_at=base,
        broker="oanda",
        expected_entry_price=150.00,
        actual_fill_price=150.05,
        slippage=0.05,
        bid_at_entry=150.00, ask_at_entry=150.06, spread_pct_at_entry=0.04,
        broker_order_id="OANDA-9123",
        order_request_time=base,
        order_response_time=base + timedelta(milliseconds=120),
        fill_time=base + timedelta(milliseconds=125),
        execution_latency_ms=120.0,
    )
    assert tid >= 1
    rows = storage.recent_trades(limit=10)
    # recent_trades only returns CLOSED trades, so close it first
    storage.close_trade(
        tid, exit_price=150.50,
        closed_at=base + timedelta(hours=2),
        bid_at_exit=150.49, ask_at_exit=150.51, spread_pct_at_exit=0.013,
    )
    rows = storage.recent_trades(limit=10)
    assert len(rows) == 1
    r = rows[0]
    assert r["broker"] == "oanda"
    assert r["expected_entry_price"] == 150.00
    assert r["actual_fill_price"] == 150.05
    assert r["slippage"] == 0.05
    assert r["broker_order_id"] == "OANDA-9123"
    assert r["execution_latency_ms"] == 120.0
    assert r["spread_pct_at_entry"] == 0.04
    assert r["bid_at_exit"] == 150.49
    assert r["ask_at_exit"] == 150.51
    assert r["spread_pct_at_exit"] == 0.013


def test_save_trade_works_without_audit_fields(tmp_path: Path):
    """Old code paths (no audit kwargs) must keep working."""
    storage = Storage(tmp_path / "fx.db")
    tid = storage.save_trade(
        symbol="X", side="BUY", entry=1.0, size=1.0,
        opened_at=datetime.now(timezone.utc),
    )
    assert tid >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
