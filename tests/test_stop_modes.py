"""Tests for stop_modes.plan_stop and validate_stop_mode."""
from __future__ import annotations

import pytest

from src.fx.stop_modes import (
    DEFAULT_STOP_MODE,
    SUPPORTED_STOP_MODES,
    StopPlan,
    plan_stop,
    validate_stop_mode,
)


def test_default_is_atr():
    assert DEFAULT_STOP_MODE == "atr"
    assert "atr" in SUPPORTED_STOP_MODES
    assert "structure" in SUPPORTED_STOP_MODES
    assert "hybrid" in SUPPORTED_STOP_MODES


def test_validate_rejects_unknown():
    with pytest.raises(ValueError):
        validate_stop_mode("trailing")


def test_atr_mode_buy_uses_atr_distance():
    p = plan_stop(
        mode="atr", side="BUY", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=98.5,
    )
    assert p.outcome == "atr"
    assert p.stop_price == pytest.approx(100.0 - 2.0)
    assert p.take_profit_price == pytest.approx(100.0 + 3.0)
    assert p.rr_realized == pytest.approx(1.5)


def test_atr_mode_sell_mirror():
    p = plan_stop(
        mode="atr", side="SELL", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=None,
    )
    assert p.outcome == "atr"
    assert p.stop_price == pytest.approx(100.0 + 2.0)
    assert p.take_profit_price == pytest.approx(100.0 - 3.0)


def test_structure_mode_uses_structure_when_present():
    p = plan_stop(
        mode="structure", side="BUY", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=98.5,
    )
    assert p.outcome == "structure"
    assert p.stop_price == pytest.approx(98.5)
    # tp = entry + 1.5 * structure_distance = 100 + 1.5 * 1.5 = 102.25
    assert p.take_profit_price == pytest.approx(100.0 + 1.5 * 1.5)


def test_structure_mode_invalid_when_missing():
    p = plan_stop(
        mode="structure", side="BUY", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=None,
    )
    assert p.outcome == "invalid_no_structure"
    assert p.stop_price is None
    assert p.take_profit_price is None


def test_hybrid_uses_structure_when_in_band():
    """Structure 1 ATR away (in 0.5..3.0) → use structure."""
    p = plan_stop(
        mode="hybrid", side="BUY", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=99.0,
    )
    assert p.outcome == "hybrid_structure"
    assert p.stop_price == pytest.approx(99.0)


def test_hybrid_falls_back_to_atr_when_structure_too_far():
    """Structure 5 ATR away → fallback to ATR."""
    p = plan_stop(
        mode="hybrid", side="BUY", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=95.0,
    )
    assert p.outcome == "hybrid_atr_fallback"
    assert p.stop_price == pytest.approx(100.0 - 2.0)


def test_hybrid_invalid_too_close():
    """Structure 0.2 ATR away → invalidate the trade."""
    p = plan_stop(
        mode="hybrid", side="BUY", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=99.8,
    )
    assert p.outcome == "invalid_too_close"
    assert p.stop_price is None
    assert p.take_profit_price is None
    assert p.invalidation_reason == "structure_stop_too_close"


def test_hybrid_falls_back_when_structure_missing():
    p = plan_stop(
        mode="hybrid", side="SELL", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=None,
    )
    assert p.outcome == "hybrid_atr_fallback"
    assert p.stop_price == pytest.approx(100.0 + 2.0)


def test_to_dict_emits_required_keys():
    p = plan_stop(
        mode="hybrid", side="BUY", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=99.0,
    )
    d = p.to_dict()
    for key in (
        "chosen_mode", "outcome", "stop_price", "take_profit_price",
        "rr_realized", "structure_stop_price",
        "structure_stop_distance_atr", "atr_stop_price",
        "invalidation_reason",
    ):
        assert key in d


def test_atr_stop_price_always_populated():
    """Even when chosen mode is structure, atr_stop_price must be
    available so traces can compare structure vs ATR."""
    p = plan_stop(
        mode="structure", side="BUY", entry=100.0, atr=1.0,
        stop_atr_mult=2.0, tp_atr_mult=3.0,
        structure_stop_price=98.5,
    )
    assert p.atr_stop_price == pytest.approx(100.0 - 2.0)
