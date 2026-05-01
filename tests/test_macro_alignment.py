"""Tests for macro_alignment.compute_macro_alignment (v2, symbol-aware)."""
from __future__ import annotations

import pytest

from src.fx.macro_alignment import compute_macro_alignment, empty_alignment


def _macro_dict(*, dxy_bucket: str | None = None, us10y_change_bp: float | None = None,
                vix: float | None = None) -> dict:
    return {
        "dxy_trend_5d_bucket": dxy_bucket,
        "us10y_change_24h_bp": us10y_change_bp,
        "vix": vix,
    }


def test_no_macro_returns_unknown():
    a = compute_macro_alignment(symbol="USDJPY=X", macro_context=None)
    assert a.dxy_alignment == "UNKNOWN"
    assert a.yield_alignment == "UNKNOWN"
    assert a.vix_regime == "UNKNOWN"
    assert a.macro_score == 0.0


def test_usdjpy_dxy_up_is_buy_leaning():
    a = compute_macro_alignment(
        symbol="USDJPY=X",
        macro_context=_macro_dict(dxy_bucket="STRONG_UP", us10y_change_bp=0, vix=18.0),
    )
    assert a.dxy_alignment == "BUY"


def test_eurusd_dxy_up_is_sell_leaning():
    a = compute_macro_alignment(
        symbol="EURUSD=X",
        macro_context=_macro_dict(dxy_bucket="UP", us10y_change_bp=0, vix=18.0),
    )
    assert a.dxy_alignment == "SELL"


def test_gbpusd_dxy_up_is_sell_leaning():
    a = compute_macro_alignment(
        symbol="GBPUSD=X",
        macro_context=_macro_dict(dxy_bucket="UP", us10y_change_bp=0, vix=18.0),
    )
    assert a.dxy_alignment == "SELL"


def test_audusd_low_vix_is_buy_leaning():
    a = compute_macro_alignment(
        symbol="AUDUSD=X",
        macro_context=_macro_dict(dxy_bucket="FLAT", us10y_change_bp=0, vix=15.0),
    )
    # AUD risk-on currency should lean BUY when VIX is low.
    assert a.vix_regime == "LOW"
    # vix_alignment for AUD on low VIX is BUY per spec; check via score
    # (positive score-direction).
    assert a.macro_score >= 0


def test_audusd_high_vix_is_sell_leaning():
    a = compute_macro_alignment(
        symbol="AUDUSD=X",
        macro_context=_macro_dict(dxy_bucket="FLAT", us10y_change_bp=0, vix=27.0),
    )
    assert a.vix_regime == "HIGH"
    assert a.macro_score < 0


def test_usdjpy_high_vix_is_sell_leaning_via_safe_haven():
    """High VIX → JPY safe-haven bid → USDJPY SELL bias even if DXY up."""
    a = compute_macro_alignment(
        symbol="USDJPY=X",
        macro_context=_macro_dict(dxy_bucket="UP", us10y_change_bp=0, vix=27.0),
    )
    # vix_align is SELL for USDJPY at HIGH VIX → should pull score down
    # vs. the same setup at low VIX.
    a_low = compute_macro_alignment(
        symbol="USDJPY=X",
        macro_context=_macro_dict(dxy_bucket="UP", us10y_change_bp=0, vix=15.0),
    )
    assert a.macro_score < a_low.macro_score


def test_very_high_vix_blocks_against_buy_for_safe_haven_sells():
    a = compute_macro_alignment(
        symbol="USDJPY=X",
        macro_context=_macro_dict(dxy_bucket="UP", us10y_change_bp=0, vix=35.0),
    )
    assert a.vix_regime == "VERY_HIGH"
    # vix_align==SELL → macro_strong_against == "BUY"
    assert a.macro_strong_against == "BUY"
    assert "vix_very_high_blocks_buy" in a.macro_block_reasons


def test_aggregate_score_clipped_to_minus_one_to_plus_one():
    a = compute_macro_alignment(
        symbol="USDJPY=X",
        macro_context=_macro_dict(
            dxy_bucket="STRONG_UP", us10y_change_bp=20.0, vix=15.0,
        ),
    )
    assert -1.0 <= a.macro_score <= 1.0


def test_to_dict_emits_required_keys():
    a = compute_macro_alignment(
        symbol="EURUSD=X",
        macro_context=_macro_dict(dxy_bucket="DOWN", us10y_change_bp=-10.0, vix=20.0),
    )
    d = a.to_dict()
    for key in (
        "schema_version", "symbol", "dxy_alignment", "yield_alignment",
        "vix_regime", "risk_on_off", "currency_bias", "macro_score",
        "macro_reasons", "macro_block_reasons", "macro_strong_against",
        "unavailable_reasons",
    ):
        assert key in d


def test_unavailable_reasons_recorded_when_partial():
    a = compute_macro_alignment(
        symbol="EURUSD=X",
        macro_context=_macro_dict(dxy_bucket=None, us10y_change_bp=None, vix=None),
    )
    assert "dxy_unavailable" in a.unavailable_reasons
    assert "us10y_unavailable" in a.unavailable_reasons
    assert "vix_unavailable" in a.unavailable_reasons
