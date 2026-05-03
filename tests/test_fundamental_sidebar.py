"""Tests for fundamental_sidebar_v1.

Covers required cases from Phase G spec:
  - FOMC ±6h → event_risk_status=BLOCK
  - BOJ ±6h → BLOCK
  - CPI ±2h → BLOCK; ±2h to ±4h → WARNING
  - NFP ±2h → BLOCK
  - DXY / yields / VIX missing → missing_data has them
  - events feed missing → status=UNKNOWN, now_trade_allowed=False
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fx.calendar import Event
from src.fx.fundamental_sidebar import (
    BLOCK,
    CLEAR,
    SCHEMA_VERSION,
    UNKNOWN,
    WARNING,
    build_fundamental_sidebar,
)


NOW = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)


def _fomc(when: datetime, impact: str = "high") -> Event:
    return Event(
        when=when, currency="USD", title="FOMC Statement",
        impact=impact, kind="FOMC",
    )


def _boj(when: datetime, impact: str = "high") -> Event:
    return Event(
        when=when, currency="JPY", title="BOJ Rate Decision",
        impact=impact, kind="BOJ",
    )


def _cpi(when: datetime, impact: str = "high") -> Event:
    return Event(
        when=when, currency="USD", title="CPI",
        impact=impact, kind="CPI",
    )


def _nfp(when: datetime, impact: str = "high") -> Event:
    return Event(
        when=when, currency="USD", title="Non-Farm Payrolls",
        impact=impact, kind="NFP",
    )


# ── Schema ───────────────────────────────────────────────────────
def test_schema_version_is_v1():
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[],
        macro_alignment=None, macro_context=None,
    )
    assert out["schema_version"] == SCHEMA_VERSION == "fundamental_sidebar_v1"


# ── FOMC ±6h ─────────────────────────────────────────────────────
@pytest.mark.parametrize("hours_offset", [-6.0, -3.0, -0.5, 0.0, 0.5, 3.0, 5.99])
def test_fomc_inside_6h_window_blocks(hours_offset: float):
    ev = _fomc(NOW + timedelta(hours=hours_offset))
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[ev],
        macro_alignment={"macro_score": 0.0}, macro_context={"vix": 18.0},
    )
    assert out["event_risk_status"] == BLOCK, hours_offset
    assert out["now_trade_allowed"] is False
    assert len(out["blocking_events"]) == 1


def test_fomc_outside_6h_inside_warning_zone():
    # Between 6h and 12h (2x window) → WARNING
    ev = _fomc(NOW + timedelta(hours=8))
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[ev],
        macro_alignment={"macro_score": 0.0}, macro_context={"vix": 18.0},
    )
    assert out["event_risk_status"] == WARNING


def test_fomc_far_future_clear():
    ev = _fomc(NOW + timedelta(hours=24))
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[ev],
        macro_alignment={"macro_score": 0.0}, macro_context={"vix": 18.0},
    )
    assert out["event_risk_status"] == CLEAR


# ── BOJ ±6h ──────────────────────────────────────────────────────
def test_boj_inside_6h_blocks_usdjpy():
    ev = _boj(NOW + timedelta(hours=4))
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[ev],
        macro_alignment={"macro_score": 0.0}, macro_context={"vix": 18.0},
    )
    assert out["event_risk_status"] == BLOCK


def test_boj_does_not_affect_eurusd():
    """JPY-only event must not surface for EURUSD=X."""
    ev = _boj(NOW + timedelta(hours=4))
    out = build_fundamental_sidebar(
        symbol="EURUSD=X", now=NOW, events=[ev],
        macro_alignment={"macro_score": 0.0}, macro_context={"vix": 18.0},
    )
    assert out["event_risk_status"] == CLEAR


# ── CPI ±2h ──────────────────────────────────────────────────────
def test_cpi_inside_2h_blocks():
    ev = _cpi(NOW + timedelta(hours=1))
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[ev],
        macro_alignment={"macro_score": 0.0}, macro_context={"vix": 18.0},
    )
    assert out["event_risk_status"] == BLOCK


def test_cpi_between_2h_and_4h_warns():
    # 3h is inside 2× window (4h) but outside 1× (2h) → WARNING
    ev = _cpi(NOW + timedelta(hours=3))
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[ev],
        macro_alignment={"macro_score": 0.0}, macro_context={"vix": 18.0},
    )
    assert out["event_risk_status"] == WARNING


# ── NFP ±2h ──────────────────────────────────────────────────────
def test_nfp_inside_2h_blocks():
    ev = _nfp(NOW + timedelta(hours=1))
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[ev],
        macro_alignment={"macro_score": 0.0}, macro_context={"vix": 18.0},
    )
    assert out["event_risk_status"] == BLOCK


# ── Missing data labels ──────────────────────────────────────────
def test_dxy_missing_appears_in_missing_data():
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[],
        macro_alignment=None,
        macro_context={"vix": 18.0},  # has vix, no DXY
    )
    names = [m["name"] for m in out["missing_data"]]
    assert "dxy" in names


def test_us_yields_missing_appears_in_missing_data():
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[],
        macro_alignment=None,
        macro_context={"vix": 18.0},
    )
    names = [m["name"] for m in out["missing_data"]]
    assert "us10y_yield" in names
    assert "us2y_yield" in names


def test_news_calendar_missing_appears_in_missing_data():
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[],
        macro_alignment={"macro_score": 0.0},  # no event_tone
        macro_context={"vix": 18.0},
    )
    names = [m["name"] for m in out["missing_data"]]
    assert "news_calendar" in names


# ── Events feed completely missing → UNKNOWN ─────────────────────
def test_events_none_yields_unknown_no_trade():
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=None,
        macro_alignment=None, macro_context=None,
    )
    assert out["event_risk_status"] == UNKNOWN
    assert out["now_trade_allowed"] is False
    assert "event_calendar" in [m["name"] for m in out["missing_data"]]


# ── Macro drivers rendering ──────────────────────────────────────
def test_macro_drivers_include_dxy_yield_vix_score():
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[],
        macro_alignment={
            "dxy_alignment": "BUY",
            "yield_alignment": "NEUTRAL",
            "vix_regime": "ELEVATED",
            "currency_bias": "BUY",
            "macro_score": 0.42,
        },
        macro_context={"vix": 22.0},
    )
    names = {d["name"] for d in out["macro_drivers"]}
    for required in ("dxy_alignment", "yield_alignment",
                      "vix_regime", "currency_bias", "macro_score"):
        assert required in names


# ── Reason text contains event title + minutes ───────────────────
def test_block_reason_contains_event_title_and_window():
    ev = _fomc(NOW + timedelta(hours=2))
    out = build_fundamental_sidebar(
        symbol="USDJPY=X", now=NOW, events=[ev],
        macro_alignment={"macro_score": 0.0}, macro_context={"vix": 18.0},
    )
    assert "FOMC Statement" in out["reason_ja"]
    assert "新規エントリー禁止" in out["reason_ja"]
    assert "6h" in out["reason_ja"]
