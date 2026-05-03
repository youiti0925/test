"""Tests for R12-R16 technical_confluence_v1 hypothesis cells (PR-C2).

Each R-test feeds `compute_r_candidates_summary` a hand-built list of
trade dicts whose `entry_trace["technical_confluence"]` payload is
crafted to land in a specific cell. Cells must:
  - exist in r_candidates_v1 with the documented key,
  - count rows correctly,
  - report `sufficient_n` against the existing N_THRESHOLD_FOR_SUFFICIENT (≥30),
  - omit trades whose entry_trace lacks `technical_confluence`.
"""
from __future__ import annotations

from typing import Any

import pytest

from src.fx.decision_trace_r_candidates import (
    N_THRESHOLD_FOR_SUFFICIENT,
    compute_r_candidates_summary,
)


# ───────────────────────── trade builder ─────────────────────────────────


def _trade(
    *,
    return_pct: float = 0.5,
    side: str = "BUY",
    label: str = "WEAK_BUY_SETUP",
    near_support: bool = False,
    near_resistance: bool = False,
    candle: dict | None = None,
    indicator: dict | None = None,
    risk: dict | None = None,
    has_confluence: bool = True,
    sym: str = "EURUSD=X",
) -> dict:
    confluence = None
    if has_confluence:
        confluence = {
            "policy_version": "technical_confluence_v1",
            "market_regime": "RANGE",
            "dow_structure": {},
            "support_resistance": {
                "near_support": near_support,
                "near_resistance": near_resistance,
            },
            "candlestick_signal": candle or {},
            "chart_pattern": {},
            "indicator_context": indicator or {},
            "risk_plan_obs": risk or {},
            "vote_breakdown": {},
            "final_confluence": {
                "label": label, "score": 0.0,
                "bullish_reasons": [], "bearish_reasons": [], "avoid_reasons": [],
            },
        }
    return {
        "sym": sym, "side": side,
        "entry_ts": "2025-06-01T12:00:00+00:00",
        "exit_ts":  "2025-06-01T18:00:00+00:00",
        "exit_reason": "tp", "return_pct": return_pct,
        "bars_held": 6,
        "entry_trace": {
            "technical_confluence": confluence,
        } if has_confluence else {"technical_confluence": None},
    }


def _summary(trades: list[dict]) -> dict:
    return compute_r_candidates_summary(traces=[], trades=trades)


# ───────────────────────── presence + sufficient_n ───────────────────────


def test_r12_to_r16_present_in_summary():
    s = _summary([])
    for k in (
        "r12_confluence_label_outcome",
        "r13_sr_proximity_outcome",
        "r14_candlestick_confirmation_outcome",
        "r15_rsi_regime_danger_outcome",
        "r16_structure_stop_distance_outcome",
    ):
        assert k in s, f"{k} must be present in r_candidates_summary"
        assert "cells" in s[k]
        assert "hypothesis" in s[k]


def test_sufficient_n_uses_existing_threshold():
    """Sanity-pin N_THRESHOLD_FOR_SUFFICIENT at 30. R12 cells with
    n<30 must report sufficient_n=False; n>=30 must report True."""
    assert N_THRESHOLD_FOR_SUFFICIENT == 30
    trades_low_n = [_trade(label="STRONG_BUY_SETUP") for _ in range(29)]
    trades_high_n = [_trade(label="STRONG_BUY_SETUP") for _ in range(35)]
    s_low = _summary(trades_low_n)
    s_high = _summary(trades_high_n)
    cell_low = next(
        c for c in s_low["r12_confluence_label_outcome"]["cells"]
        if c["label"] == "label=STRONG_BUY_SETUP"
    )
    cell_high = next(
        c for c in s_high["r12_confluence_label_outcome"]["cells"]
        if c["label"] == "label=STRONG_BUY_SETUP"
    )
    assert cell_low["n"] == 29 and cell_low["sufficient_n"] is False
    assert cell_high["n"] == 35 and cell_high["sufficient_n"] is True


# ───────────────────────── R12: confluence_label outcome ─────────────────


def test_r12_buckets_trades_by_label():
    trades = (
        [_trade(label="STRONG_BUY_SETUP", return_pct=1.0) for _ in range(3)]
        + [_trade(label="STRONG_BUY_SETUP", return_pct=-0.5) for _ in range(2)]
        + [_trade(label="WEAK_BUY_SETUP",   return_pct=0.5) for _ in range(4)]
        + [_trade(label="AVOID_TRADE",       return_pct=-1.0) for _ in range(1)]
    )
    cells = _summary(trades)["r12_confluence_label_outcome"]["cells"]
    by_label = {c["label"]: c for c in cells}
    assert by_label["label=STRONG_BUY_SETUP"]["n"] == 5
    assert by_label["label=STRONG_BUY_SETUP"]["wins"] == 3
    assert by_label["label=STRONG_BUY_SETUP"]["losses"] == 2
    assert by_label["label=WEAK_BUY_SETUP"]["n"] == 4
    assert by_label["label=AVOID_TRADE"]["n"] == 1
    # Untouched labels still present with n=0
    assert by_label["label=NO_TRADE"]["n"] == 0
    assert by_label["label=UNKNOWN"]["n"] == 0


def test_r12_label_set_is_closed():
    """All seven labels must appear as cells, even if n=0."""
    cells = _summary([])["r12_confluence_label_outcome"]["cells"]
    labels = {c["label"] for c in cells}
    assert labels == {
        "label=STRONG_BUY_SETUP", "label=WEAK_BUY_SETUP",
        "label=STRONG_SELL_SETUP", "label=WEAK_SELL_SETUP",
        "label=NO_TRADE", "label=AVOID_TRADE", "label=UNKNOWN",
    }


# ───────────────────────── R13: SR proximity outcome ─────────────────────


def test_r13_partitions_by_proximity():
    trades = (
        [_trade(near_support=True) for _ in range(3)]
        + [_trade(near_resistance=True) for _ in range(2)]
        + [_trade() for _ in range(4)]
    )
    cells = {c["label"]: c for c in _summary(trades)["r13_sr_proximity_outcome"]["cells"]}
    assert cells["near_support"]["n"] == 3
    assert cells["near_resistance"]["n"] == 2
    assert cells["not_near_any_level"]["n"] == 4


# ───────────────────────── R14: candlestick confirmation ─────────────────


def test_r14_bullish_and_bearish_partition():
    trades = (
        [_trade(candle={"bullish_pinbar": True}) for _ in range(2)]
        + [_trade(candle={"bullish_engulfing": True}) for _ in range(2)]
        + [_trade(candle={"strong_bull_body": True}) for _ in range(1)]
        + [_trade(candle={"bearish_pinbar": True}) for _ in range(3)]
        + [_trade(candle={}) for _ in range(4)]  # no confirmation
    )
    cells = {c["label"]: c for c in _summary(trades)["r14_candlestick_confirmation_outcome"]["cells"]}
    assert cells["bullish_candle_confirmation"]["n"] == 5
    assert cells["bearish_candle_confirmation"]["n"] == 3
    assert cells["any_candlestick_confirmation"]["n"] == 8
    assert cells["no_candlestick_confirmation"]["n"] == 4


def test_r14_a_trade_with_both_bullish_and_bearish_lands_in_both():
    """A bar with both flags shows up in both directional cells AND in
    `any` (this is the conservative observation-only choice — analysts
    can intersect later)."""
    trades = [
        _trade(candle={"bullish_pinbar": True, "bearish_pinbar": True}),
    ]
    cells = {c["label"]: c for c in _summary(trades)["r14_candlestick_confirmation_outcome"]["cells"]}
    assert cells["bullish_candle_confirmation"]["n"] == 1
    assert cells["bearish_candle_confirmation"]["n"] == 1
    assert cells["any_candlestick_confirmation"]["n"] == 1
    assert cells["no_candlestick_confirmation"]["n"] == 0


# ───────────────────────── R15: RSI regime danger ────────────────────────


def test_r15_pivots_by_rsi_regime_flags():
    trades = (
        [_trade(indicator={"rsi_trend_danger": True}) for _ in range(2)]
        + [_trade(indicator={"rsi_trend_danger": False}) for _ in range(3)]
        + [_trade(indicator={"rsi_range_valid": True}) for _ in range(4)]
    )
    cells = {c["label"]: c for c in _summary(trades)["r15_rsi_regime_danger_outcome"]["cells"]}
    assert cells["rsi_trend_danger_true"]["n"] == 2
    assert cells["rsi_trend_danger_false"]["n"] == 3
    assert cells["rsi_range_valid_true"]["n"] == 4


# ───────────────────────── R16: structure stop distance ──────────────────


def test_r16_buckets_structure_stop_distance():
    trades = (
        [_trade(risk={"structure_stop_distance_atr": 0.2}) for _ in range(2)]
        + [_trade(risk={"structure_stop_distance_atr": 1.0}) for _ in range(3)]
        + [_trade(risk={"structure_stop_distance_atr": 2.5}) for _ in range(1)]
        + [_trade(risk={"structure_stop_distance_atr": 5.0}) for _ in range(2)]
        + [_trade(risk={"structure_stop_distance_atr": None}) for _ in range(4)]
    )
    cells = {c["label"]: c for c in _summary(trades)["r16_structure_stop_distance_outcome"]["cells"]}
    assert cells["structure_stop_lt_0_5_atr"]["n"] == 2
    assert cells["structure_stop_0_5_to_1_5_atr"]["n"] == 3
    assert cells["structure_stop_1_5_to_3_0_atr"]["n"] == 1
    assert cells["structure_stop_gt_3_0_atr"]["n"] == 2
    assert cells["structure_stop_missing"]["n"] == 4


def test_r16_bucket_order_is_canonical():
    """Bucket cells must appear in the canonical order regardless of
    insertion / counter dict ordering."""
    s = _summary([])
    labels = [c["label"] for c in s["r16_structure_stop_distance_outcome"]["cells"]]
    assert labels == [
        "structure_stop_missing",
        "structure_stop_lt_0_5_atr",
        "structure_stop_0_5_to_1_5_atr",
        "structure_stop_1_5_to_3_0_atr",
        "structure_stop_gt_3_0_atr",
    ]


# ───────────────────────── trades w/o confluence ─────────────────────────


def test_trades_without_confluence_excluded():
    """Trades whose entry_trace has no technical_confluence must not
    crash the R-cells. They contribute `0` to the labelled cells and to
    `not_near_any_level` / `no_candlestick_confirmation` /
    `structure_stop_missing` (the 'missing-data' fallback cells)."""
    trades = [
        _trade(label="STRONG_BUY_SETUP", has_confluence=False),
        _trade(label="WEAK_BUY_SETUP", has_confluence=True),
    ]
    s = _summary(trades)

    # R12 — only the WEAK_BUY_SETUP one counts; the no-confluence one
    # falls into UNKNOWN (because _tc_label returns "UNKNOWN").
    cells = {c["label"]: c for c in s["r12_confluence_label_outcome"]["cells"]}
    assert cells["label=STRONG_BUY_SETUP"]["n"] == 0
    assert cells["label=WEAK_BUY_SETUP"]["n"] == 1
    assert cells["label=UNKNOWN"]["n"] == 1

    # R13 — both fall into not_near_any_level (neither sets near_*)
    cells = {c["label"]: c for c in s["r13_sr_proximity_outcome"]["cells"]}
    assert cells["not_near_any_level"]["n"] == 2

    # R16 — both lack structure_stop_distance_atr → structure_stop_missing
    cells = {c["label"]: c for c in s["r16_structure_stop_distance_outcome"]["cells"]}
    assert cells["structure_stop_missing"]["n"] == 2


# ───────────────────────── observation-only invariant ────────────────────


def test_r12_through_r16_do_not_reference_decision_engine():
    """The R-helpers must remain pure — they read trade rows + traces
    and never import decide_action / risk_gate."""
    src = open("src/fx/decision_trace_r_candidates.py").read()
    assert "from .decision_engine import" not in src
    assert "from .risk_gate import" not in src
