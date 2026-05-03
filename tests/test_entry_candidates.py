"""Tests for the Phase I-1/I-2 entry candidate foundation.

These pin the schema, common-gate behaviour, selector behaviour, and
the entry_plan_v1 → EntryCandidate compatibility bridge.

Phase I must not change the existing entry_plan_v1 semantics or the
final action — these tests assert that the candidate layer is purely
additive.
"""
from __future__ import annotations

from dataclasses import replace

from src.fx.entry_candidates import (
    ENTRY_CANDIDATE_SCHEMA_VERSION,
    EntryCandidate,
    EntryMethodContext,
    apply_common_entry_gates,
    build_entry_candidates_from_existing_plan,
    candidate_from_entry_plan,
    select_best_entry_candidate,
    selected_candidate_to_entry_plan,
)


def _ctx(**overrides) -> EntryMethodContext:
    base = dict(
        symbol="EURUSD=X",
        timeframe="1h",
        df_window=None,
        atr_value=0.001,
        last_close=1.1000,
        current_ts=None,
        pattern_levels={},
        wave_derived_lines=[],
        breakout_quality_gate={},
        fundamental_sidebar={},
        support_resistance_v2={},
        trendline_context={},
        dow_structure_review={},
        ma_context_review={},
        candlestick_anatomy_review={},
        entry_settings={"min_rr_default": 2.0},
    )
    base.update(overrides)
    return EntryMethodContext(**base)


def _ready(side: str = "BUY", rr: float = 2.2, score: float = 80.0) -> EntryCandidate:
    if side == "BUY":
        entry, stop, target = 1.1000, 1.0950, 1.1110
    else:
        entry, stop, target = 1.1000, 1.1050, 1.0890

    return EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type="neckline_retest",
        status="READY",
        side=side,  # type: ignore[arg-type]
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        rr=rr,
        final_score=score,
    )


# ─────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────

def test_entry_candidate_to_dict_has_required_fields():
    c = _ready()
    d = c.to_dict()
    assert d["schema_version"] == ENTRY_CANDIDATE_SCHEMA_VERSION
    assert d["entry_type"] == "neckline_retest"
    assert d["status"] == "READY"
    assert d["side"] == "BUY"
    assert "final_score" in d
    assert "block_reasons" in d
    assert "cautions" in d
    assert "reasons_ja" in d
    assert "debug" in d


def test_schema_version_pin():
    """Locks the candidate schema name. Bumping this is a breaking
    change for downstream payload consumers + visual_audit."""
    assert ENTRY_CANDIDATE_SCHEMA_VERSION == "entry_candidate_v1"


# ─────────────────────────────────────────────────────────────────
# Common gates
# ─────────────────────────────────────────────────────────────────

def test_common_gate_event_block_downgrades_ready_to_wait_event_clear():
    c = _ready()
    ctx = _ctx(fundamental_sidebar={"event_risk_status": "BLOCK"})
    out = apply_common_entry_gates(c, ctx)
    assert out.status == "WAIT_EVENT_CLEAR"
    assert "event_risk_block" in out.block_reasons


def test_common_gate_does_not_touch_non_ready_candidates():
    c = replace(_ready(), status="WAIT_RETEST")
    ctx = _ctx(fundamental_sidebar={"event_risk_status": "BLOCK"})
    out = apply_common_entry_gates(c, ctx)
    # WAIT_RETEST is left alone — only READY is touched by the gate.
    assert out.status == "WAIT_RETEST"
    assert "event_risk_block" not in out.block_reasons


def test_common_gate_missing_entry_stop_target_holds():
    c = EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type="neckline_retest",
        status="READY",
        side="BUY",
        entry_price=None,
        stop_price=1.0,
        target_price=1.2,
        rr=2.0,
    )
    out = apply_common_entry_gates(c, _ctx())
    assert out.status == "HOLD"
    assert "entry_stop_target_missing" in out.block_reasons


def test_common_gate_rr_missing_holds():
    c = replace(_ready(), rr=None)
    out = apply_common_entry_gates(c, _ctx())
    assert out.status == "HOLD"
    assert "rr_missing" in out.block_reasons


def test_common_gate_rr_too_low_holds():
    c = _ready(rr=1.2)
    out = apply_common_entry_gates(c, _ctx())
    assert out.status == "HOLD"
    assert any(x.startswith("rr_too_low") for x in out.block_reasons)


def test_common_gate_invalid_buy_price_order_holds():
    c = EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type="neckline_retest",
        status="READY",
        side="BUY",
        entry_price=1.1000,
        stop_price=1.1050,  # stop above entry — invalid for BUY
        target_price=1.1110,
        rr=2.0,
    )
    out = apply_common_entry_gates(c, _ctx())
    assert out.status == "HOLD"
    assert "invalid_price_order_buy" in out.block_reasons


def test_common_gate_invalid_sell_price_order_holds():
    c = EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type="neckline_retest",
        status="READY",
        side="SELL",
        entry_price=1.1000,
        stop_price=1.0950,  # stop below entry — invalid for SELL
        target_price=1.0890,
        rr=2.0,
    )
    out = apply_common_entry_gates(c, _ctx())
    assert out.status == "HOLD"
    assert "invalid_price_order_sell" in out.block_reasons


def test_common_gate_bad_stop_quality_holds():
    c = replace(_ready(), stop_quality="BAD")
    out = apply_common_entry_gates(c, _ctx())
    assert out.status == "HOLD"
    assert "bad_stop_quality" in out.block_reasons


def test_common_gate_target_blocked_holds():
    c = replace(_ready(), target_clearance="BLOCKED")
    out = apply_common_entry_gates(c, _ctx())
    assert out.status == "HOLD"
    assert "target_blocked_by_near_sr" in out.block_reasons


def test_common_gate_breakout_quality_block_holds():
    c = replace(_ready(), breakout_quality="BLOCK")
    out = apply_common_entry_gates(c, _ctx())
    assert out.status == "HOLD"
    assert "breakout_quality_block" in out.block_reasons


def test_common_gate_clean_ready_passes():
    c = _ready()
    out = apply_common_entry_gates(c, _ctx())
    assert out.status == "READY"
    assert out.block_reasons == []


# ─────────────────────────────────────────────────────────────────
# Selector
# ─────────────────────────────────────────────────────────────────

def test_selector_ready_uses_final_score_not_type_priority():
    low = _ready(score=70.0)
    high = EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type="future_method_lower_priority_name",
        status="READY",
        side="BUY",
        entry_price=1.1000,
        stop_price=1.0950,
        target_price=1.1110,
        rr=2.2,
        final_score=95.0,
    )
    out = select_best_entry_candidate([low, high])
    assert out.entry_type == "future_method_lower_priority_name"
    assert out.final_score == 95.0


def test_selector_wait_order_when_no_ready():
    hold = EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type="a",
        status="HOLD",
        side="NEUTRAL",
        entry_price=None,
        stop_price=None,
        target_price=None,
        rr=None,
        final_score=99.0,
    )
    wait_retest = EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type="b",
        status="WAIT_RETEST",
        side="BUY",
        entry_price=None,
        stop_price=1.0,
        target_price=1.2,
        rr=2.0,
        final_score=10.0,
    )
    out = select_best_entry_candidate([hold, wait_retest])
    # Even with much lower final_score, WAIT_RETEST outranks HOLD when
    # no READY candidate exists.
    assert out.status == "WAIT_RETEST"


def test_selector_empty_list_returns_none_placeholder():
    out = select_best_entry_candidate([])
    assert out.entry_type == "none"
    assert out.status == "HOLD"
    assert out.used_in_decision is False
    assert "no_entry_candidates" in out.block_reasons


def test_selector_skips_neutral_ready_candidates():
    """A READY+NEUTRAL candidate should not be selected over a
    well-formed WAIT_RETEST BUY/SELL — NEUTRAL has no actionable
    direction."""
    neutral_ready = EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type="neutral",
        status="READY",
        side="NEUTRAL",
        entry_price=1.1,
        stop_price=1.0,
        target_price=1.2,
        rr=2.0,
        final_score=99.0,
    )
    wait_retest = EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type="neckline_retest",
        status="WAIT_RETEST",
        side="BUY",
        entry_price=None,
        stop_price=1.0,
        target_price=1.2,
        rr=2.0,
        final_score=10.0,
    )
    out = select_best_entry_candidate([neutral_ready, wait_retest])
    assert out.status == "WAIT_RETEST"
    assert out.side == "BUY"


# ─────────────────────────────────────────────────────────────────
# entry_plan_v1 → candidate bridge
# ─────────────────────────────────────────────────────────────────

def test_candidate_from_existing_entry_plan_ready():
    ep = {
        "entry_status": "READY",
        "side": "BUY",
        "entry_price": 1.1000,
        "stop_price": 1.0950,
        "target_extended_price": 1.1110,
        "rr": 2.2,
        "trigger_line_id": "WNL1",
        "trigger_line_price": 1.0990,
        "breakout_confirmed": True,
        "retest_confirmed": True,
        "confirmation_candle": "bullish_pinbar",
        "reason_ja": "既存READY",
        "block_reasons": [],
    }
    c = candidate_from_entry_plan(ep)
    assert c.entry_type == "neckline_retest"
    assert c.status == "READY"
    assert c.side == "BUY"
    assert c.trigger_line_id == "WNL1"
    assert c.trigger_line_price == 1.0990
    assert c.trigger_confirmed is True
    assert c.retest_confirmed is True
    assert c.confirmation_candle == "bullish_pinbar"
    assert c.rr == 2.2
    assert c.target_price == 1.1110  # target_extended_price preferred


def test_candidate_from_entry_plan_falls_back_to_target_price():
    ep = {
        "entry_status": "READY",
        "side": "BUY",
        "entry_price": 1.1,
        "stop_price": 1.0,
        "target_price": 1.2,
        "rr": 2.0,
    }
    c = candidate_from_entry_plan(ep)
    assert c.target_price == 1.2


def test_candidate_from_unknown_status_falls_back_to_hold():
    ep = {
        "entry_status": "TOTALLY_UNKNOWN",
        "side": "BUY",
        "entry_price": 1.1,
        "stop_price": 1.0,
        "target_price": 1.2,
        "rr": 2.0,
    }
    c = candidate_from_entry_plan(ep)
    assert c.status == "HOLD"


def test_candidate_from_none_returns_hold():
    c = candidate_from_entry_plan(None)
    assert c.status == "HOLD"
    assert c.side == "NEUTRAL"


def test_selected_candidate_matches_existing_entry_plan_core_fields():
    ep = {
        "entry_status": "WAIT_RETEST",
        "entry_type": "retest",
        "side": "BUY",
        "entry_price": None,
        "stop_price": 1.095,
        "target_price": 1.110,
        "target_extended_price": 1.115,
        "rr": 2.5,
        "trigger_line_id": "WNL1",
        "trigger_line_price": 1.100,
        "breakout_confirmed": True,
        "retest_confirmed": False,
        "block_reasons": ["awaiting_retest_confirmation"],
    }
    c = candidate_from_entry_plan(ep)
    assert c.entry_type == "neckline_retest"
    assert c.status == ep["entry_status"]
    assert c.side == ep["side"]
    assert c.stop_price == ep["stop_price"]
    assert c.rr == ep["rr"]
    # trigger metadata is preserved
    assert c.trigger_line_id == "WNL1"
    assert c.trigger_line_price == 1.100
    # bridge is observation-only — block_reasons carry through
    assert "awaiting_retest_confirmation" in c.block_reasons


# ─────────────────────────────────────────────────────────────────
# build_entry_candidates_from_existing_plan
# ─────────────────────────────────────────────────────────────────

def test_builder_returns_single_neckline_retest_candidate():
    ep = {
        "entry_status": "READY",
        "side": "BUY",
        "entry_price": 1.1,
        "stop_price": 1.0,
        "target_price": 1.2,
        "rr": 2.5,
    }
    cands = build_entry_candidates_from_existing_plan(
        entry_plan=ep, ctx=_ctx(),
    )
    assert len(cands) == 1
    assert cands[0].entry_type == "neckline_retest"
    assert cands[0].status == "READY"


def test_builder_applies_common_gate_event_block():
    ep = {
        "entry_status": "READY",
        "side": "BUY",
        "entry_price": 1.1,
        "stop_price": 1.0,
        "target_price": 1.2,
        "rr": 2.5,
    }
    cands = build_entry_candidates_from_existing_plan(
        entry_plan=ep,
        ctx=_ctx(fundamental_sidebar={"event_risk_status": "BLOCK"}),
    )
    assert len(cands) == 1
    assert cands[0].status == "WAIT_EVENT_CLEAR"


# ─────────────────────────────────────────────────────────────────
# selected_candidate_to_entry_plan — additive bridge
# ─────────────────────────────────────────────────────────────────

def test_selected_candidate_to_entry_plan_preserves_existing_core_fields():
    """Phase I must not overwrite the original entry_plan core fields.
    The bridge is additive — only adds selected_entry_candidate_type
    / selected_entry_candidate_score (and fills missing fields)."""
    original = {
        "entry_status": "WAIT_RETEST",
        "side": "BUY",
        "entry_price": 1.1,
        "stop_price": 1.0,
        "target_price": 1.2,
        "rr": 2.0,
        "reason_ja": "preserved-original-reason",
    }
    selected = candidate_from_entry_plan(original)
    out = selected_candidate_to_entry_plan(
        selected, original_entry_plan=original,
    )
    # Originals untouched.
    assert out["entry_status"] == "WAIT_RETEST"
    assert out["side"] == "BUY"
    assert out["entry_price"] == 1.1
    assert out["stop_price"] == 1.0
    assert out["target_price"] == 1.2
    assert out["rr"] == 2.0
    assert out["reason_ja"] == "preserved-original-reason"
    # Selector metadata added.
    assert out["selected_entry_candidate_type"] == "neckline_retest"
    assert "selected_entry_candidate_score" in out


def test_selected_candidate_to_entry_plan_fills_when_original_is_empty():
    """If the original entry_plan is empty/None, the bridge fills in
    the selected candidate's fields so downstream payload consumers
    still see a well-formed entry_plan."""
    selected = _ready()
    out = selected_candidate_to_entry_plan(
        selected, original_entry_plan=None,
    )
    assert out["entry_status"] == "READY"
    assert out["side"] == "BUY"
    assert out["entry_price"] == 1.1000
    assert out["stop_price"] == 1.0950
    assert out["target_price"] == 1.1110
    assert out["selected_entry_candidate_type"] == "neckline_retest"
