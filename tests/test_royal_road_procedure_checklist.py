"""Tests for the royal-road procedure checklist (Phase I follow-up).

Pin the 14-step canonical order, the per-step status mapping, and
the aggregate p0_pass / wait_reasons / block_reasons output.
"""
from __future__ import annotations

from src.fx.royal_road_procedure import (
    PROCEDURE_SCHEMA_VERSION,
    RoyalRoadProcedureChecklist,
    RoyalRoadProcedureStep,
    build_royal_road_procedure_checklist,
)


_ALL_STEP_KEYS = (
    "environment",
    "dow_structure",
    "support_resistance",
    "trendline_context",
    "wave_pattern",
    "wave_lines",
    "breakout_confirmed",
    "retest_confirmed",
    "confirmation_candle",
    "entry_price",
    "stop_price",
    "target_price",
    "rr_ok",
    "event_clear",
)


def _ready_inputs(*, side: str = "SELL") -> dict:
    """A WAIT/READY-shaped happy-path input set."""
    if side == "BUY":
        entry, stop, target = 1.1000, 1.0950, 1.1110
    else:
        entry, stop, target = 1.1000, 1.1050, 1.0890

    entry_plan = {
        "entry_status": "READY",
        "side": side,
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "target_extended_price": target,
        "rr": 2.2,
        "trigger_line_id": "WNL1",
        "trigger_line_price": entry,
        "breakout_confirmed": True,
        "retest_confirmed": True,
        "confirmation_candle":
            "bearish_pinbar" if side == "SELL" else "bullish_pinbar",
        "block_reasons": [],
    }
    pattern_levels = {
        "available": True,
        "pattern_kind": "double_top" if side == "SELL" else "double_bottom",
        "parts": {"P1": {"price": 1.11}, "P2": {"price": 1.11}},
    }
    wave_derived_lines = [
        {"role": "entry_confirmation_line", "id": "WNL1", "price": entry},
        {"role": "stop_candidate", "id": "WSL1", "price": stop},
        {"role": "target_candidate", "id": "WTP1", "price": target},
    ]
    sr_v2 = {
        "selected_level_zones_top5": [
            {"kind": "support", "zone_low": 1.09, "zone_high": 1.092},
        ],
        "rejected_level_zones": [],
    }
    tl_ctx = {
        "selected_trendlines_top3": [
            {"slope": 0.0001, "intercept": 1.0,
             "anchor_indices": [0, 1, 2]},
            {"slope": -0.0001, "intercept": 1.2,
             "anchor_indices": [0, 1, 2]},
        ],
        "rejected_trendlines": [],
    }
    return {
        "entry_plan": entry_plan,
        "pattern_levels": pattern_levels,
        "wave_derived_lines": wave_derived_lines,
        "breakout_quality_gate": {},
        "fundamental_sidebar": {"event_risk_status": "CLEAR"},
        "support_resistance_v2": sr_v2,
        "trendline_context": tl_ctx,
        "dow_structure_review": {
            "trend": "DOWN" if side == "SELL" else "UP",
        },
        "candlestick_anatomy_review": {},
        "min_rr": 2.0,
    }


# ─────────────────────────────────────────────────────────────────
# Schema + ordering
# ─────────────────────────────────────────────────────────────────

def test_checklist_schema_version_pin():
    assert PROCEDURE_SCHEMA_VERSION == "royal_road_procedure_checklist_v1"


def test_checklist_returns_all_14_steps_in_canonical_order():
    cl = build_royal_road_procedure_checklist(**_ready_inputs())
    assert isinstance(cl, RoyalRoadProcedureChecklist)
    assert len(cl.steps) == 14
    keys_in_order = tuple(s.key for s in cl.steps)
    assert keys_in_order == _ALL_STEP_KEYS


def test_checklist_to_dict_round_trip_includes_steps_and_aggregates():
    cl = build_royal_road_procedure_checklist(**_ready_inputs())
    d = cl.to_dict()
    assert d["schema_version"] == PROCEDURE_SCHEMA_VERSION
    assert d["final_status"] == "READY"
    assert len(d["steps"]) == 14
    assert "p0_pass" in d
    assert "p0_missing_or_blocked" in d
    assert "wait_reasons" in d
    assert "block_reasons" in d


# ─────────────────────────────────────────────────────────────────
# Happy path — READY
# ─────────────────────────────────────────────────────────────────

def test_ready_input_has_p0_pass_true_and_summary_says_ready():
    cl = build_royal_road_procedure_checklist(**_ready_inputs())
    assert cl.p0_pass is True
    assert cl.ready is True
    assert "READY" in cl.summary_ja


def test_ready_input_breakout_retest_confirmation_all_pass():
    cl = build_royal_road_procedure_checklist(**_ready_inputs())
    by_key = {s.key: s for s in cl.steps}
    assert by_key["breakout_confirmed"].status == "PASS"
    assert by_key["retest_confirmed"].status == "PASS"
    assert by_key["confirmation_candle"].status == "PASS"
    assert by_key["entry_price"].status == "PASS"
    assert by_key["stop_price"].status == "PASS"
    assert by_key["target_price"].status == "PASS"
    assert by_key["rr_ok"].status == "PASS"
    assert by_key["event_clear"].status == "PASS"


# ─────────────────────────────────────────────────────────────────
# WAIT_BREAKOUT
# ─────────────────────────────────────────────────────────────────

def test_wait_breakout_shows_wnl_not_broken_on_breakout_step():
    inputs = _ready_inputs()
    inputs["entry_plan"] = {
        **inputs["entry_plan"],
        "entry_status": "WAIT_BREAKOUT",
        "breakout_confirmed": False,
        "retest_confirmed": False,
        "confirmation_candle": "",
        "entry_price": None,
    }
    cl = build_royal_road_procedure_checklist(**inputs)
    by_key = {s.key: s for s in cl.steps}
    assert by_key["breakout_confirmed"].status == "WAIT"
    assert "wnl_not_broken" in by_key["breakout_confirmed"].wait_reasons
    assert "WAIT BREAKOUT" in cl.summary_ja or "ブレイク" in cl.summary_ja
    # P0 must NOT pass when a P0 step is in WAIT.
    assert cl.ready is False


# ─────────────────────────────────────────────────────────────────
# WAIT_RETEST
# ─────────────────────────────────────────────────────────────────

def test_wait_retest_shows_awaiting_retest_on_retest_step():
    inputs = _ready_inputs()
    inputs["entry_plan"] = {
        **inputs["entry_plan"],
        "entry_status": "WAIT_RETEST",
        "breakout_confirmed": True,
        "retest_confirmed": False,
        "confirmation_candle": "",
        "entry_price": None,
    }
    cl = build_royal_road_procedure_checklist(**inputs)
    by_key = {s.key: s for s in cl.steps}
    assert by_key["breakout_confirmed"].status == "PASS"
    assert by_key["retest_confirmed"].status == "WAIT"
    assert (
        "awaiting_retest_confirmation"
        in by_key["retest_confirmed"].wait_reasons
    )
    assert "リターンムーブ" in cl.summary_ja
    assert cl.ready is False


# ─────────────────────────────────────────────────────────────────
# WAIT_EVENT_CLEAR
# ─────────────────────────────────────────────────────────────────

def test_wait_event_clear_keeps_technical_setup_visible():
    """WAIT_EVENT_CLEAR must NOT wipe the wave/entry steps. Only
    environment + event_clear should turn into BLOCK."""
    inputs = _ready_inputs()
    inputs["entry_plan"] = {
        **inputs["entry_plan"],
        "entry_status": "WAIT_EVENT_CLEAR",
        "breakout_confirmed": True,
        "retest_confirmed": True,
        "confirmation_candle": "bearish_pinbar",
    }
    inputs["fundamental_sidebar"] = {
        "event_risk_status": "BLOCK",
        "blocking_events": [
            {"title": "FOMC", "window_hours": 1, "minutes_until": 30},
        ],
    }
    cl = build_royal_road_procedure_checklist(**inputs)
    by_key = {s.key: s for s in cl.steps}
    # event-driven steps go BLOCK
    assert by_key["event_clear"].status == "BLOCK"
    assert "event_risk_block" in by_key["event_clear"].block_reasons
    assert by_key["environment"].status == "BLOCK"
    # technical setup is preserved (NOT wiped)
    assert by_key["wave_pattern"].status == "PASS"
    assert by_key["wave_lines"].status == "PASS"
    assert by_key["breakout_confirmed"].status == "PASS"
    assert by_key["retest_confirmed"].status == "PASS"
    assert by_key["confirmation_candle"].status == "PASS"
    assert by_key["entry_price"].status == "PASS"
    assert by_key["stop_price"].status == "PASS"
    assert by_key["target_price"].status == "PASS"
    assert by_key["rr_ok"].status == "PASS"
    # summary says event-blocked
    assert "イベント" in cl.summary_ja
    assert cl.p0_pass is False
    assert "event_clear" in cl.p0_missing_or_blocked
    assert "event_risk_block" in cl.block_reasons


# ─────────────────────────────────────────────────────────────────
# Wave lines missing → BLOCK on wave_lines
# ─────────────────────────────────────────────────────────────────

def test_missing_wave_derived_lines_block_wave_lines_step():
    inputs = _ready_inputs()
    inputs["wave_derived_lines"] = []  # empty
    cl = build_royal_road_procedure_checklist(**inputs)
    by_key = {s.key: s for s in cl.steps}
    assert by_key["wave_lines"].status == "BLOCK"
    assert any(
        b.startswith("missing_") for b in by_key["wave_lines"].block_reasons
    )
    assert cl.p0_pass is False
    assert "wave_lines" in cl.p0_missing_or_blocked


# ─────────────────────────────────────────────────────────────────
# RR too low → BLOCK on rr_ok
# ─────────────────────────────────────────────────────────────────

def test_rr_too_low_blocks_rr_step():
    inputs = _ready_inputs()
    inputs["entry_plan"] = {**inputs["entry_plan"], "rr": 1.2}
    cl = build_royal_road_procedure_checklist(**inputs)
    by_key = {s.key: s for s in cl.steps}
    assert by_key["rr_ok"].status == "BLOCK"
    assert any(
        x.startswith("rr_too_low") for x in by_key["rr_ok"].block_reasons
    )
    assert cl.p0_pass is False


def test_rr_missing_blocks_rr_step():
    inputs = _ready_inputs()
    inputs["entry_plan"] = {**inputs["entry_plan"], "rr": None}
    cl = build_royal_road_procedure_checklist(**inputs)
    by_key = {s.key: s for s in cl.steps}
    assert by_key["rr_ok"].status == "BLOCK"
    assert "rr_missing" in by_key["rr_ok"].block_reasons


# ─────────────────────────────────────────────────────────────────
# 0-line warnings — WARN, not BLOCK
# ─────────────────────────────────────────────────────────────────

def test_zero_support_resistance_is_warn_not_block():
    inputs = _ready_inputs()
    inputs["support_resistance_v2"] = {
        "selected_level_zones_top5": [],
        "rejected_level_zones": [],
    }
    cl = build_royal_road_procedure_checklist(**inputs)
    by_key = {s.key: s for s in cl.steps}
    assert by_key["support_resistance"].status == "WARN"
    # 0-SR is P1, so p0_pass is still True for the rest of the
    # READY-shaped input.
    assert cl.p0_pass is True


def test_zero_trendlines_is_warn_not_block():
    inputs = _ready_inputs()
    inputs["trendline_context"] = {
        "selected_trendlines_top3": [],
        "rejected_trendlines": [],
    }
    cl = build_royal_road_procedure_checklist(**inputs)
    by_key = {s.key: s for s in cl.steps}
    assert by_key["trendline_context"].status == "WARN"
    assert cl.p0_pass is True


# ─────────────────────────────────────────────────────────────────
# Wave pattern missing → BLOCK on P0
# ─────────────────────────────────────────────────────────────────

def test_missing_pattern_kind_blocks_wave_pattern_step():
    inputs = _ready_inputs()
    inputs["pattern_levels"] = {"available": False}
    cl = build_royal_road_procedure_checklist(**inputs)
    by_key = {s.key: s for s in cl.steps}
    assert by_key["wave_pattern"].status == "BLOCK"
    assert "wave_pattern_missing" in by_key["wave_pattern"].block_reasons
    assert cl.p0_pass is False


# ─────────────────────────────────────────────────────────────────
# All-empty input — should be the deterministic HOLD path
# ─────────────────────────────────────────────────────────────────

def test_all_empty_input_returns_hold_with_14_steps():
    cl = build_royal_road_procedure_checklist(
        entry_plan=None,
        pattern_levels=None,
        wave_derived_lines=None,
        breakout_quality_gate=None,
        fundamental_sidebar=None,
        support_resistance_v2=None,
        trendline_context=None,
        dow_structure_review=None,
        candlestick_anatomy_review=None,
    )
    assert len(cl.steps) == 14
    assert cl.final_status == "HOLD"
    assert cl.p0_pass is False
    assert cl.ready is False
    assert "HOLD" in cl.summary_ja


# ─────────────────────────────────────────────────────────────────
# Each step is a frozen dataclass — pin invariants
# ─────────────────────────────────────────────────────────────────

def test_each_step_is_a_dataclass_with_to_dict():
    cl = build_royal_road_procedure_checklist(**_ready_inputs())
    for s in cl.steps:
        assert isinstance(s, RoyalRoadProcedureStep)
        d = s.to_dict()
        assert d["key"] == s.key
        assert d["label_ja"] == s.label_ja
        assert d["status"] in (
            "PASS", "WAIT", "BLOCK", "WARN", "UNKNOWN"
        )
        assert d["importance"] in ("P0", "P1", "P2")
