"""Tests for structural_lines (Phase I follow-up).

Pin the schema, the per-source builder paths, the numeric-alignment
annotation, and the snapshot counts.
"""
from __future__ import annotations

from src.fx.structural_lines import (
    STRUCTURAL_LINES_SCHEMA_VERSION,
    StructuralLine,
    StructuralLinesSnapshot,
    annotate_numeric_alignment,
    build_structural_lines,
)


def _base_pattern_levels() -> dict:
    return {
        "available": True,
        "side": "SELL",
        "pattern_kind": "double_top",
        "trigger_line_id": "WNL1",
        "trigger_line_price": 1.1000,
        "stop_price": 1.1030,
        "target_price": 1.0950,
        "target_extended_price": 1.0910,
        "parts": {
            "P1": {"index": 10, "price": 1.1050},
            "NL": {"index": 15, "price": 1.1000},
            "P2": {"index": 20, "price": 1.1045},
            "BR": {"index": 25, "price": 1.0980},
        },
    }


def _base_pattern_levels_db() -> dict:
    return {
        "available": True,
        "side": "BUY",
        "pattern_kind": "double_bottom",
        "trigger_line_id": "WNL1",
        "trigger_line_price": 1.1000,
        "stop_price": 1.0950,
        "target_price": 1.1050,
        "target_extended_price": 1.1080,
        "parts": {
            "B1": {"index": 10, "price": 1.0960},
            "NL": {"index": 15, "price": 1.1000},
            "B2": {"index": 20, "price": 1.0965},
            "BR": {"index": 25, "price": 1.1010},
        },
    }


# ─────────────────────────────────────────────────────────────────
# Schema + return type
# ─────────────────────────────────────────────────────────────────

def test_schema_version_pin():
    assert STRUCTURAL_LINES_SCHEMA_VERSION == "structural_lines_v1"


def test_build_returns_snapshot_with_to_dict():
    snap = build_structural_lines(
        pattern_levels=None, wave_derived_lines=None,
        multi_scale_chart=None, dow_structure_review=None,
        support_resistance_v2=None, trendline_context=None,
    )
    assert isinstance(snap, StructuralLinesSnapshot)
    d = snap.to_dict()
    assert d["schema_version"] == STRUCTURAL_LINES_SCHEMA_VERSION
    assert "lines" in d
    assert "counts" in d
    assert "summary_ja" in d


# ─────────────────────────────────────────────────────────────────
# wave_derived_lines path
# ─────────────────────────────────────────────────────────────────

def test_build_structural_lines_from_wave_derived_lines():
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels(),
        wave_derived_lines=[
            {"id": "WNL1", "role": "entry_confirmation_line",
             "price": 1.1000},
            {"id": "WSL1", "role": "stop_candidate", "price": 1.1030},
            {"id": "WTP1", "role": "target_candidate", "price": 1.0950},
        ],
        multi_scale_chart={},
        dow_structure_review={"trend": "DOWN"},
        support_resistance_v2={},
        trendline_context={},
    )
    d = snap.to_dict()
    assert d["counts"]["total"] >= 3

    kinds = {line["kind"] for line in d["lines"]}
    roles = {line["role"] for line in d["lines"]}

    assert "structural_neckline" in kinds
    assert "structural_invalidation" in kinds
    assert "structural_target" in kinds
    assert "entry_trigger" in roles
    assert "stop_candidate" in roles
    assert "target_candidate" in roles


def test_wave_derived_lines_carry_anchor_parts_and_reason():
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels(),
        wave_derived_lines=[
            {"id": "WNL1", "role": "entry_confirmation_line",
             "price": 1.1000},
        ],
        multi_scale_chart={}, dow_structure_review={},
        support_resistance_v2={}, trendline_context={},
    )
    d = snap.to_dict()
    snl = next(
        line for line in d["lines"]
        if line["kind"] == "structural_neckline"
    )
    assert snl["anchor_parts"] == ["NL"]
    assert "ネックライン" in snl["reason_ja"]
    assert snl["source"] == "wave_derived_lines"
    assert snl["used_in_royal_road"] is True


# ─────────────────────────────────────────────────────────────────
# pattern_levels fallback
# ─────────────────────────────────────────────────────────────────

def test_pattern_levels_fallback_creates_neckline_stop_target():
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels(),
        wave_derived_lines=[],
        multi_scale_chart={}, dow_structure_review={},
        support_resistance_v2={}, trendline_context={},
    )
    d = snap.to_dict()
    assert d["counts"]["total"] >= 3
    assert any(
        line["kind"] == "structural_neckline" for line in d["lines"]
    )
    assert any(
        line["kind"] == "structural_invalidation" for line in d["lines"]
    )
    assert any(
        line["kind"] == "structural_target" for line in d["lines"]
    )
    # Each fallback line carries source=pattern_levels.
    fallback_lines = [
        line for line in d["lines"]
        if line["kind"] in (
            "structural_neckline",
            "structural_invalidation",
            "structural_target",
        )
    ]
    assert all(
        ln["source"] == "pattern_levels" for ln in fallback_lines
    )


def test_wave_derived_lines_take_precedence_over_pattern_levels():
    """When both wave_derived_lines and pattern_levels carry the same
    kind, the wave-derived version wins and pattern_levels does NOT
    duplicate it."""
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels(),
        wave_derived_lines=[
            {"id": "WNL1", "role": "entry_confirmation_line",
             "price": 1.1000},
            {"id": "WSL1", "role": "stop_candidate", "price": 1.1030},
            {"id": "WTP1", "role": "target_candidate", "price": 1.0950},
        ],
        multi_scale_chart={}, dow_structure_review={},
        support_resistance_v2={}, trendline_context={},
    )
    d = snap.to_dict()
    assert d["counts"]["structural_neckline"] == 1
    assert d["counts"]["structural_invalidation"] == 1
    assert d["counts"]["structural_target"] == 1
    snl = next(ln for ln in d["lines"] if ln["kind"] == "structural_neckline")
    assert snl["source"] == "wave_derived_lines"


# ─────────────────────────────────────────────────────────────────
# DT P1↔P2 / DB B1↔B2 → structural trendline
# ─────────────────────────────────────────────────────────────────

def test_double_top_p1_p2_creates_structural_resistance_line():
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels(),
        wave_derived_lines=[],
        multi_scale_chart={}, dow_structure_review={},
        support_resistance_v2={}, trendline_context={},
    )
    d = snap.to_dict()
    assert any(
        line["kind"] == "structural_trendline"
        and line["role"] in (
            "downtrend_resistance",
            "resistance",
            "top_structure_line",
        )
        for line in d["lines"]
    )


def test_double_bottom_b1_b2_creates_structural_support_line():
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels_db(),
        wave_derived_lines=[],
        multi_scale_chart={}, dow_structure_review={},
        support_resistance_v2={}, trendline_context={},
    )
    d = snap.to_dict()
    assert any(
        line["kind"] == "structural_trendline"
        and line["role"] in (
            "uptrend_support",
            "support",
            "bottom_structure_line",
        )
        for line in d["lines"]
    )


# ─────────────────────────────────────────────────────────────────
# Numeric alignment
# ─────────────────────────────────────────────────────────────────

def test_numeric_alignment_none_when_no_numeric_trendlines():
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels(),
        wave_derived_lines=[],
        multi_scale_chart={}, dow_structure_review={},
        support_resistance_v2={}, trendline_context={},
    )
    d = snap.to_dict()
    # Diagonal structural trendline must be NONE; horizontal lines
    # stay UNKNOWN (numeric layer is for trendlines only).
    diagonal = [
        ln for ln in d["lines"]
        if ln["kind"] == "structural_trendline"
    ]
    if diagonal:
        for ln in diagonal:
            assert ln["numeric_alignment"] == "NONE"


def test_numeric_alignment_match_when_slope_close():
    """Structural P1-P2 slope is (1.1045 - 1.105)/(20 - 10) = -0.00005.
    A numeric T1 with slope -0.00005 should MATCH (rel diff 0)."""
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels(),
        wave_derived_lines=[],
        multi_scale_chart={}, dow_structure_review={},
        support_resistance_v2={},
        trendline_context={
            "selected_trendlines_top3": [
                {"id": "T1", "slope": -0.00005},
            ]
        },
    )
    d = snap.to_dict()
    diagonal = [
        ln for ln in d["lines"]
        if ln["kind"] == "structural_trendline"
    ]
    assert diagonal, "expected one structural trendline from DT"
    assert diagonal[0]["numeric_alignment"] == "MATCH"
    assert diagonal[0]["matched_numeric_line_id"] == "T1"


def test_numeric_alignment_conflict_on_sign_mismatch():
    """Structural P1-P2 slope is negative; a numeric +0.001 should
    produce CONFLICT (sign mismatch)."""
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels(),
        wave_derived_lines=[],
        multi_scale_chart={}, dow_structure_review={},
        support_resistance_v2={},
        trendline_context={
            "selected_trendlines_top3": [
                {"id": "T1", "slope": 0.001},
            ]
        },
    )
    d = snap.to_dict()
    diagonal = [
        ln for ln in d["lines"]
        if ln["kind"] == "structural_trendline"
    ]
    assert diagonal
    assert diagonal[0]["numeric_alignment"] == "CONFLICT"


# ─────────────────────────────────────────────────────────────────
# multi_scale_chart pivot extraction
# ─────────────────────────────────────────────────────────────────

def test_multi_scale_higher_lows_become_uptrend_support():
    snap = build_structural_lines(
        pattern_levels={"available": False, "side": "BUY"},
        wave_derived_lines=[],
        multi_scale_chart={
            "wave_skeleton": {
                "pivots": [
                    {"index": 0,  "price": 1.10, "kind": "L"},
                    {"index": 5,  "price": 1.12, "kind": "H"},
                    {"index": 8,  "price": 1.11, "kind": "L"},
                    {"index": 12, "price": 1.13, "kind": "H"},
                    {"index": 16, "price": 1.115, "kind": "L"},
                ],
            },
        },
        dow_structure_review={"trend": "UP"},
        support_resistance_v2={}, trendline_context={},
    )
    d = snap.to_dict()
    msc_lines = [
        ln for ln in d["lines"]
        if ln["source"] == "multi_scale_chart"
    ]
    assert msc_lines, "no structural trendline built from multi_scale"
    assert msc_lines[0]["role"] == "uptrend_support"
    assert msc_lines[0]["kind"] == "structural_trendline"


def test_multi_scale_pivots_insufficient_caution():
    snap = build_structural_lines(
        pattern_levels={"available": False, "side": "BUY"},
        wave_derived_lines=[],
        multi_scale_chart={"wave_skeleton": {"pivots": []}},
        dow_structure_review={"trend": "UP"},
        support_resistance_v2={}, trendline_context={},
    )
    d = snap.to_dict()
    assert (
        "multi_scale_pivots_insufficient_for_structural_trendline"
        in d["cautions"]
    )


# ─────────────────────────────────────────────────────────────────
# Counts + summary
# ─────────────────────────────────────────────────────────────────

def test_counts_aggregate_kinds_and_alignments():
    snap = build_structural_lines(
        pattern_levels=_base_pattern_levels(),
        wave_derived_lines=[
            {"id": "WNL1", "role": "entry_confirmation_line",
             "price": 1.1000},
            {"id": "WSL1", "role": "stop_candidate", "price": 1.1030},
            {"id": "WTP1", "role": "target_candidate", "price": 1.0950},
        ],
        multi_scale_chart={}, dow_structure_review={"trend": "DOWN"},
        support_resistance_v2={},
        trendline_context={
            "selected_trendlines_top3": [
                {"id": "T1", "slope": 0.001},  # CONFLICT vs P1-P2
            ]
        },
    )
    d = snap.to_dict()
    counts = d["counts"]
    # 3 horizontal + 1 diagonal P1-P2
    assert counts["total"] == 4
    assert counts["structural_neckline"] == 1
    assert counts["structural_invalidation"] == 1
    assert counts["structural_target"] == 1
    assert counts["structural_trendline"] == 1
    assert counts["numeric_conflict"] == 1
    # Summary string mentions the count.
    assert "構造ライン" in d["summary_ja"]


def test_empty_inputs_produce_empty_snapshot_with_summary():
    snap = build_structural_lines(
        pattern_levels=None, wave_derived_lines=None,
        multi_scale_chart=None, dow_structure_review=None,
        support_resistance_v2=None, trendline_context=None,
    )
    d = snap.to_dict()
    assert d["counts"]["total"] == 0
    assert "構造ライン" in d["summary_ja"]


# ─────────────────────────────────────────────────────────────────
# Regression: numeric trendline + support_resistance_v2 not removed
# (this module is purely additive, no detector code is touched)
# ─────────────────────────────────────────────────────────────────

def test_module_does_not_touch_numeric_inputs():
    """Calling build_structural_lines must not mutate the input dicts.
    """
    pl = _base_pattern_levels()
    pl_snapshot = repr(pl)
    tc = {
        "selected_trendlines_top3": [
            {"id": "T1", "slope": -0.0001},
        ]
    }
    tc_snapshot = repr(tc)
    sr = {"selected_level_zones_top5": [{"low": 1.09, "high": 1.092}]}
    sr_snapshot = repr(sr)
    build_structural_lines(
        pattern_levels=pl,
        wave_derived_lines=[],
        multi_scale_chart={}, dow_structure_review={},
        support_resistance_v2=sr,
        trendline_context=tc,
    )
    assert repr(pl) == pl_snapshot
    assert repr(tc) == tc_snapshot
    assert repr(sr) == sr_snapshot
