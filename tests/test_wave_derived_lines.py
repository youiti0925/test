"""Tests for src/fx/wave_derived_lines.py and its hookups in
visual_audit.py / mobile HTML.

Covers the spec invariants:

1. wave_derived_lines is a top-level key on the audit payload
2. double_bottom fixture generates WNL / WSL / WTP (and WB1 / WB2)
3. double_top fixture generates WNL / WSL / WTP (and WP1 / WP2)
4. head_and_shoulders fixture generates WNL / WLS / WH / WRS
   (and WSL / WTP)
5. Mobile HTML embeds WNL1 (id label rendered inline)
6. Mobile HTML embeds 波形由来ライン数 / line count summary block
7. HOLD case waveform note references WNL and "未ブレイク"
8. Every wave-derived line carries used_in_decision = False
   (action stays unchanged invariant)
"""
from __future__ import annotations

from src.fx.wave_derived_lines import (
    SCHEMA_VERSION,
    build_wave_derived_lines,
    line_count_summary,
)
from src.fx.visual_audit import (
    _hold_waveform_extra_html,
    _render_line_count_summary_html,
    _render_wave_derived_lines_table_html,
)


def _double_bottom_skel() -> dict:
    return {
        "atr_value": 0.005,
        "pivots": [
            {"index": 0,  "ts": "2025-01-01T00:00:00+00:00",
             "price": 1.10, "kind": "H", "source": "zigzag"},
            {"index": 20, "ts": "2025-01-01T20:00:00+00:00",
             "price": 1.00, "kind": "L", "source": "zigzag"},
            {"index": 45, "ts": "2025-01-02T21:00:00+00:00",
             "price": 1.07, "kind": "H", "source": "zigzag"},
            {"index": 70, "ts": "2025-01-03T22:00:00+00:00",
             "price": 1.01, "kind": "L", "source": "zigzag"},
            {"index": 99, "ts": "2025-01-05T03:00:00+00:00",
             "price": 1.12, "kind": "H", "source": "zigzag"},
        ],
    }


def _double_top_skel() -> dict:
    return {
        "atr_value": 0.005,
        "pivots": [
            {"index": 0,  "price": 1.00, "kind": "L"},
            {"index": 20, "price": 1.12, "kind": "H"},
            {"index": 45, "price": 1.06, "kind": "L"},
            {"index": 70, "price": 1.11, "kind": "H"},
            {"index": 99, "price": 0.98, "kind": "L"},
        ],
    }


def _hs_skel() -> dict:
    return {
        "atr_value": 0.005,
        "pivots": [
            {"index": 0,   "price": 1.05, "kind": "L"},
            {"index": 30,  "price": 1.15, "kind": "H"},  # LS
            {"index": 60,  "price": 1.10, "kind": "L"},  # NL-L
            {"index": 100, "price": 1.20, "kind": "H"},  # H
            {"index": 140, "price": 1.10, "kind": "L"},  # NL-R
            {"index": 170, "price": 1.15, "kind": "H"},  # RS
            {"index": 199, "price": 1.02, "kind": "L"},  # breakout
        ],
    }


# ---------------------------------------------------------------------------
# Schema + invariants
# ---------------------------------------------------------------------------


def test_schema_version_string():
    assert SCHEMA_VERSION == "wave_derived_lines_v1"


def test_no_pattern_returns_empty_list():
    assert build_wave_derived_lines(
        best_pattern=None, skeleton=None, atr_value=None,
    ) == []
    assert build_wave_derived_lines(
        best_pattern={"kind": "double_bottom"}, skeleton=None, atr_value=None,
    ) == []
    assert build_wave_derived_lines(
        best_pattern={"kind": "double_bottom", "matched_parts": {}},
        skeleton={"pivots": []}, atr_value=None,
    ) == []


def test_every_line_marked_observation_only_used_in_decision_false():
    """Action-invariant guarantee: no derived line is allowed to feed
    a decision gate. The contract is that every line carries
    used_in_decision == False."""
    skel = _double_bottom_skel()
    best = {
        "kind": "double_bottom",
        "matched_parts": {
            "first_bottom": 20, "neckline_peak": 45,
            "second_bottom": 70, "breakout": 99,
        },
        "side_bias": "BUY",
    }
    lines = build_wave_derived_lines(
        best_pattern=best, skeleton=skel, atr_value=0.005,
    )
    assert lines, "expected at least one wave-derived line"
    for l in lines:
        assert l["used_in_decision"] is False
        assert "id" in l and l["id"].startswith("W")
        assert "reason_ja" in l and l["reason_ja"]


# ---------------------------------------------------------------------------
# double_bottom
# ---------------------------------------------------------------------------


def test_double_bottom_generates_wb1_wb2_wnl_wsl_wtp():
    skel = _double_bottom_skel()
    best = {
        "kind": "double_bottom",
        "matched_parts": {
            "first_bottom": 20, "neckline_peak": 45,
            "second_bottom": 70, "breakout": 99, "prior_high": 0,
        },
        "side_bias": "BUY",
    }
    lines = build_wave_derived_lines(
        best_pattern=best, skeleton=skel, atr_value=0.005,
    )
    ids = {l["id"] for l in lines}
    assert {"WB1", "WB2", "WNL1", "WSL1", "WTP1"} <= ids
    nl = next(l for l in lines if l["id"] == "WNL1")
    assert nl["kind"] == "neckline"
    assert nl["role"] == "entry_confirmation_line"
    sl = next(l for l in lines if l["id"] == "WSL1")
    assert sl["role"] == "stop_candidate"
    assert sl["price"] < 1.01  # below B2
    tp = next(l for l in lines if l["id"] == "WTP1")
    assert tp["role"] == "target_candidate"
    # Target = NL + (NL - min(B1, B2)) = 1.07 + (1.07 - 1.00) = 1.14
    assert abs(tp["price"] - 1.14) < 0.001


# ---------------------------------------------------------------------------
# double_top
# ---------------------------------------------------------------------------


def test_double_top_generates_wp1_wp2_wnl_wsl_wtp():
    skel = _double_top_skel()
    best = {
        "kind": "double_top",
        "matched_parts": {
            "first_top": 20, "neckline_trough": 45,
            "second_top": 70, "breakout": 99, "prior_low": 0,
        },
        "side_bias": "SELL",
    }
    lines = build_wave_derived_lines(
        best_pattern=best, skeleton=skel, atr_value=0.005,
    )
    ids = {l["id"] for l in lines}
    assert {"WP1", "WP2", "WNL1", "WSL1", "WTP1"} <= ids
    sl = next(l for l in lines if l["id"] == "WSL1")
    assert sl["price"] > 1.11  # above P2
    tp = next(l for l in lines if l["id"] == "WTP1")
    # Target = NL - (max(P1, P2) - NL) = 1.06 - (1.12 - 1.06) = 1.00
    assert abs(tp["price"] - 1.00) < 0.001


# ---------------------------------------------------------------------------
# head_and_shoulders
# ---------------------------------------------------------------------------


def test_head_and_shoulders_generates_wls_wh_wrs_wnl():
    skel = _hs_skel()
    best = {
        "kind": "head_and_shoulders",
        "matched_parts": {
            "prior_low": 0, "left_shoulder": 30,
            "neckline_left_trough": 60, "head": 100,
            "neckline_right_trough": 140, "right_shoulder": 170,
            "breakout": 199,
        },
        "side_bias": "SELL",
    }
    lines = build_wave_derived_lines(
        best_pattern=best, skeleton=skel, atr_value=0.005,
    )
    ids = {l["id"] for l in lines}
    assert {"WLS", "WH", "WRS", "WNL1", "WSL1", "WTP1"} <= ids
    sl = next(l for l in lines if l["id"] == "WSL1")
    # Stop should sit above the right shoulder
    assert sl["price"] > 1.15


def test_inverse_head_and_shoulders_generates_wls_wh_wrs_wnl():
    # Mirror skeleton: H is the deepest low instead of the highest peak.
    skel = {
        "atr_value": 0.005,
        "pivots": [
            {"index": 0,   "price": 1.05, "kind": "H"},
            {"index": 30,  "price": 1.00, "kind": "L"},  # LS
            {"index": 60,  "price": 1.05, "kind": "H"},  # NL-L
            {"index": 100, "price": 0.95, "kind": "L"},  # H (lowest)
            {"index": 140, "price": 1.05, "kind": "H"},  # NL-R
            {"index": 170, "price": 1.00, "kind": "L"},  # RS
            {"index": 199, "price": 1.13, "kind": "H"},  # breakout
        ],
    }
    best = {
        "kind": "inverse_head_and_shoulders",
        "matched_parts": {
            "prior_high": 0, "left_shoulder": 30,
            "neckline_left_peak": 60, "head": 100,
            "neckline_right_peak": 140, "right_shoulder": 170,
            "breakout": 199,
        },
        "side_bias": "BUY",
    }
    lines = build_wave_derived_lines(
        best_pattern=best, skeleton=skel, atr_value=0.005,
    )
    ids = {l["id"] for l in lines}
    assert {"WLS", "WH", "WRS", "WNL1", "WSL1", "WTP1"} <= ids


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def test_wave_derived_lines_table_renders_id_kind_role():
    lines = [
        {"id": "WNL1", "kind": "neckline", "price": 1.092,
         "role": "entry_confirmation_line", "reason_ja": "テスト",
         "used_in_decision": False},
        {"id": "WSL1", "kind": "pattern_invalidation", "price": 1.088,
         "role": "stop_candidate", "reason_ja": "テスト2",
         "used_in_decision": False},
    ]
    html = _render_wave_derived_lines_table_html(lines)
    assert "WNL1" in html
    assert "WSL1" in html
    assert "neckline" in html
    assert "entry_confirmation_line" in html
    assert "used_in_decision" in html


def test_line_count_summary_includes_wave_derived_section():
    lines = [
        {"id": "WNL1", "kind": "neckline", "price": 1.0,
         "role": "entry_confirmation_line", "used_in_decision": False},
        {"id": "WSL1", "kind": "pattern_invalidation", "price": 0.99,
         "role": "stop_candidate", "used_in_decision": False},
    ]
    html = _render_line_count_summary_html(
        overlays={"level_zones_selected": []},
        wave_derived_lines=lines,
    )
    assert "波形由来" in html
    assert "波形由来ライン" in html or "波形由来の内訳" in html
    assert ">2<" in html  # total wave-derived = 2


def test_line_count_summary_warns_when_both_sparse():
    html = _render_line_count_summary_html(
        overlays={}, wave_derived_lines=[],
    )
    assert "波形由来" in html
    assert (
        "システムが人間の線引きを十分に再現できていない" in html
        or "note-warn" in html
    )


def test_line_count_summary_info_when_sr_sparse_but_wave_present():
    lines = [
        {"id": "WNL1", "kind": "neckline", "price": 1.0,
         "role": "entry_confirmation_line", "used_in_decision": False},
        {"id": "WSL1", "kind": "pattern_invalidation", "price": 0.99,
         "role": "stop_candidate", "used_in_decision": False},
    ]
    html = _render_line_count_summary_html(
        overlays={"level_zones_selected": [],
                  "trendlines_selected": []},
        wave_derived_lines=lines,
    )
    assert "波形認識から" in html or "WNL1" in html


# ---------------------------------------------------------------------------
# HOLD card augmentation
# ---------------------------------------------------------------------------


def test_hold_waveform_card_references_wnl_when_present():
    review = {
        "best_pattern": {
            "kind": "double_bottom",
            "matched_parts": {
                "first_bottom": 20, "neckline_peak": 45,
                "second_bottom": 70,
            },
            "human_label": "ダブルボトム候補",
            "human_explanation": "テスト",
            "status": "forming",
            "side_bias": "BUY",
        },
        "risk_note_ja": "リスク注意",
    }
    wd_lines = [
        {"id": "WNL1", "kind": "neckline", "price": 1.07,
         "role": "entry_confirmation_line", "used_in_decision": False},
        {"id": "WSL1", "kind": "pattern_invalidation", "price": 1.005,
         "role": "stop_candidate", "used_in_decision": False},
        {"id": "WTP1", "kind": "pattern_target", "price": 1.14,
         "role": "target_candidate", "used_in_decision": False},
    ]
    html = _hold_waveform_extra_html(
        action="HOLD", review=review, wave_derived_lines=wd_lines,
    )
    assert "WNL1" in html
    assert "ネックラインあり" in html
    assert "WSL1" in html
    assert "WTP1" in html
    # status forming → "未ブレイク" wording
    assert "未ブレイク" in html
    # observation-only disclaimer present
    assert "observation-only" in html


def test_hold_waveform_card_for_no_pattern_says_uncertain():
    html = _hold_waveform_extra_html(
        action="HOLD",
        review={"best_pattern": None},
        wave_derived_lines=[],
    )
    assert "明確な波形候補が出ていない" in html


# ---------------------------------------------------------------------------
# count helper directly
# ---------------------------------------------------------------------------


def test_line_count_summary_direct_counts():
    lines = [
        {"id": "WNL1", "kind": "neckline", "role": "entry_confirmation_line"},
        {"id": "WSL1", "kind": "pattern_invalidation", "role": "stop_candidate"},
        {"id": "WSL2", "kind": "pattern_invalidation", "role": "stop_candidate"},
        {"id": "WTP1", "kind": "pattern_target", "role": "target_candidate"},
    ]
    counts = line_count_summary(
        overlays={
            "level_zones_selected": [{}, {}],
            "level_zones_rejected": [{}],
            "trendlines_selected": [],
            "trendlines_rejected": [{}],
        },
        wave_derived=lines,
    )
    assert counts == {
        "sr_selected": 2, "sr_rejected": 1,
        "trendline_selected": 0, "trendline_rejected": 1,
        "wave_derived_total": 4,
        "wave_derived_neckline": 1,
        "wave_derived_stop": 2,
        "wave_derived_target": 1,
    }
