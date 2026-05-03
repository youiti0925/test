"""Tests for visual_audit waveform-shape SVG drawing helpers.

These exercise the new SVG overlay helpers
`_wave_overlay_svg_fragment`, `_build_wave_only_svg`, and
`_select_wave_overlay` directly so we don't have to spin up
`run_engine_backtest` (which is slow and synthetic).

Pinned invariants
-----------------
- mobile chart embeds <polyline class='wave-skeleton-line'> when a
  best_pattern is selected
- mobile chart embeds <circle class='wave-pivot-dot'> per pivot
- double_bottom fixture renders B1 / B2 / NL labels in SVG
- head_and_shoulders fixture renders LS / H / RS / NL labels in SVG
- a separate <svg class='wave-only-chart'> block is emitted
- HOLD-action cases include "未ブレイク" (or "形成中") wording in the
  wave-only-chart status banner
- default current_runtime profile produces no wave_shape_review
  (verified via _select_wave_overlay returning None)
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.fx.visual_audit import (
    _build_wave_only_svg,
    _select_wave_overlay,
    _wave_overlay_svg_fragment,
)


def _double_bottom_skel() -> dict:
    return {
        "schema_version": "wave_skeleton_v1",
        "scale": "short",
        "bars_used": 100,
        "pivots": [
            {"index": 0,  "ts": "2025-01-01T00:00:00+00:00",
             "price": 1.10, "kind": "H", "source": "zigzag", "strength": 0.5},
            {"index": 20, "ts": "2025-01-01T20:00:00+00:00",
             "price": 1.00, "kind": "L", "source": "zigzag", "strength": 0.7},
            {"index": 45, "ts": "2025-01-02T21:00:00+00:00",
             "price": 1.07, "kind": "H", "source": "zigzag", "strength": 0.5},
            {"index": 70, "ts": "2025-01-03T22:00:00+00:00",
             "price": 1.01, "kind": "L", "source": "zigzag", "strength": 0.6},
            {"index": 99, "ts": "2025-01-05T03:00:00+00:00",
             "price": 1.12, "kind": "H", "source": "zigzag", "strength": 0.8},
        ],
        "normalized_points": [],
        "price_min": 1.00, "price_max": 1.12,
        "atr_value": 0.005, "trend_hint": "RANGE",
        "reason": "ok", "params": {},
    }


def _hs_skel() -> dict:
    return {
        "schema_version": "wave_skeleton_v1",
        "scale": "short",
        "bars_used": 200,
        "pivots": [
            {"index": 0,   "ts": "2025-01-01T00:00:00+00:00",
             "price": 1.05, "kind": "L", "source": "zigzag", "strength": 0.5},
            {"index": 30,  "ts": "2025-01-02T06:00:00+00:00",
             "price": 1.15, "kind": "H", "source": "zigzag", "strength": 0.6},
            {"index": 60,  "ts": "2025-01-03T12:00:00+00:00",
             "price": 1.10, "kind": "L", "source": "zigzag", "strength": 0.5},
            {"index": 100, "ts": "2025-01-05T04:00:00+00:00",
             "price": 1.20, "kind": "H", "source": "zigzag", "strength": 0.9},
            {"index": 140, "ts": "2025-01-06T20:00:00+00:00",
             "price": 1.10, "kind": "L", "source": "zigzag", "strength": 0.5},
            {"index": 170, "ts": "2025-01-08T02:00:00+00:00",
             "price": 1.15, "kind": "H", "source": "zigzag", "strength": 0.6},
            {"index": 199, "ts": "2025-01-09T07:00:00+00:00",
             "price": 1.02, "kind": "L", "source": "zigzag", "strength": 0.8},
        ],
        "normalized_points": [],
        "price_min": 1.02, "price_max": 1.20,
        "atr_value": 0.01, "trend_hint": "DOWN",
        "reason": "ok", "params": {},
    }


# ---------------------------------------------------------------------------
# wave-only-chart
# ---------------------------------------------------------------------------


def test_wave_only_svg_double_bottom_renders_b1_b2_nl_br_db1():
    skel = _double_bottom_skel()
    matched = {
        "prior_high":   0,
        "first_bottom": 20,
        "neckline_peak": 45,
        "second_bottom": 70,
        "breakout":     99,
    }
    svg = _build_wave_only_svg(
        skeleton=skel, matched_parts=matched, kind="double_bottom",
        human_label="ダブルボトム候補", status="neckline_broken",
        side_bias="BUY",
        entry_summary={"entry_price": 1.119, "stop_price": 0.999,
                       "take_profit_price": 1.135,
                       "structure_stop_price": None,
                       "atr_stop_price": None},
    )
    assert svg is not None
    assert "wave-only-chart" in svg
    assert "wave-skeleton-line" in svg
    assert "wave-pivot-dot" in svg
    for label in (">B1<", ">B2<", ">NL<", ">BR<", "DB1"):
        assert label in svg, f"missing label {label} in wave-only svg"
    # neckline horizontal line
    assert "wave-neckline" in svg


def test_wave_only_svg_head_and_shoulders_renders_ls_h_rs_nl():
    skel = _hs_skel()
    matched = {
        "prior_low":               0,
        "left_shoulder":           30,
        "neckline_left_trough":    60,
        "head":                    100,
        "neckline_right_trough":   140,
        "right_shoulder":          170,
        "breakout":                199,
    }
    svg = _build_wave_only_svg(
        skeleton=skel, matched_parts=matched, kind="head_and_shoulders",
        human_label="三尊候補", status="forming",
        side_bias="SELL",
        entry_summary={},
    )
    assert svg is not None
    for label in (">LS<", ">H<", ">RS<", "HS1", ">BR<"):
        assert label in svg, f"missing label {label} in HS wave-only svg"
    # NL appears as either >NL-L<, >NL-R<, or >NL< — head_and_shoulders
    # uses the L/R variants in this build.
    assert (">NL-L<" in svg) or (">NL-R<" in svg) or (">NL<" in svg), (
        "expected at least one neckline label in HS svg"
    )


def test_wave_only_svg_status_banner_shows_forming_label():
    skel = _double_bottom_skel()
    svg = _build_wave_only_svg(
        skeleton=skel, matched_parts={"first_bottom": 20},
        kind="double_bottom", human_label="ダブルボトム候補",
        status="forming", side_bias="BUY", entry_summary={},
    )
    assert svg is not None
    assert "形成中" in svg


def test_wave_only_svg_returns_none_when_skeleton_is_empty():
    svg = _build_wave_only_svg(
        skeleton={"pivots": []}, matched_parts={},
        kind="double_bottom", human_label="",
        status="not_matched", side_bias="BUY", entry_summary=None,
    )
    assert svg is None


# ---------------------------------------------------------------------------
# candle-chart wave overlay
# ---------------------------------------------------------------------------


def test_candle_overlay_emits_skeleton_polyline_and_pivot_dots():
    skel = _double_bottom_skel()
    visible_index = pd.date_range(
        "2025-01-01", periods=100, freq="1h", tz="UTC",
    )
    matched = {"first_bottom": 20, "neckline_peak": 45,
               "second_bottom": 70, "breakout": 99}

    def x_of(i: int) -> float:
        return 60.0 + (i + 0.5) * (1120.0 / 100)

    def y_of(p: float) -> float:
        return 28.0 + (1.0 - (p - 0.95) / 0.20) * 544.0

    fragment = _wave_overlay_svg_fragment(
        wave_overlay={
            "skeleton": skel, "matched_parts": matched,
            "kind": "double_bottom", "human_label": "ダブルボトム候補",
        },
        visible_index=list(visible_index),
        x_of=x_of, y_of=y_of,
        price_lo=0.95, price_hi=1.15,
        margin_l=60, margin_t=28, plot_w=1120, plot_h=544,
    )
    assert "wave-skeleton-overlay" in fragment
    assert "wave-skeleton-line" in fragment
    assert "wave-pivot-dot" in fragment
    assert ">B1<" in fragment
    assert ">B2<" in fragment
    assert ">NL<" in fragment
    assert ">BR<" in fragment
    assert "wave-neckline" in fragment
    assert "DB1" in fragment


def test_candle_overlay_skips_pivots_outside_visible_window():
    skel = _double_bottom_skel()
    # visible window starts in February — January pivots are out
    visible_index = pd.date_range(
        "2025-02-01", periods=100, freq="1h", tz="UTC",
    )

    def x_of(i: int) -> float:
        return 60.0 + (i + 0.5) * (1120.0 / 100)

    def y_of(p: float) -> float:
        return 28.0 + (1.0 - (p - 0.95) / 0.20) * 544.0

    fragment = _wave_overlay_svg_fragment(
        wave_overlay={"skeleton": skel, "matched_parts": {}, "kind": ""},
        visible_index=list(visible_index),
        x_of=x_of, y_of=y_of,
        price_lo=0.95, price_hi=1.15,
        margin_l=60, margin_t=28, plot_w=1120, plot_h=544,
    )
    # All pivots filtered out → no overlay
    assert fragment == ""


# ---------------------------------------------------------------------------
# _select_wave_overlay
# ---------------------------------------------------------------------------


def test_select_wave_overlay_returns_none_when_no_review():
    payload = {"wave_shape_review": {}, "royal_road_decision_v2": {}}
    assert _select_wave_overlay(payload) is None


def test_select_wave_overlay_returns_skeleton_when_best_pattern_present():
    skel = _double_bottom_skel()
    payload = {
        "wave_shape_review": {
            "best_pattern": {
                "kind": "double_bottom",
                "scale": "short",
                "matched_parts": {"first_bottom": 20, "second_bottom": 70},
                "human_label": "ダブルボトム候補",
                "status": "forming",
                "side_bias": "BUY",
                "shape_score": 0.84,
            },
        },
        "royal_road_decision_v2": {
            "multi_scale_chart": {
                "scales": {"short": {"wave_skeleton": skel}},
            },
        },
    }
    out = _select_wave_overlay(payload)
    assert out is not None
    assert out["kind"] == "double_bottom"
    assert out["skeleton"]["pivots"]
    assert out["matched_parts"] == {"first_bottom": 20, "second_bottom": 70}


def test_select_wave_overlay_handles_missing_scale_skeleton():
    payload = {
        "wave_shape_review": {
            "best_pattern": {
                "kind": "double_bottom", "scale": "short",
                "matched_parts": {}, "human_label": "x", "status": "x",
                "side_bias": "BUY", "shape_score": 0.5,
            },
        },
        "royal_road_decision_v2": {
            "multi_scale_chart": {"scales": {}},
        },
    }
    assert _select_wave_overlay(payload) is None
