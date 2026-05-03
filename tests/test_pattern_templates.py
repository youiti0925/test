"""Tests for src/fx/pattern_templates.py.

The matcher in `pattern_shape_matcher` depends on a stable template
registry. These tests pin:
  - the existence of the major pattern kinds the user expects
  - left-to-right point ordering for every template
  - normalised point ranges in [0, 1]
"""
from __future__ import annotations

from src.fx.pattern_templates import (
    SCHEMA_VERSION,
    PatternTemplate,
    all_templates,
    template_by_id,
    templates_by_kind,
)


def test_schema_version_string():
    assert SCHEMA_VERSION == "pattern_templates_v1"


def test_required_kinds_are_present():
    kinds = {t.kind for t in all_templates()}
    expected = {
        "double_bottom", "double_top",
        "head_and_shoulders", "inverse_head_and_shoulders",
        "bullish_flag", "bearish_flag",
        "rising_wedge", "falling_wedge",
        "ascending_triangle", "descending_triangle", "symmetric_triangle",
    }
    missing = expected - kinds
    assert not missing, f"missing pattern kinds: {missing}"


def test_double_bottom_has_multiple_variants():
    variants = templates_by_kind("double_bottom")
    assert len(variants) >= 2, (
        "expected ≥ 2 double_bottom variants for shape robustness"
    )


def test_template_points_are_left_to_right_normalised():
    for t in all_templates():
        xs = [p[0] for p in t.points]
        ys = [p[1] for p in t.points]
        assert xs == sorted(xs), (
            f"{t.template_id} points are not chronologically ordered"
        )
        for x in xs:
            assert 0.0 <= x <= 1.0, f"{t.template_id} x out of [0,1]: {x}"
        for y in ys:
            assert 0.0 <= y <= 1.0, f"{t.template_id} y out of [0,1]: {y}"


def test_template_required_parts_match_point_count():
    for t in all_templates():
        assert len(t.required_parts) == len(t.points), (
            f"{t.template_id}: required_parts len {len(t.required_parts)} "
            f"!= points len {len(t.points)}"
        )


def test_double_bottom_breakout_is_above_neckline_peak():
    t = template_by_id("double_bottom_v1_standard")
    assert t is not None
    # last point (breakout) y must be ≥ neckline_peak y (mid template)
    breakout_y = t.points[-1][1]
    neckline_idx = t.required_parts.index("neckline_peak")
    neckline_y = t.points[neckline_idx][1]
    assert breakout_y >= neckline_y


def test_double_top_breakout_is_below_neckline_trough():
    t = template_by_id("double_top_v1_standard")
    assert t is not None
    breakout_y = t.points[-1][1]
    neckline_idx = t.required_parts.index("neckline_trough")
    neckline_y = t.points[neckline_idx][1]
    assert breakout_y <= neckline_y


def test_head_and_shoulders_head_is_the_highest_point():
    t = template_by_id("head_and_shoulders_v1_standard")
    assert t is not None
    head_idx = t.required_parts.index("head")
    head_y = t.points[head_idx][1]
    for name, (_x, y) in zip(t.required_parts, t.points):
        if name in ("head", "breakout"):
            continue
        assert head_y >= y, (
            f"head ({head_y}) should dominate {name} ({y})"
        )


def test_inverse_head_and_shoulders_head_is_the_lowest_point():
    t = template_by_id("inverse_head_and_shoulders_v1_standard")
    assert t is not None
    head_idx = t.required_parts.index("head")
    head_y = t.points[head_idx][1]
    for name, (_x, y) in zip(t.required_parts, t.points):
        if name in ("head", "breakout"):
            continue
        assert head_y <= y


def test_template_to_dict_keys():
    t = all_templates()[0]
    d = t.to_dict()
    assert {"template_id", "kind", "side_bias", "points",
            "required_parts", "description_ja"} <= set(d.keys())


def test_template_is_frozen():
    t = all_templates()[0]
    try:
        t.kind = "x"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("PatternTemplate should be frozen")
