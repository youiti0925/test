"""Tests for pattern_level_derivation_v1.

Covers the user spec's required cases 1-4:

  1. double_bottom from skeleton sequence yields B1/B2/NL/BR + stop/target/RR
  2. double_top yields P1/P2/NL/BR + stop/target/RR
  3. head_and_shoulders yields LS/H/RS/NL/BR + stop/target
  4. inverse_head_and_shoulders yields LS/H/RS/NL/BR + stop/target
"""
from __future__ import annotations

from src.fx.pattern_level_derivation import (
    SCHEMA_VERSION,
    derive_pattern_levels,
)


def _skeleton(pivots: list[tuple[int, str, float]]) -> dict:
    """Build a minimal skeleton dict with the given pivots."""
    return {
        "schema_version": "wave_skeleton_v1",
        "scale": "medium",
        "bars_used": 300,
        "pivots": [
            {"index": idx, "kind": kind, "price": price,
             "ts": "2025-01-01T00:00:00+00:00",
             "source": "test", "strength": 1.0}
            for idx, kind, price in pivots
        ],
        "atr_value": 0.005,
        "trend_hint": "UP",
    }


def _wave_lines() -> list[dict]:
    return []


# ── Case 1: double_bottom ────────────────────────────────────────
def test_double_bottom_extracts_b1_b2_nl_br():
    skel = _skeleton([
        (50, "L", 1.0900),    # B1
        (130, "H", 1.1000),   # NL
        (210, "L", 1.0910),   # B2
        (290, "H", 1.1100),   # BR
    ])
    pl = derive_pattern_levels(
        wave_shape_review={
            "best_pattern": {
                "kind": "double_bottom",
                "side_bias": "BUY",
                "status": "neckline_broken",
                "matched_parts": {},
                "scale": "medium",
            },
        },
        wave_derived_lines=_wave_lines(),
        skeleton=skel,
        last_close=1.1100,
    )
    assert pl["available"] is True
    assert pl["pattern_kind"] == "double_bottom"
    assert pl["side"] == "BUY"
    assert "B1" in pl["parts"]
    assert "B2" in pl["parts"]
    assert "NL" in pl["parts"]
    assert "BR" in pl["parts"]
    assert pl["parts"]["B1"]["price"] == 1.0900
    assert pl["parts"]["NL"]["price"] == 1.1000
    assert pl["parts"]["B2"]["price"] == 1.0910
    assert pl["stop_price"] is not None
    assert pl["target_price"] is not None
    assert pl["target_extended_price"] is not None
    assert pl["rr_at_extended_target"] is not None
    assert pl["breakout_confirmed"] is True


# ── Case 2: double_top ──────────────────────────────────────────
def test_double_top_extracts_p1_p2_nl_br():
    skel = _skeleton([
        (50, "H", 1.1000),    # P1
        (130, "L", 1.0900),   # NL
        (210, "H", 1.0995),   # P2
        (290, "L", 1.0800),   # BR
    ])
    pl = derive_pattern_levels(
        wave_shape_review={
            "best_pattern": {
                "kind": "double_top",
                "side_bias": "SELL",
                "status": "neckline_broken",
                "matched_parts": {},
                "scale": "medium",
            },
        },
        wave_derived_lines=_wave_lines(),
        skeleton=skel,
        last_close=1.0800,
    )
    assert pl["available"] is True
    assert pl["pattern_kind"] == "double_top"
    assert pl["side"] == "SELL"
    assert "P1" in pl["parts"]
    assert "P2" in pl["parts"]
    assert "NL" in pl["parts"]
    assert "BR" in pl["parts"]
    assert pl["parts"]["P1"]["price"] == 1.1000
    assert pl["parts"]["NL"]["price"] == 1.0900
    assert pl["stop_price"] is not None
    assert pl["target_price"] is not None
    assert pl["target_extended_price"] is not None


# ── Case 3: head_and_shoulders ───────────────────────────────────
def test_head_and_shoulders_extracts_ls_h_rs_nl_br():
    # H&S sequence: H L H L H L
    skel = _skeleton([
        (40, "H", 1.0950),    # LS
        (90, "L", 1.0900),    # neck low 1
        (140, "H", 1.1100),   # H (head — highest)
        (190, "L", 1.0905),   # neck low 2
        (240, "H", 1.0960),   # RS
        (290, "L", 1.0850),   # BR (below neck)
    ])
    pl = derive_pattern_levels(
        wave_shape_review={
            "best_pattern": {
                "kind": "head_and_shoulders",
                "side_bias": "SELL",
                "status": "neckline_broken",
                "matched_parts": {},
                "scale": "medium",
            },
        },
        wave_derived_lines=_wave_lines(),
        skeleton=skel,
        last_close=1.0850,
    )
    assert pl["available"] is True
    assert pl["pattern_kind"] == "head_and_shoulders"
    assert pl["side"] == "SELL"
    assert "LS" in pl["parts"]
    assert "H" in pl["parts"]
    assert "RS" in pl["parts"]
    assert "NL" in pl["parts"]
    assert "BR" in pl["parts"]
    assert pl["parts"]["H"]["price"] == 1.1100
    assert pl["parts"]["LS"]["price"] == 1.0950
    assert pl["stop_price"] is not None
    assert pl["target_price"] is not None


# ── Case 4: inverse_head_and_shoulders ───────────────────────────
def test_inverse_head_and_shoulders_extracts_ls_h_rs_nl_br():
    skel = _skeleton([
        (40, "L", 1.0900),    # LS
        (90, "H", 1.0995),    # neck high 1
        (140, "L", 1.0850),   # H (lowest)
        (190, "H", 1.0990),   # neck high 2
        (240, "L", 1.0905),   # RS
        (290, "H", 1.1050),   # BR (above neck)
    ])
    pl = derive_pattern_levels(
        wave_shape_review={
            "best_pattern": {
                "kind": "inverse_head_and_shoulders",
                "side_bias": "BUY",
                "status": "neckline_broken",
                "matched_parts": {},
                "scale": "medium",
            },
        },
        wave_derived_lines=_wave_lines(),
        skeleton=skel,
        last_close=1.1050,
    )
    assert pl["available"] is True
    assert pl["pattern_kind"] == "inverse_head_and_shoulders"
    assert pl["side"] == "BUY"
    assert "LS" in pl["parts"]
    assert "H" in pl["parts"]
    assert "RS" in pl["parts"]
    assert "NL" in pl["parts"]
    assert "BR" in pl["parts"]
    assert pl["parts"]["H"]["price"] == 1.0850
    assert pl["stop_price"] is not None
    assert pl["target_price"] is not None


# ── Schema invariants ────────────────────────────────────────────
def test_schema_version_is_v1():
    pl = derive_pattern_levels(
        wave_shape_review=None,
        wave_derived_lines=None,
        skeleton=None,
        last_close=None,
    )
    assert pl["schema_version"] == SCHEMA_VERSION == "pattern_levels_v1"
    assert pl["available"] is False


def test_missing_inputs_returns_empty_panel():
    pl = derive_pattern_levels(
        wave_shape_review=None,
        wave_derived_lines=None,
        skeleton=None,
        last_close=None,
    )
    assert pl["available"] is False
    assert pl["unavailable_reason"] == "missing_wave_shape_review"


def test_no_best_pattern_returns_empty():
    pl = derive_pattern_levels(
        wave_shape_review={"best_pattern": None},
        wave_derived_lines=None,
        skeleton=None,
        last_close=None,
    )
    assert pl["available"] is False
    assert pl["unavailable_reason"] == "no_best_pattern"


def test_extended_target_is_2x_pattern_height():
    """Extended target should be approximately 2× pattern height
    above the trigger line, used for the integrated RR gate."""
    skel = _skeleton([
        (50, "L", 1.0900),
        (130, "H", 1.1000),
        (210, "L", 1.0910),
        (290, "H", 1.1100),
    ])
    pl = derive_pattern_levels(
        wave_shape_review={
            "best_pattern": {
                "kind": "double_bottom",
                "side_bias": "BUY",
                "status": "neckline_broken",
                "matched_parts": {},
                "scale": "medium",
            },
        },
        wave_derived_lines=_wave_lines(),
        skeleton=skel,
        last_close=1.1100,
    )
    nl = pl["parts"]["NL"]["price"]   # 1.1000
    height = pl["pattern_height"]
    target_ext = pl["target_extended_price"]
    # target_ext should be ≈ NL + 2*height
    assert abs(target_ext - (nl + 2 * height)) < 1e-6
