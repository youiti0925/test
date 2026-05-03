"""Tests that decision_bridge upgrades the AUDIT_ONLY classification
when the integrated profile is active, and that visual_audit emits the
integrated-profile banner.

The legacy v2 path must keep its existing AUDIT_ONLY classification —
those existing tests (tests/test_decision_bridge.py) continue to pass
under the same payload shape.
"""
from __future__ import annotations

import re

from src.fx.decision_bridge import (
    AUDIT_ONLY, PARTIAL, SCHEMA_VERSION, USED,
    build_decision_bridge,
)
from src.fx.visual_audit import _render_decision_bridge_html


# ── Payload helpers ──────────────────────────────────────────────
def _integrated_v2_slice(
    *, action: str = "BUY", mode: str = "integrated_balanced",
    block_reasons: list[str] | None = None,
    axes_overrides: dict[str, str] | None = None,
) -> dict:
    """Build a v2 slice that represents an integrated-profile decision."""
    overrides = axes_overrides or {}
    axes = [
        {
            "axis": "wave_pattern",
            "side": "BUY" if action == "BUY" else "SELL",
            "status": overrides.get("chart_patterns", "PASS"),
            "strength": 0.8, "confidence": 0.7,
            "used_in_decision": True, "required": False,
            "reason_ja": "波形 PASS",
            "source": "chart_patterns",
        },
        {
            "axis": "wave_lines",
            "side": "BUY" if action == "BUY" else "SELL",
            "status": overrides.get("wave_derived_lines", "PASS"),
            "strength": 0.9, "confidence": 0.85,
            "used_in_decision": True, "required": True,
            "reason_ja": "WNL 突破 + WSL/WTP + RR>=2",
            "source": "wave_derived_lines",
        },
        {
            "axis": "fibonacci",
            "side": "BUY" if action == "BUY" else "SELL",
            "status": overrides.get("source_pack_fibonacci", "PASS"),
            "strength": 0.7, "confidence": 0.5,
            "used_in_decision": True, "required": False,
            "reason_ja": "fib 50.0–61.8 PASS",
            "source": "source_pack_fibonacci",
        },
        {
            "axis": "dow_structure",
            "side": "BUY" if action == "BUY" else "SELL",
            "status": "PASS",
            "strength": 0.8, "confidence": 0.8,
            "used_in_decision": True, "required": True,
            "reason_ja": "ダウ UP",
            "source": "masterclass_dow",
        },
        {
            "axis": "candlestick",
            "side": "BUY" if action == "BUY" else "SELL",
            "status": "PASS",
            "strength": 0.8, "confidence": 0.7,
            "used_in_decision": True, "required": False,
            "reason_ja": "ピンバー at_support",
            "source": "masterclass_candlestick",
        },
        {
            "axis": "daily_roadmap", "side": "NEUTRAL",
            "status": overrides.get("source_pack_daily_roadmap", "PASS"),
            "strength": 0.5, "confidence": 0.5,
            "used_in_decision": True, "required": False,
            "reason_ja": "roadmap OK",
            "source": "source_pack_daily_roadmap",
        },
    ]
    return {
        "action": action,
        "profile": "royal_road_decision_v2_integrated",
        "block_reasons": block_reasons or [],
        "structure_stop_plan": {
            "stop_price": 0.99, "chosen_mode": "structure", "outcome": "ok",
        },
        "macro_alignment": {"macro_score": 0.0},
        "support_resistance_v2": {"near_strong_support": True},
        "integrated_decision": {
            "schema_version": "royal_road_integrated_decision_v1",
            "mode": mode,
            "action": action,
            "label": f"{action}_INTEGRATED",
            "side_bias": action if action in ("BUY", "SELL") else "NEUTRAL",
            "confidence": 0.7,
            "block_reasons": block_reasons or [],
            "cautions": [],
            "axes": axes,
            "used_modules": ["chart_patterns", "wave_derived_lines"],
            "audit_only_modules": [],
            "not_connected_modules": [],
            "explanation_ja": f"integrated {action}",
        },
    }


def _legacy_v2_slice(action: str = "HOLD") -> dict:
    """Existing royal_road_decision_v2 (NOT integrated) slice."""
    return {
        "action": action,
        "profile": "royal_road_decision_v2",
        "block_reasons": [] if action != "HOLD" else ["weak_setup"],
        "structure_stop_plan": {"stop_price": 0.99},
        "macro_alignment": {"macro_score": 0.0},
        "support_resistance_v2": {"near_strong_support": True},
    }


def _payload(v2_slice: dict, *, with_panels: bool = True) -> dict:
    pl = {
        "royal_road_decision_v2": v2_slice,
        "wave_shape_review": {"available": True},
        "wave_derived_lines": [{"id": "WNL1", "price": 1.0}],
        "entry_summary": {"entry_price": 1.0, "stop_price": 0.99,
                          "take_profit_price": 1.025, "rr": 2.5},
    }
    if with_panels:
        pl["masterclass_panels"] = {
            "available": True,
            "panels": {
                "fibonacci_context_review": {"available": True},
                "daily_roadmap_review": {"available": True},
            },
        }
    return pl


# ── Bridge classification under integrated profile ───────────────
def test_integrated_flag_set_when_profile_is_integrated():
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    assert bridge["available"] is True
    assert bridge["integrated_profile_active"] is True
    assert bridge["integrated_mode"] == "integrated_balanced"
    assert bridge["final_action"] == "BUY"
    assert bridge["final_action_source"] == "royal_road_decision_v2_integrated"


def test_legacy_v2_keeps_audit_only_classification():
    """Critical regression: when the integrated profile is NOT active,
    the bridge must emit the same AUDIT_ONLY entries as before."""
    bridge = build_decision_bridge(_payload(_legacy_v2_slice("HOLD")))
    assert bridge["integrated_profile_active"] is False
    assert bridge["integrated_mode"] is None
    audit_categories = {
        e["category"]: e["status"]
        for e in bridge["audit_only_references"]
    }
    assert audit_categories.get("wave_shape_review") == AUDIT_ONLY
    assert audit_categories.get("wave_derived_lines") == AUDIT_ONLY
    assert audit_categories.get("masterclass_panels") == AUDIT_ONLY
    assert audit_categories.get("fibonacci_context_review") == AUDIT_ONLY


def test_integrated_layout_p0_axes_in_used_list():
    """P0 axes (wave_pattern, wave_lines, dow_structure,
    invalidation_rr) MUST appear in used_for_final_decision when
    integrated profile is active. raw panels go to audit_only."""
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    used_cats = {e["category"] for e in bridge["used_for_final_decision"]}
    # P0 axis entries must all be in used_for_final_decision
    assert "wave_pattern_axis" in used_cats, used_cats
    assert "wave_lines_axis" in used_cats, used_cats
    assert "dow_structure_axis" in used_cats, used_cats


def test_integrated_layout_p1_axes_in_used_list():
    """P1 axes (fibonacci, candlestick, ma, levels) → used list."""
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    used_cats = {e["category"] for e in bridge["used_for_final_decision"]}
    assert "fibonacci_axis" in used_cats, used_cats
    assert "candlestick_axis" in used_cats, used_cats


def test_integrated_layout_raw_panels_in_audit_only():
    """raw panel display (Grand Confluence summary, raw masterclass,
    wave raw display) must appear in audit_only_references — NOT in
    used_for_final_decision — to make the distinction explicit."""
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    audit_cats = {e["category"] for e in bridge["audit_only_references"]}
    assert "wave_shape_review_raw" in audit_cats, audit_cats
    assert "wave_derived_lines_raw" in audit_cats, audit_cats
    assert "masterclass_panels_raw" in audit_cats, audit_cats
    # raw entries must all be AUDIT_ONLY status
    for e in bridge["audit_only_references"]:
        assert e["status"] == AUDIT_ONLY, (
            f"raw panel entry {e['category']!r} must be AUDIT_ONLY, "
            f"got {e['status']!r}"
        )


def test_integrated_fibonacci_pass_appears_as_used():
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    by_cat = {e["category"]: e for e in bridge["used_for_final_decision"]}
    assert by_cat["fibonacci_axis"]["status"] == USED, by_cat["fibonacci_axis"]


def test_integrated_fibonacci_warn_appears_as_partial():
    """Integrated PASS=USED, WARN=PARTIAL (P1 — block-as-partial)."""
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(
        action="BUY",
        axes_overrides={"source_pack_fibonacci": "WARN"},
    )))
    by_cat = {e["category"]: e for e in bridge["used_for_final_decision"]}
    assert by_cat["fibonacci_axis"]["status"] == PARTIAL


def test_integrated_p0_block_reports_as_used_not_partial():
    """When a P0 required axis BLOCKs (forces HOLD), the bridge
    reports it as USED — the BLOCK actively drove the action."""
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(
        action="HOLD",
        block_reasons=["wave_lines_blocked"],
        axes_overrides={"wave_derived_lines": "BLOCK"},
    )))
    by_cat = {e["category"]: e for e in bridge["used_for_final_decision"]}
    assert by_cat["wave_lines_axis"]["status"] == USED, (
        f"P0 BLOCK must report as USED, got {by_cat['wave_lines_axis']['status']!r}"
    )


def test_integrated_used_for_final_decision_uses_integrated_core():
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    used_cats = {e["category"] for e in bridge["used_for_final_decision"]}
    assert "royal_road_integrated_core" in used_cats
    # Legacy "royal_road_v2_core" must NOT appear (different label)
    assert "royal_road_v2_core" not in used_cats


def test_integrated_action_message_states_integrated_logic():
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    msg = bridge["action_message_ja"]
    assert "王道統合ロジック" in msg
    assert "波形" in msg
    assert "Wライン" in msg


def test_integrated_hold_message_includes_block_summary():
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(
        action="HOLD",
        block_reasons=["wave_lines_blocked", "invalidation_rr_blocked"],
    )))
    msg = bridge["action_message_ja"]
    assert "HOLD" in msg
    assert "wave_lines_blocked" in msg or "invalidation_rr_blocked" in msg


def test_integrated_strict_mode_message_mentions_strict():
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(
        action="HOLD", mode="integrated_strict",
        block_reasons=["macro_blocked"],
    )))
    assert "strict" in bridge["action_message_ja"].lower()


def test_integrated_balanced_mode_message_mentions_balanced():
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(
        action="BUY", mode="integrated_balanced",
    )))
    assert "balanced" in bridge["action_message_ja"]


def test_legacy_action_message_unchanged():
    """Existing audit-only message must remain for the legacy v2."""
    bridge = build_decision_bridge(_payload(_legacy_v2_slice("BUY")))
    msg = bridge["action_message_ja"]
    assert "現時点では補助監査" in msg
    assert "王道統合ロジック" not in msg


# ── visual_audit HTML banner ─────────────────────────────────────
def test_html_renders_integrated_banner():
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    html = _render_decision_bridge_html(
        bridge, entry_summary={"entry_price": 1.0, "rr": 2.5},
    )
    assert "integrated-banner" in html
    assert "王道統合ロジック" in html
    assert "balanced" in html


def test_html_no_integrated_banner_for_legacy_v2():
    bridge = build_decision_bridge(_payload(_legacy_v2_slice("BUY")))
    html = _render_decision_bridge_html(
        bridge, entry_summary={"entry_price": 1.0, "rr": 2.5},
    )
    assert "integrated-banner" not in html
    assert "王道統合ロジック" not in html


def test_html_strict_banner_text():
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(
        action="BUY", mode="integrated_strict",
    )))
    html = _render_decision_bridge_html(bridge)
    assert "integrated-banner" in html
    assert "strict" in html


# ── Schema invariants ────────────────────────────────────────────
def test_schema_version_unchanged():
    """The schema_version constant must stay at decision_bridge_v1.
    The integrated-profile flag is an additive field, not a new schema."""
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    assert bridge["schema_version"] == SCHEMA_VERSION == "decision_bridge_v1"


def test_observation_only_contract_preserved():
    """The bridge ITSELF is observation-only — even when it reports
    the integrated profile is active, used_in_decision MUST be False."""
    bridge = build_decision_bridge(_payload(_integrated_v2_slice(action="BUY")))
    assert bridge["observation_only"] is True
    assert bridge["used_in_decision"] is False


def test_integrated_branch_does_not_break_unavailable_payload():
    bridge = build_decision_bridge(None)
    assert bridge["available"] is False
    assert "integrated_profile_active" not in bridge or bridge.get(
        "integrated_profile_active"
    ) is False or bridge.get("integrated_profile_active") is None
