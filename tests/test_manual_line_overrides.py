"""Tests for manual_line_overrides — policy + override behaviour.

Spec required cases:
  - manual_line_policy=display_only: USER lines never have
    used_in_decision=True
  - manual_line_policy=assist: USER lines appear in audit but still
    used_in_decision=False
  - manual_line_policy=decision_override: USER lines CAN influence
  - reject_auto_line filters the auto line out of the rendered list
  - default policy is display_only
"""
from __future__ import annotations

import pytest

from src.fx.manual_line_overrides import (
    DEFAULT_POLICY,
    POLICY_ASSIST,
    POLICY_DECISION_OVERRIDE,
    POLICY_DISPLAY_ONLY,
    SUPPORTED_POLICIES,
    apply_user_overrides_to_wave_lines,
    validate_policy,
)
from src.fx.user_chart_annotations import (
    add_user_line, empty_annotations, reject_auto_line,
)


def _auto() -> list[dict]:
    return [
        {"id": "AUTO_WNL1", "kind": "neckline", "price": 1.10,
         "role": "entry_confirmation_line", "used_in_decision": False},
        {"id": "AUTO_WSL1", "kind": "pattern_invalidation",
         "price": 1.09, "role": "stop_candidate", "used_in_decision": False},
        {"id": "AUTO_WTP1", "kind": "pattern_target",
         "price": 1.13, "role": "target_candidate", "used_in_decision": False},
    ]


# ── Defaults / policy validation ─────────────────────────────────
def test_default_policy_is_display_only():
    assert DEFAULT_POLICY == POLICY_DISPLAY_ONLY == "display_only"


def test_supported_policies_three():
    assert set(SUPPORTED_POLICIES) == {
        "display_only", "assist", "decision_override",
    }


def test_validate_policy_accepts_known():
    for p in SUPPORTED_POLICIES:
        assert validate_policy(p) == p


def test_validate_policy_rejects_typo():
    with pytest.raises(ValueError):
        validate_policy("decision-override")  # hyphen, not underscore


# ── No annotations → pass-through ────────────────────────────────
def test_no_annotations_keeps_auto_lines():
    auto = _auto()
    final, summary = apply_user_overrides_to_wave_lines(auto, None)
    assert [l["id"] for l in final] == ["AUTO_WNL1", "AUTO_WSL1", "AUTO_WTP1"]
    assert summary["auto_kept"] == ["AUTO_WNL1", "AUTO_WSL1", "AUTO_WTP1"]
    assert summary["auto_rejected"] == []
    assert summary["user_added"] == []
    assert summary["user_decision_used"] is False
    assert summary["policy"] == "display_only"


# ── Reject auto line filters it out ──────────────────────────────
def test_reject_auto_line_removed_from_render_list():
    c = empty_annotations("USDJPY=X", "1h")
    c = reject_auto_line(c, auto_line_id="AUTO_WNL1",
                         reject_reason_ja="too high")
    final, summary = apply_user_overrides_to_wave_lines(_auto(), c)
    ids = [l["id"] for l in final]
    assert "AUTO_WNL1" not in ids  # rejected → filtered
    assert "AUTO_WSL1" in ids
    assert summary["auto_rejected"] == ["AUTO_WNL1"]


# ── User add line in display_only never decisions ────────────────
def test_display_only_user_line_not_used_in_decision():
    c = empty_annotations("USDJPY=X", "1h",
                          manual_line_policy=POLICY_DISPLAY_ONLY)
    c = add_user_line(
        c, annotation_id="USER_001", kind="support",
        points=[{"ts": "2026-04-20T10:00:00+00:00", "price": 150.10}],
    )
    final, summary = apply_user_overrides_to_wave_lines(_auto(), c)
    user_line = next(l for l in final if l["id"] == "USER_001")
    assert user_line["used_in_decision"] is False
    assert summary["user_decision_used"] is False
    # User line shows up in render list
    assert "USER_001" in [l["id"] for l in final]


# ── assist policy: still no decision input ───────────────────────
def test_assist_policy_user_line_not_used_in_decision():
    c = empty_annotations("USDJPY=X", "1h",
                          manual_line_policy=POLICY_ASSIST)
    c = add_user_line(
        c, annotation_id="USER_002", kind="resistance",
        points=[{"ts": "2026-04-20T10:00:00+00:00", "price": 152.0}],
    )
    final, summary = apply_user_overrides_to_wave_lines(_auto(), c)
    user_line = next(l for l in final if l["id"] == "USER_002")
    assert user_line["used_in_decision"] is False
    assert summary["policy"] == "assist"


# ── decision_override policy: explicit acceptance ────────────────
def test_decision_override_policy_can_let_user_line_decide():
    """When the policy is decision_override AND the annotation was
    constructed with used_in_decision=True, the line is allowed to
    influence decisions. AUTO source still cannot."""
    c = empty_annotations("USDJPY=X", "1h",
                          manual_line_policy=POLICY_DECISION_OVERRIDE)
    c = add_user_line(
        c, annotation_id="USER_003", kind="trendline",
        points=[{"ts": "2026-04-20T10:00:00+00:00", "price": 150.5}],
        label="manual trendline",
    )
    # add_user_line produces used_in_decision=False by default, but
    # the policy will not force it down. Manually flip the contract
    # by replacing the annotation tuple.
    new_anns = list(c.annotations)
    a = new_anns[0]
    # ChartAnnotation is frozen — produce a copy with the override
    from dataclasses import replace
    new_anns[0] = replace(a, used_in_decision=True,
                          source="USER_OVERRIDE")
    c2 = c.__class__(
        schema_version=c.schema_version, symbol=c.symbol,
        timeframe=c.timeframe,
        manual_line_policy=c.manual_line_policy,
        annotations=tuple(new_anns),
    )
    final, summary = apply_user_overrides_to_wave_lines(_auto(), c2)
    user_line = next(l for l in final if l["id"] == "USER_003")
    assert user_line["used_in_decision"] is True
    assert summary["user_decision_used"] is True


# ── Container's policy used when no policy kwarg passed ──────────
def test_container_policy_picked_up_when_no_kwarg():
    c = empty_annotations("USDJPY=X", "1h",
                          manual_line_policy=POLICY_ASSIST)
    final, summary = apply_user_overrides_to_wave_lines(_auto(), c)
    assert summary["policy"] == "assist"


def test_kwarg_policy_overrides_container():
    c = empty_annotations("USDJPY=X", "1h",
                          manual_line_policy=POLICY_DISPLAY_ONLY)
    final, summary = apply_user_overrides_to_wave_lines(
        _auto(), c, policy=POLICY_ASSIST,
    )
    assert summary["policy"] == "assist"
