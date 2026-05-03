"""manual_line_overrides — apply user_chart_annotations to the
auto-built wave_derived_lines + pattern_levels stream.

The integration point is intentionally narrow: this module only
filters / annotates the existing auto-line lists. It NEVER mutates
prices or invents new entries unless the user explicitly authorised
them via `manual_line_policy`.

Three policies (defined in user_chart_annotations):

  display_only  (default, safe)
    User lines are displayed on the chart but contribute NOTHING to
    BUY / SELL / HOLD logic. Rejected auto lines are still hidden
    from the chart (the user said "stop showing me this").

  assist
    User lines + auto-line rejections appear in the audit trail and
    affect display/sidebar reasoning. Still no direct action input.

  decision_override
    User lines CAN influence entry/stop/target derivation. This is
    deliberately gated and turned OFF by default.

`apply_user_overrides_to_wave_lines` is the canonical entrypoint:
takes the raw auto wave_derived_lines + a UserChartAnnotations
container, returns a NEW list of lines with rejected ones filtered
out and user lines appended (display-only by default).
"""
from __future__ import annotations

from typing import Final

from .user_chart_annotations import (
    SOURCE_USER, SOURCE_USER_OVERRIDE,
    STATUS_ACTIVE,
    UserChartAnnotations,
    active_annotations,
    rejected_auto_ids,
)


POLICY_DISPLAY_ONLY: Final[str] = "display_only"
POLICY_ASSIST: Final[str] = "assist"
POLICY_DECISION_OVERRIDE: Final[str] = "decision_override"

SUPPORTED_POLICIES: Final[tuple[str, ...]] = (
    POLICY_DISPLAY_ONLY, POLICY_ASSIST, POLICY_DECISION_OVERRIDE,
)

DEFAULT_POLICY: Final[str] = POLICY_DISPLAY_ONLY


def validate_policy(policy: str) -> str:
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            f"unknown manual_line_policy: {policy!r}. "
            f"Supported: {', '.join(SUPPORTED_POLICIES)}"
        )
    return policy


def _annotation_to_line(ann) -> dict:
    """Convert a USER annotation into a wave_derived_lines-compatible
    dict so the chart renderer treats it uniformly. Importantly,
    `used_in_decision=False` is forced for display_only policy."""
    points = list(ann.points or ())
    price = None
    if points:
        try:
            price = float(points[0].get("price"))
        except (TypeError, ValueError, AttributeError):
            price = None
    return {
        "id": ann.id,
        "kind": ann.kind,
        "source_pattern": "user_manual",
        "price": price,
        "zone_low": None,
        "zone_high": None,
        "role": ann.kind,
        "used_in_decision": bool(ann.used_in_decision),
        "reason_ja": ann.label or "ユーザー追加の手動ライン",
        "user_provenance": ann.source,
    }


def apply_user_overrides_to_wave_lines(
    wave_derived_lines: list[dict] | None,
    annotations: UserChartAnnotations | None,
    *,
    policy: str | None = None,
) -> tuple[list[dict], dict]:
    """Filter rejected auto lines and append user annotations.

    Returns (final_lines, summary_dict). The summary records what
    was kept / rejected / added so visual_audit can render an
    explanation of what the user changed.
    """
    if not wave_derived_lines:
        wave_derived_lines = []
    if annotations is None:
        return list(wave_derived_lines), {
            "schema_version": "manual_line_overrides_summary_v1",
            "policy": DEFAULT_POLICY,
            "auto_kept": [ln.get("id") for ln in wave_derived_lines if ln.get("id")],
            "auto_rejected": [],
            "user_added": [],
            "user_decision_used": False,
        }

    effective_policy = validate_policy(
        policy or annotations.manual_line_policy or DEFAULT_POLICY
    )
    rejected = rejected_auto_ids(annotations)

    auto_kept_ids: list[str] = []
    auto_rejected_ids: list[str] = []
    final_lines: list[dict] = []
    for ln in wave_derived_lines:
        ln_id = ln.get("id")
        if ln_id and ln_id in rejected:
            auto_rejected_ids.append(ln_id)
            continue
        # Auto lines NEVER influence decision under display_only/assist.
        # Only flagged-true under decision_override (and even then,
        # only when the user explicitly accepted them).
        adjusted = dict(ln)
        if effective_policy != POLICY_DECISION_OVERRIDE:
            adjusted["used_in_decision"] = False
        final_lines.append(adjusted)
        if ln_id:
            auto_kept_ids.append(ln_id)

    user_added_ids: list[str] = []
    for ann in active_annotations(annotations):
        if ann.source not in (SOURCE_USER, SOURCE_USER_OVERRIDE):
            continue
        line = _annotation_to_line(ann)
        # Even if the annotation says used_in_decision=True, force False
        # under display_only / assist policies. Only decision_override
        # honors the annotation flag.
        if effective_policy != POLICY_DECISION_OVERRIDE:
            line["used_in_decision"] = False
        final_lines.append(line)
        user_added_ids.append(ann.id)

    return final_lines, {
        "schema_version": "manual_line_overrides_summary_v1",
        "policy": effective_policy,
        "auto_kept": auto_kept_ids,
        "auto_rejected": auto_rejected_ids,
        "user_added": user_added_ids,
        "user_decision_used": (
            effective_policy == POLICY_DECISION_OVERRIDE
            and any(line.get("used_in_decision") for line in final_lines
                    if line.get("source_pattern") == "user_manual")
        ),
    }


__all__ = [
    "POLICY_DISPLAY_ONLY",
    "POLICY_ASSIST",
    "POLICY_DECISION_OVERRIDE",
    "SUPPORTED_POLICIES",
    "DEFAULT_POLICY",
    "validate_policy",
    "apply_user_overrides_to_wave_lines",
]
