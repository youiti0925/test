"""Tests for user_chart_annotations_v1 module.

Spec required cases:
  - user annotation can be saved + round-tripped via from_dict
  - reject_auto_line marks the auto line REJECTED_BY_USER
    (no physical delete; appears in saved JSON)
  - delete_user_line marks USER as DELETED_BY_USER
  - manual_line_policy default is display_only
  - hard contract: AUTO/REJECTED/DELETED can never have
    used_in_decision=True
"""
from __future__ import annotations

import pytest

from src.fx.user_chart_annotations import (
    SCHEMA_VERSION,
    SOURCE_AUTO,
    SOURCE_REJECTED_BY_USER,
    SOURCE_USER,
    STATUS_ACTIVE,
    STATUS_DELETED_BY_USER,
    STATUS_REJECTED_BY_USER,
    ChartAnnotation,
    UserChartAnnotations,
    active_annotations,
    add_user_line,
    delete_user_line,
    empty_annotations,
    from_dict,
    reject_auto_line,
    rejected_auto_ids,
)


# ── Schema + defaults ────────────────────────────────────────────
def test_schema_version_is_v1():
    c = empty_annotations("USDJPY=X", "1h")
    d = c.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION == "user_chart_annotations_v1"


def test_default_manual_line_policy_is_display_only():
    c = empty_annotations("USDJPY=X", "1h")
    assert c.manual_line_policy == "display_only"


# ── User line lifecycle ──────────────────────────────────────────
def test_add_user_line_appends_active_annotation():
    c = empty_annotations("USDJPY=X", "1h")
    c2 = add_user_line(
        c, annotation_id="USER_SR_001", kind="support",
        points=[{"ts": "2026-04-20T10:00:00+00:00", "price": 150.10}],
        label="自分で引いたサポート",
    )
    assert len(c2.annotations) == 1
    a = c2.annotations[0]
    assert a.id == "USER_SR_001"
    assert a.kind == "support"
    assert a.source == SOURCE_USER
    assert a.status == STATUS_ACTIVE
    assert a.used_in_decision is False  # forced by display_only spirit
    assert a.points[0]["price"] == 150.10


def test_add_user_line_returns_new_container_not_mutating_input():
    c = empty_annotations("USDJPY=X", "1h")
    c2 = add_user_line(
        c, annotation_id="USER_001", kind="resistance",
        points=[{"ts": "2026-04-20T10:00:00+00:00", "price": 151.0}],
    )
    assert len(c.annotations) == 0  # original untouched
    assert len(c2.annotations) == 1


# ── Reject auto line ─────────────────────────────────────────────
def test_reject_auto_line_records_rejection():
    c = empty_annotations("USDJPY=X", "1h")
    c2 = reject_auto_line(
        c, auto_line_id="AUTO_WNL1",
        reject_reason_ja="ネックライン位置が高すぎる",
    )
    assert len(c2.annotations) == 1
    a = c2.annotations[0]
    assert a.id == "AUTO_WNL1"
    assert a.source == SOURCE_REJECTED_BY_USER
    assert a.status == STATUS_REJECTED_BY_USER
    assert a.reject_reason_ja == "ネックライン位置が高すぎる"
    assert a.used_in_decision is False


def test_rejected_auto_ids_returns_set():
    c = empty_annotations("USDJPY=X", "1h")
    c2 = reject_auto_line(c, auto_line_id="AUTO_WNL1")
    c3 = reject_auto_line(c2, auto_line_id="AUTO_WSL1")
    ids = rejected_auto_ids(c3)
    assert ids == {"AUTO_WNL1", "AUTO_WSL1"}


# ── Delete user line ─────────────────────────────────────────────
def test_delete_user_line_marks_deleted_by_user():
    c = empty_annotations("USDJPY=X", "1h")
    c2 = add_user_line(
        c, annotation_id="USER_001", kind="support",
        points=[{"ts": "2026-04-20T10:00:00+00:00", "price": 150.0}],
    )
    c3 = delete_user_line(c2, annotation_id="USER_001")
    assert len(c3.annotations) == 1  # not physically deleted
    a = c3.annotations[0]
    assert a.status == STATUS_DELETED_BY_USER


def test_active_annotations_filters_out_deleted_and_rejected():
    c = empty_annotations("USDJPY=X", "1h")
    c2 = add_user_line(
        c, annotation_id="USER_001", kind="support",
        points=[{"ts": "2026-04-20T10:00:00+00:00", "price": 150.0}],
    )
    c3 = reject_auto_line(c2, auto_line_id="AUTO_WNL1")
    active = active_annotations(c3)
    assert len(active) == 1
    assert active[0].id == "USER_001"


# ── Hard contract: provenance ───────────────────────────────────
def test_auto_source_cannot_have_used_in_decision_true():
    with pytest.raises(ValueError):
        ChartAnnotation(
            id="AUTO_X", kind="neckline",
            source=SOURCE_AUTO, status=STATUS_ACTIVE,
            points=({"ts": "2026-04-20T10:00:00+00:00", "price": 150.0},),
            used_in_decision=True,  # <-- forbidden
        )


def test_rejected_source_cannot_have_used_in_decision_true():
    with pytest.raises(ValueError):
        ChartAnnotation(
            id="AUTO_WNL1", kind="neckline",
            source=SOURCE_REJECTED_BY_USER,
            status=STATUS_REJECTED_BY_USER,
            points=(),
            used_in_decision=True,
        )


def test_invalid_kind_raises():
    with pytest.raises(ValueError):
        ChartAnnotation(
            id="X", kind="bogus_kind", source=SOURCE_USER,
            status=STATUS_ACTIVE,
        )


# ── JSON round-trip ──────────────────────────────────────────────
def test_to_dict_from_dict_round_trip():
    c = empty_annotations("USDJPY=X", "1h")
    c = add_user_line(
        c, annotation_id="USER_SR_001", kind="support",
        points=[{"ts": "2026-04-20T10:00:00+00:00", "price": 150.10}],
        label="lvl",
    )
    c = reject_auto_line(c, auto_line_id="AUTO_WNL1",
                         reject_reason_ja="too high")
    d = c.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["symbol"] == "USDJPY=X"
    assert d["timeframe"] == "1h"
    assert d["manual_line_policy"] == "display_only"
    assert len(d["annotations"]) == 2
    # round-trip via from_dict
    rebuilt = [from_dict(a) for a in d["annotations"]]
    assert rebuilt[0].id == "USER_SR_001"
    assert rebuilt[1].source == SOURCE_REJECTED_BY_USER
