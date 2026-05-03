"""user_chart_annotations_v1 — user-supplied chart annotations.

The integrated decision draws automatic lines (WNL / WSL / WTP /
WBR / pivots), but those lines can be wrong. Phase G adds a
serialisable container for the user to:

  - draw their own lines (USER source)
  - reject an automatic line (REJECTED_BY_USER status)
  - mark an automatic line as accepted (AUTO_ACCEPTED)
  - override an automatic line with a corrected version (USER_OVERRIDE)
  - delete a user-supplied line later (DELETED_BY_USER)

Every annotation carries provenance (source / status / created_at /
updated_at) so the system can later learn which automatic lines the
human consistently rejects.

Strict default: `manual_line_policy = display_only`. Even when the
user adds a line, it does NOT influence BUY / SELL / HOLD until the
user explicitly raises the policy to `assist` or `decision_override`.

Schema:

  {
    "schema_version": "user_chart_annotations_v1",
    "symbol": str,
    "timeframe": str,
    "manual_line_policy": "display_only" | "assist" | "decision_override",
    "annotations": [
      {
        "id": str,                     # USER_SR_001 / AUTO_WNL1 / etc.
        "kind": "support" / "resistance" / "trendline" / "neckline"
                / "stop" / "target" / "note",
        "source": "AUTO" / "USER" / "AUTO_ACCEPTED" / "USER_OVERRIDE"
                  / "REJECTED_BY_USER" / "DELETED_BY_USER",
        "status": "ACTIVE" / "REJECTED_BY_USER" / "DELETED_BY_USER",
        "points": [{"ts": str, "price": float}, ...],
        "label": str,
        "created_at": str,
        "updated_at": str,
        "used_in_decision": bool,
        "reject_reason_ja": str | None,
      }, ...
    ]
  }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Final


SCHEMA_VERSION: Final[str] = "user_chart_annotations_v1"


# Source provenance tokens
SOURCE_AUTO: Final[str] = "AUTO"
SOURCE_USER: Final[str] = "USER"
SOURCE_AUTO_ACCEPTED: Final[str] = "AUTO_ACCEPTED"
SOURCE_USER_OVERRIDE: Final[str] = "USER_OVERRIDE"
SOURCE_REJECTED_BY_USER: Final[str] = "REJECTED_BY_USER"
SOURCE_DELETED_BY_USER: Final[str] = "DELETED_BY_USER"

SUPPORTED_SOURCES: Final[tuple[str, ...]] = (
    SOURCE_AUTO, SOURCE_USER, SOURCE_AUTO_ACCEPTED,
    SOURCE_USER_OVERRIDE, SOURCE_REJECTED_BY_USER, SOURCE_DELETED_BY_USER,
)

# Status tokens (intersect with source for some values)
STATUS_ACTIVE: Final[str] = "ACTIVE"
STATUS_REJECTED_BY_USER: Final[str] = "REJECTED_BY_USER"
STATUS_DELETED_BY_USER: Final[str] = "DELETED_BY_USER"

SUPPORTED_STATUS: Final[tuple[str, ...]] = (
    STATUS_ACTIVE, STATUS_REJECTED_BY_USER, STATUS_DELETED_BY_USER,
)

# Annotation kinds
SUPPORTED_KINDS: Final[tuple[str, ...]] = (
    "support", "resistance", "trendline", "neckline",
    "stop", "target", "note",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ChartAnnotation:
    """A single user / auto chart annotation. Frozen so accidental
    mutation is caught; updates produce a new instance via
    `with_updates()`."""
    id: str
    kind: str
    source: str
    status: str = STATUS_ACTIVE
    points: tuple[dict, ...] = ()
    label: str = ""
    created_at: str = ""
    updated_at: str = ""
    used_in_decision: bool = False
    reject_reason_ja: str | None = None

    def __post_init__(self) -> None:
        if self.source not in SUPPORTED_SOURCES:
            raise ValueError(f"unknown source: {self.source!r}")
        if self.status not in SUPPORTED_STATUS:
            raise ValueError(f"unknown status: {self.status!r}")
        if self.kind not in SUPPORTED_KINDS:
            raise ValueError(f"unknown kind: {self.kind!r}")
        # used_in_decision is a hard contract: only USER_OVERRIDE +
        # USER + AUTO_ACCEPTED can ever be True; AUTO/REJECTED/DELETED
        # are always False.
        if self.used_in_decision and self.source in (
            SOURCE_AUTO, SOURCE_REJECTED_BY_USER, SOURCE_DELETED_BY_USER,
        ):
            raise ValueError(
                f"source={self.source} cannot have used_in_decision=True"
            )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "source": self.source,
            "status": self.status,
            "points": [dict(p) for p in self.points],
            "label": self.label,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "used_in_decision": bool(self.used_in_decision),
            "reject_reason_ja": self.reject_reason_ja,
        }


def from_dict(d: dict) -> ChartAnnotation:
    """Inverse of `ChartAnnotation.to_dict()` — used by the JSON loader."""
    return ChartAnnotation(
        id=str(d.get("id") or ""),
        kind=str(d.get("kind") or "note"),
        source=str(d.get("source") or SOURCE_USER),
        status=str(d.get("status") or STATUS_ACTIVE),
        points=tuple(
            {"ts": str(p.get("ts") or ""),
             "price": float(p.get("price"))
             if p.get("price") is not None else None}
            for p in (d.get("points") or [])
        ),
        label=str(d.get("label") or ""),
        created_at=str(d.get("created_at") or ""),
        updated_at=str(d.get("updated_at") or ""),
        used_in_decision=bool(d.get("used_in_decision") or False),
        reject_reason_ja=d.get("reject_reason_ja"),
    )


@dataclass(frozen=True)
class UserChartAnnotations:
    schema_version: str = SCHEMA_VERSION
    symbol: str = ""
    timeframe: str = ""
    manual_line_policy: str = "display_only"
    annotations: tuple[ChartAnnotation, ...] = ()

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "manual_line_policy": self.manual_line_policy,
            "annotations": [a.to_dict() for a in self.annotations],
        }


def empty_annotations(
    symbol: str, timeframe: str,
    *, manual_line_policy: str = "display_only",
) -> UserChartAnnotations:
    return UserChartAnnotations(
        symbol=symbol, timeframe=timeframe,
        manual_line_policy=manual_line_policy,
        annotations=(),
    )


def add_user_line(
    container: UserChartAnnotations,
    *,
    annotation_id: str,
    kind: str,
    points: list[dict],
    label: str = "",
) -> UserChartAnnotations:
    """Add a USER-source ACTIVE annotation. Returns a NEW container."""
    now = _now_iso()
    new_ann = ChartAnnotation(
        id=annotation_id, kind=kind,
        source=SOURCE_USER, status=STATUS_ACTIVE,
        points=tuple(points),
        label=label or annotation_id,
        created_at=now, updated_at=now,
        used_in_decision=False,
    )
    return UserChartAnnotations(
        schema_version=container.schema_version,
        symbol=container.symbol,
        timeframe=container.timeframe,
        manual_line_policy=container.manual_line_policy,
        annotations=container.annotations + (new_ann,),
    )


def reject_auto_line(
    container: UserChartAnnotations,
    *, auto_line_id: str, kind: str = "neckline",
    reject_reason_ja: str = "",
) -> UserChartAnnotations:
    """Record that the user rejected an auto-built line.

    Stores REJECTED_BY_USER status (no physical deletion) so the
    system can later learn which auto lines the user disagrees with.
    """
    now = _now_iso()
    rejection = ChartAnnotation(
        id=auto_line_id, kind=kind,
        source=SOURCE_REJECTED_BY_USER,
        status=STATUS_REJECTED_BY_USER,
        points=(),
        label=auto_line_id,
        created_at=now, updated_at=now,
        used_in_decision=False,
        reject_reason_ja=reject_reason_ja or None,
    )
    return UserChartAnnotations(
        schema_version=container.schema_version,
        symbol=container.symbol,
        timeframe=container.timeframe,
        manual_line_policy=container.manual_line_policy,
        annotations=container.annotations + (rejection,),
    )


def delete_user_line(
    container: UserChartAnnotations, *, annotation_id: str,
) -> UserChartAnnotations:
    """Mark a USER annotation as DELETED_BY_USER (kept in history)."""
    now = _now_iso()
    new_anns: list[ChartAnnotation] = []
    for ann in container.annotations:
        if ann.id == annotation_id and ann.source == SOURCE_USER:
            new_anns.append(ChartAnnotation(
                id=ann.id, kind=ann.kind,
                source=SOURCE_DELETED_BY_USER,
                status=STATUS_DELETED_BY_USER,
                points=ann.points,
                label=ann.label,
                created_at=ann.created_at,
                updated_at=now,
                used_in_decision=False,
                reject_reason_ja=ann.reject_reason_ja,
            ))
        else:
            new_anns.append(ann)
    return UserChartAnnotations(
        schema_version=container.schema_version,
        symbol=container.symbol,
        timeframe=container.timeframe,
        manual_line_policy=container.manual_line_policy,
        annotations=tuple(new_anns),
    )


def active_annotations(
    container: UserChartAnnotations,
) -> list[ChartAnnotation]:
    """Return only annotations with status=ACTIVE."""
    return [a for a in container.annotations if a.status == STATUS_ACTIVE]


def rejected_auto_ids(
    container: UserChartAnnotations,
) -> set[str]:
    """Return the set of auto-line ids the user has rejected.

    Used by visual_audit to filter out auto lines the user said no to.
    """
    return {
        a.id for a in container.annotations
        if a.source == SOURCE_REJECTED_BY_USER
        or a.status == STATUS_REJECTED_BY_USER
    }


__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_AUTO", "SOURCE_USER", "SOURCE_AUTO_ACCEPTED",
    "SOURCE_USER_OVERRIDE", "SOURCE_REJECTED_BY_USER",
    "SOURCE_DELETED_BY_USER",
    "SUPPORTED_SOURCES",
    "STATUS_ACTIVE", "STATUS_REJECTED_BY_USER", "STATUS_DELETED_BY_USER",
    "SUPPORTED_STATUS",
    "SUPPORTED_KINDS",
    "ChartAnnotation",
    "UserChartAnnotations",
    "empty_annotations",
    "add_user_line",
    "reject_auto_line",
    "delete_user_line",
    "active_annotations",
    "rejected_auto_ids",
    "from_dict",
]
