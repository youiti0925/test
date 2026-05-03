"""Entry candidate foundation for royal-road integrated profile.

Phase I-1/I-2:
- Add EntryCandidate / EntryMethodContext schema.
- Add common hard gates.
- Add selector.
- Convert existing entry_plan_v1 output into a neckline_retest candidate.
- Observation/display foundation only.

This module must not change current_runtime, live, OANDA, paper, or the
existing royal_road_decision_v2 profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal


ENTRY_CANDIDATE_SCHEMA_VERSION = "entry_candidate_v1"

EntryCandidateStatus = Literal[
    "READY",
    "WAIT_BREAKOUT",
    "WAIT_RETEST",
    "WAIT_TRIGGER",
    "WAIT_EVENT_CLEAR",
    "WATCH",
    "HOLD",
]

EntrySide = Literal["BUY", "SELL", "NEUTRAL"]


@dataclass(frozen=True)
class EntryCandidate:
    schema_version: str
    entry_type: str
    status: EntryCandidateStatus
    side: EntrySide

    entry_price: float | None
    stop_price: float | None
    target_price: float | None
    rr: float | None

    # Larger setup invalidation level.
    setup_invalidation_price: float | None = None

    # Actual trade stop. Initially same as stop_price unless a later
    # short-term method separates them.
    trade_stop_price: float | None = None

    # Trigger / retest state.
    trigger_line_id: str | None = None
    trigger_line_price: float | None = None
    trigger_confirmed: bool = False
    retest_confirmed: bool = False
    confirmation_candle: str | None = None

    # Environment labels.
    htf_alignment: str = "UNKNOWN"
    location_quality: str = "UNKNOWN"
    target_clearance: str = "UNKNOWN"
    stop_quality: str = "UNKNOWN"
    breakout_quality: str = "UNKNOWN"

    # Scoring.
    method_base_score: float = 0.0
    rr_score: float = 0.0
    confluence_score: float = 0.0
    entry_quality_score: float = 0.0
    risk_penalty: float = 0.0
    distance_penalty: float = 0.0
    final_score: float = 0.0

    reasons_ja: list[str] = field(default_factory=list)
    block_reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    used_in_decision: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entry_type": self.entry_type,
            "status": self.status,
            "side": self.side,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "rr": self.rr,
            "setup_invalidation_price": self.setup_invalidation_price,
            "trade_stop_price": self.trade_stop_price,
            "trigger_line_id": self.trigger_line_id,
            "trigger_line_price": self.trigger_line_price,
            "trigger_confirmed": self.trigger_confirmed,
            "retest_confirmed": self.retest_confirmed,
            "confirmation_candle": self.confirmation_candle,
            "htf_alignment": self.htf_alignment,
            "location_quality": self.location_quality,
            "target_clearance": self.target_clearance,
            "stop_quality": self.stop_quality,
            "breakout_quality": self.breakout_quality,
            "method_base_score": self.method_base_score,
            "rr_score": self.rr_score,
            "confluence_score": self.confluence_score,
            "entry_quality_score": self.entry_quality_score,
            "risk_penalty": self.risk_penalty,
            "distance_penalty": self.distance_penalty,
            "final_score": self.final_score,
            "reasons_ja": list(self.reasons_ja),
            "block_reasons": list(self.block_reasons),
            "cautions": list(self.cautions),
            "debug": dict(self.debug),
            "used_in_decision": self.used_in_decision,
        }


@dataclass(frozen=True)
class EntryMethodContext:
    symbol: str
    timeframe: str
    df_window: Any

    atr_value: float | None
    last_close: float | None
    current_ts: Any

    pattern_levels: dict[str, Any]
    wave_derived_lines: list[dict[str, Any]]
    breakout_quality_gate: dict[str, Any]
    fundamental_sidebar: dict[str, Any] | None

    support_resistance_v2: dict[str, Any]
    trendline_context: dict[str, Any]
    dow_structure_review: dict[str, Any] | None
    ma_context_review: dict[str, Any] | None
    candlestick_anatomy_review: dict[str, Any] | None

    entry_settings: dict[str, Any] = field(default_factory=dict)


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _score_rr(rr: float | None) -> float:
    if rr is None:
        return 0.0
    if rr >= 3.0:
        return 20.0
    if rr >= 2.0:
        return 15.0
    if rr >= 1.5:
        return 8.0
    return 0.0


def apply_common_entry_gates(
    candidate: EntryCandidate,
    ctx: EntryMethodContext,
) -> EntryCandidate:
    """Apply common hard gates to a candidate.

    The gate is intentionally strict. It only downgrades READY candidates.
    WAIT_* / WATCH / HOLD candidates pass through unchanged.
    """
    if candidate.status != "READY":
        return candidate

    blocks = list(candidate.block_reasons)
    cautions = list(candidate.cautions)
    reasons = list(candidate.reasons_ja)

    fs = ctx.fundamental_sidebar or {}
    if str(fs.get("event_risk_status") or "").upper() == "BLOCK":
        return replace(
            candidate,
            status="WAIT_EVENT_CLEAR",
            block_reasons=blocks + ["event_risk_block"],
            reasons_ja=reasons + [
                "イベント危険時間のため、新規エントリーは禁止です。"
            ],
        )

    if (
        candidate.entry_price is None
        or candidate.stop_price is None
        or candidate.target_price is None
    ):
        return replace(
            candidate,
            status="HOLD",
            block_reasons=blocks + ["entry_stop_target_missing"],
        )

    if candidate.rr is None:
        return replace(
            candidate,
            status="HOLD",
            block_reasons=blocks + ["rr_missing"],
        )

    min_rr = float(ctx.entry_settings.get("min_rr_default", 2.0))
    if candidate.rr < min_rr:
        return replace(
            candidate,
            status="HOLD",
            block_reasons=blocks + [
                f"rr_too_low:{candidate.rr:.2f}<{min_rr:.2f}"
            ],
        )

    if candidate.stop_quality == "BAD":
        return replace(
            candidate,
            status="HOLD",
            block_reasons=blocks + ["bad_stop_quality"],
        )

    if candidate.target_clearance == "BLOCKED":
        return replace(
            candidate,
            status="HOLD",
            block_reasons=blocks + ["target_blocked_by_near_sr"],
        )

    if candidate.breakout_quality == "BLOCK":
        return replace(
            candidate,
            status="HOLD",
            block_reasons=blocks + ["breakout_quality_block"],
        )

    e = candidate.entry_price
    s = candidate.stop_price
    t = candidate.target_price

    if candidate.side == "BUY" and not (s < e < t):
        return replace(
            candidate,
            status="HOLD",
            block_reasons=blocks + ["invalid_price_order_buy"],
        )

    if candidate.side == "SELL" and not (t < e < s):
        return replace(
            candidate,
            status="HOLD",
            block_reasons=blocks + ["invalid_price_order_sell"],
        )

    return candidate


def select_best_entry_candidate(
    candidates: list[EntryCandidate],
) -> EntryCandidate:
    """Select best candidate.

    READY candidates are selected by final_score, not simple entry_type
    priority. If no READY exists, select the most advanced waiting state.
    """
    ready = [
        c for c in candidates
        if c.status == "READY" and c.side in ("BUY", "SELL")
    ]

    if ready:
        return max(ready, key=lambda c: c.final_score)

    if not candidates:
        return EntryCandidate(
            schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
            entry_type="none",
            status="HOLD",
            side="NEUTRAL",
            entry_price=None,
            stop_price=None,
            target_price=None,
            rr=None,
            block_reasons=["no_entry_candidates"],
            reasons_ja=["エントリー候補がありません。"],
            used_in_decision=False,
        )

    status_order = {
        "WAIT_EVENT_CLEAR": 6,
        "WAIT_RETEST": 5,
        "WAIT_BREAKOUT": 4,
        "WAIT_TRIGGER": 3,
        "WATCH": 2,
        "HOLD": 1,
    }

    return max(
        candidates,
        key=lambda c: (status_order.get(c.status, 0), c.final_score),
    )


def candidate_from_entry_plan(
    entry_plan: dict[str, Any] | None,
    *,
    entry_type: str = "neckline_retest",
) -> EntryCandidate:
    """Convert existing entry_plan_v1 dict into EntryCandidate.

    This is the Phase I-2 compatibility bridge. It must preserve
    existing entry_plan semantics and must not change action gating.
    """
    ep = entry_plan or {}

    status = ep.get("entry_status") or "HOLD"
    if status not in (
        "READY",
        "WAIT_BREAKOUT",
        "WAIT_RETEST",
        "WAIT_TRIGGER",
        "WAIT_EVENT_CLEAR",
        "WATCH",
        "HOLD",
    ):
        status = "HOLD"

    side = ep.get("side") or "NEUTRAL"
    if side not in ("BUY", "SELL", "NEUTRAL"):
        side = "NEUTRAL"

    entry_price = _as_float(ep.get("entry_price"))
    stop_price = _as_float(ep.get("stop_price"))
    target_price = _as_float(
        ep.get("target_extended_price")
        if ep.get("target_extended_price") is not None
        else ep.get("target_price")
    )
    rr = _as_float(ep.get("rr"))

    method_base_score = 70.0 if entry_type == "neckline_retest" else 50.0
    rr_score = _score_rr(rr)

    confluence_score = 0.0
    entry_quality_score = 0.0
    risk_penalty = 0.0
    distance_penalty = 0.0

    if status == "READY":
        entry_quality_score += 10.0
    elif status in ("WAIT_RETEST", "WAIT_BREAKOUT"):
        entry_quality_score += 3.0

    final_score = (
        method_base_score
        + rr_score
        + confluence_score
        + entry_quality_score
        - risk_penalty
        - distance_penalty
    )

    reasons: list[str] = []
    if ep.get("reason_ja"):
        reasons.append(str(ep.get("reason_ja")))
    else:
        reasons.append(
            "既存 entry_plan から neckline_retest 候補を生成しました。"
        )

    return EntryCandidate(
        schema_version=ENTRY_CANDIDATE_SCHEMA_VERSION,
        entry_type=entry_type,
        status=status,  # type: ignore[arg-type]
        side=side,      # type: ignore[arg-type]
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        rr=rr,
        setup_invalidation_price=stop_price,
        trade_stop_price=stop_price,
        trigger_line_id=ep.get("trigger_line_id"),
        trigger_line_price=_as_float(ep.get("trigger_line_price")),
        trigger_confirmed=bool(ep.get("breakout_confirmed")),
        retest_confirmed=bool(ep.get("retest_confirmed")),
        confirmation_candle=ep.get("confirmation_candle") or None,
        htf_alignment="UNKNOWN",
        location_quality="UNKNOWN",
        target_clearance="UNKNOWN",
        stop_quality="UNKNOWN",
        breakout_quality="UNKNOWN",
        method_base_score=method_base_score,
        rr_score=rr_score,
        confluence_score=confluence_score,
        entry_quality_score=entry_quality_score,
        risk_penalty=risk_penalty,
        distance_penalty=distance_penalty,
        final_score=final_score,
        reasons_ja=reasons,
        block_reasons=list(ep.get("block_reasons") or []),
        cautions=list(ep.get("cautions") or []),
        debug={
            "source": "entry_plan_v1",
            "original_entry_status": ep.get("entry_status"),
            "downgraded_from_ready_by_event_risk": bool(
                ep.get("downgraded_from_ready_by_event_risk")
            ),
        },
        used_in_decision=True,
    )


def build_entry_candidates_from_existing_plan(
    *,
    entry_plan: dict[str, Any] | None,
    ctx: EntryMethodContext,
) -> list[EntryCandidate]:
    """Phase I-1/I-2 builder.

    For now, only the existing entry_plan is converted into a single
    neckline_retest candidate. Later phases will append
    direct_breakout_quality, pullback_buy, return_sell, range_bounce,
    false_breakout_reversal, etc.
    """
    raw = [
        candidate_from_entry_plan(
            entry_plan,
            entry_type="neckline_retest",
        )
    ]

    return [
        apply_common_entry_gates(c, ctx)
        for c in raw
    ]


def selected_candidate_to_entry_plan(
    selected: EntryCandidate,
    *,
    original_entry_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compatibility helper.

    Phase I must not break entry_plan_v1 schema. Start from the original
    entry_plan and only add candidate metadata fields. We deliberately do
    NOT overwrite the original entry_plan's entry_status / side /
    entry_price / stop_price / target_price / rr unless the original
    entry_plan was None or empty — the gating decision belongs to
    entry_plan_v1, not the candidate selector.
    """
    out = dict(original_entry_plan or {})

    # Only fill in missing core fields from the selected candidate.
    out.setdefault("entry_type", selected.entry_type)
    out.setdefault("entry_status", selected.status)
    out.setdefault("side", selected.side)
    if out.get("entry_price") is None:
        out["entry_price"] = selected.entry_price
    if out.get("stop_price") is None:
        out["stop_price"] = selected.stop_price
    if out.get("target_price") is None:
        out["target_price"] = selected.target_price
    if out.get("rr") is None:
        out["rr"] = selected.rr

    # Always surface the selector's identification + score so the
    # visual-audit panel and downstream consumers can show what was
    # picked and why.
    out["selected_entry_candidate_type"] = selected.entry_type
    out["selected_entry_candidate_score"] = selected.final_score
    return out


__all__ = [
    "ENTRY_CANDIDATE_SCHEMA_VERSION",
    "EntryCandidate",
    "EntryMethodContext",
    "apply_common_entry_gates",
    "select_best_entry_candidate",
    "candidate_from_entry_plan",
    "build_entry_candidates_from_existing_plan",
    "selected_candidate_to_entry_plan",
]
