"""Royal-road procedure checklist.

Phase I follow-up: turn the existing entry_plan_v1 + integrated panels
into an ordered, auditable "did we follow the royal-road procedure"
checklist. The list is observation/display-only — it never changes
the existing entry_plan READY condition or the final action.

Royal-road order (locked):

  1. environment           – イベント危険・マクロ・セッション
  2. dow_structure         – ダウ理論の方向
  3. support_resistance    – 重要水平線
  4. trendline_context     – トレンドライン / チャネル
  5. wave_pattern          – 波形認識 (DB / DT / HS / IHS / ...)
  6. wave_lines            – WNL / WSL / WTP
  7. breakout_confirmed    – WNL / trigger ブレイクの確認
  8. retest_confirmed      – リターンムーブの確認
  9. confirmation_candle   – ローソク足の確認 (P0必須)
 10. entry_price           – ENTRY
 11. stop_price            – STOP
 12. target_price          – TP
 13. rr_ok                 – RR ≥ min_rr
 14. event_clear           – イベント危険なし

`build_royal_road_procedure_checklist` always emits all 14 steps in
this order, so visual_audit can render them as a single table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PROCEDURE_SCHEMA_VERSION = "royal_road_procedure_checklist_v1"

ProcedureStatus = Literal["PASS", "WAIT", "BLOCK", "WARN", "UNKNOWN"]
ProcedureImportance = Literal["P0", "P1", "P2"]


@dataclass(frozen=True)
class RoyalRoadProcedureStep:
    key: str
    label_ja: str
    status: ProcedureStatus
    importance: ProcedureImportance
    used_in_decision: bool
    condition_ja: str
    result_ja: str
    evidence: dict[str, Any] = field(default_factory=dict)
    block_reasons: list[str] = field(default_factory=list)
    wait_reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label_ja": self.label_ja,
            "status": self.status,
            "importance": self.importance,
            "used_in_decision": self.used_in_decision,
            "condition_ja": self.condition_ja,
            "result_ja": self.result_ja,
            "evidence": dict(self.evidence),
            "block_reasons": list(self.block_reasons),
            "wait_reasons": list(self.wait_reasons),
            "cautions": list(self.cautions),
        }


@dataclass(frozen=True)
class RoyalRoadProcedureChecklist:
    schema_version: str
    side: str
    final_status: str
    ready: bool
    summary_ja: str
    steps: list[RoyalRoadProcedureStep]
    p0_pass: bool
    p0_missing_or_blocked: list[str] = field(default_factory=list)
    wait_reasons: list[str] = field(default_factory=list)
    block_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "side": self.side,
            "final_status": self.final_status,
            "ready": self.ready,
            "summary_ja": self.summary_ja,
            "steps": [s.to_dict() for s in self.steps],
            "p0_pass": self.p0_pass,
            "p0_missing_or_blocked": list(self.p0_missing_or_blocked),
            "wait_reasons": list(self.wait_reasons),
            "block_reasons": list(self.block_reasons),
        }


# ─────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────


def _summary_for(final_status: str, *, event_block: bool) -> str:
    if event_block and final_status != "READY":
        return (
            "テクニカル条件はありますが、イベント危険のため待機です。"
        )
    if final_status == "READY":
        return "王道手順のP0条件が揃っているためREADYです。"
    if final_status == "WAIT_BREAKOUT":
        return (
            "王道手順上、WNL / trigger line のブレイク待ちです。"
        )
    if final_status == "WAIT_RETEST":
        return (
            "王道手順上、ブレイク後のリターンムーブ確認待ちです。"
        )
    if final_status == "WAIT_EVENT_CLEAR":
        return (
            "テクニカル条件はありますが、イベント危険のため待機です。"
        )
    if final_status == "WAIT_TRIGGER":
        return "王道手順上、下位足のトリガー待ちです。"
    if final_status == "WATCH":
        return "王道手順上、まだ監視段階です。"
    return "王道手順の必須条件が不足しているためHOLDです。"


def _step(
    key: str,
    label_ja: str,
    status: ProcedureStatus,
    importance: ProcedureImportance,
    *,
    condition_ja: str,
    result_ja: str,
    evidence: dict[str, Any] | None = None,
    block_reasons: list[str] | None = None,
    wait_reasons: list[str] | None = None,
    cautions: list[str] | None = None,
    used_in_decision: bool = True,
) -> RoyalRoadProcedureStep:
    return RoyalRoadProcedureStep(
        key=key,
        label_ja=label_ja,
        status=status,
        importance=importance,
        used_in_decision=used_in_decision,
        condition_ja=condition_ja,
        result_ja=result_ja,
        evidence=dict(evidence or {}),
        block_reasons=list(block_reasons or []),
        wait_reasons=list(wait_reasons or []),
        cautions=list(cautions or []),
    )


def _has_value(v: Any) -> bool:
    return v is not None


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def build_royal_road_procedure_checklist(
    *,
    entry_plan: dict[str, Any] | None,
    pattern_levels: dict[str, Any] | None,
    wave_derived_lines: list[dict[str, Any]] | None,
    breakout_quality_gate: dict[str, Any] | None,
    fundamental_sidebar: dict[str, Any] | None,
    support_resistance_v2: dict[str, Any] | None,
    trendline_context: dict[str, Any] | None,
    dow_structure_review: dict[str, Any] | None,
    candlestick_anatomy_review: dict[str, Any] | None,
    min_rr: float = 2.0,
) -> RoyalRoadProcedureChecklist:
    """Build the ordered royal-road procedure checklist.

    Observation-only — the checklist does not modify entry_plan or
    the final action.
    """
    ep = entry_plan or {}
    pl = pattern_levels or {}
    wdl = wave_derived_lines or []
    bq = breakout_quality_gate or {}
    fs = fundamental_sidebar or {}
    sr = support_resistance_v2 or {}
    tl = trendline_context or {}
    dow = dow_structure_review or {}
    candle = candlestick_anatomy_review or {}

    final_status = str(ep.get("entry_status") or "HOLD")
    side = str(ep.get("side") or "NEUTRAL")
    event_status = str(fs.get("event_risk_status") or "").upper()
    # When fundamental_sidebar lacks an explicit event_risk_status
    # (e.g. synthetic fixtures with no calendar feed report it as
    # empty or "UNKNOWN"), fall back to the entry_plan_v1 downgrade
    # signal: WAIT_EVENT_CLEAR ⇒ BLOCK, otherwise CLEAR. The
    # event-risk gate is what actually downgrades READY →
    # WAIT_EVENT_CLEAR upstream, so this reflects reality without
    # falsely keeping an actionable READY page in UNKNOWN limbo.
    if not event_status or event_status == "UNKNOWN":
        if final_status == "WAIT_EVENT_CLEAR":
            event_status = "BLOCK"
        elif final_status in (
            "READY",
            "WAIT_BREAKOUT", "WAIT_RETEST",
            "WAIT_TRIGGER", "WATCH",
        ):
            event_status = "CLEAR"
        # HOLD / unknown → leave UNKNOWN
    event_block = event_status == "BLOCK"

    steps: list[RoyalRoadProcedureStep] = []

    # ── 1. environment ──────────────────────────────────────────
    if event_status == "BLOCK":
        env_status: ProcedureStatus = "BLOCK"
        env_result = "イベント危険時間帯です。"
        env_blocks = ["event_risk_block"]
    elif event_status == "WARNING":
        env_status = "WARN"
        env_result = "イベント間近のため注意。"
        env_blocks = []
    elif event_status in ("CLEAR", "PASS", "OK"):
        env_status = "PASS"
        env_result = "イベント危険なし。"
        env_blocks = []
    else:
        env_status = "UNKNOWN"
        env_result = "fundamental_sidebar が未提供です。"
        env_blocks = []
    steps.append(_step(
        "environment", "環境認識", env_status, "P1",
        condition_ja=(
            "重大イベント・マクロが新規エントリーを禁止していないこと。"
        ),
        result_ja=env_result,
        evidence={
            "event_risk_status": event_status or "UNKNOWN",
            "blocking_events_count": len(fs.get("blocking_events") or []),
            "warning_events_count": len(fs.get("warning_events") or []),
        },
        block_reasons=env_blocks,
    ))

    # ── 2. dow_structure ────────────────────────────────────────
    dow_trend = str(
        (dow.get("trend") or dow.get("dow_trend") or "")
    ).upper()
    if dow_trend in ("UP", "DOWN"):
        dow_status: ProcedureStatus = "PASS"
        dow_result = f"{dow_trend} のダウ構造を確認。"
        dow_cautions: list[str] = []
    elif dow_trend in ("RANGE", "MIXED", "SIDEWAYS"):
        dow_status = "WARN"
        dow_result = f"{dow_trend} — 方向性が弱い可能性。"
        dow_cautions = ["dow_trend_unclear"]
    elif dow_trend == "":
        dow_status = "UNKNOWN"
        dow_result = "dow_structure_review が未提供です。"
        dow_cautions = []
    else:
        dow_status = "UNKNOWN"
        dow_result = f"dow_trend={dow_trend or '—'}"
        dow_cautions = []
    steps.append(_step(
        "dow_structure", "ダウ理論", dow_status, "P1",
        condition_ja=(
            "ダウ理論で UP / DOWN の方向性が見えていること。"
        ),
        result_ja=dow_result,
        evidence={"dow_trend": dow_trend or "UNKNOWN"},
        cautions=dow_cautions,
    ))

    # ── 3. support_resistance ───────────────────────────────────
    sr_selected = list(
        sr.get("selected_level_zones_top5")
        or sr.get("selected_level_zones")
        or []
    )
    sr_rejected = list(sr.get("rejected_level_zones") or [])
    if len(sr_selected) >= 1:
        sr_status: ProcedureStatus = "PASS"
        sr_result = f"重要水平線 {len(sr_selected)} 本を採用。"
        sr_cautions = []
    else:
        sr_status = "WARN"
        sr_result = "サポレジ検出: 0本 — 線引き根拠が弱い。"
        sr_cautions = ["sr_zero_selected"]
    steps.append(_step(
        "support_resistance", "重要水平線", sr_status, "P1",
        condition_ja="サポート/レジスタンスが認識されていること。",
        result_ja=sr_result,
        evidence={
            "selected_count": len(sr_selected),
            "rejected_count": len(sr_rejected),
        },
        cautions=sr_cautions,
    ))

    # ── 4. trendline_context ────────────────────────────────────
    tl_selected = list(
        tl.get("selected_trendlines_top3")
        or tl.get("selected_trendlines")
        or []
    )
    tl_rejected = list(tl.get("rejected_trendlines") or [])
    if len(tl_selected) >= 1:
        tl_status: ProcedureStatus = "PASS"
        tl_result = f"トレンドライン {len(tl_selected)} 本を採用。"
        tl_cautions = []
    else:
        tl_status = "WARN"
        tl_result = "トレンドライン検出: 0本 — 線引き根拠が弱い。"
        tl_cautions = ["trendline_zero_selected"]
    steps.append(_step(
        "trendline_context", "トレンドライン", tl_status, "P1",
        condition_ja=(
            "トレンドライン / チャネルが認識されていること。"
        ),
        result_ja=tl_result,
        evidence={
            "selected_count": len(tl_selected),
            "rejected_count": len(tl_rejected),
        },
        cautions=tl_cautions,
    ))

    # ── 5. wave_pattern ─────────────────────────────────────────
    pattern_kind = pl.get("pattern_kind") or pl.get("kind") or ""
    pl_available = bool(pl.get("available"))
    if pl_available and pattern_kind:
        wp_status: ProcedureStatus = "PASS"
        wp_result = f"波形 {pattern_kind} を認識。"
        wp_blocks: list[str] = []
    else:
        wp_status = "BLOCK"
        wp_result = "波形が認識できていません。"
        wp_blocks = ["wave_pattern_missing"]
    steps.append(_step(
        "wave_pattern", "波形認識", wp_status, "P0",
        condition_ja="ダブルボトム / ダブルトップ等の王道波形を認識。",
        result_ja=wp_result,
        evidence={
            "available": pl_available,
            "pattern_kind": pattern_kind or "—",
        },
        block_reasons=wp_blocks,
    ))

    # ── 6. wave_lines (WNL / WSL / WTP) ────────────────────────
    line_roles = {
        str((line or {}).get("role") or "")
        for line in wdl
    }
    has_wnl = "entry_confirmation_line" in line_roles
    has_wsl = "stop_candidate" in line_roles
    has_wtp = "target_candidate" in line_roles
    missing_w = []
    if not has_wnl:
        missing_w.append("WNL")
    if not has_wsl:
        missing_w.append("WSL")
    if not has_wtp:
        missing_w.append("WTP")
    if not missing_w:
        wl_status: ProcedureStatus = "PASS"
        wl_result = "WNL / WSL / WTP 全て揃っています。"
        wl_blocks: list[str] = []
    else:
        wl_status = "BLOCK"
        wl_result = f"不足: {', '.join(missing_w)}"
        wl_blocks = [f"missing_{x.lower()}" for x in missing_w]
    steps.append(_step(
        "wave_lines", "Wライン (WNL / WSL / WTP)", wl_status, "P0",
        condition_ja=(
            "波形からネックライン (WNL)、損切り (WSL)、利確 (WTP) "
            "が引けていること。"
        ),
        result_ja=wl_result,
        evidence={
            "has_wnl": has_wnl, "has_wsl": has_wsl, "has_wtp": has_wtp,
        },
        block_reasons=wl_blocks,
    ))

    # ── 7. breakout_confirmed ───────────────────────────────────
    breakout_confirmed = bool(ep.get("breakout_confirmed"))
    if breakout_confirmed:
        br_status: ProcedureStatus = "PASS"
        br_result = "WNL / trigger を抜けました。"
        br_waits: list[str] = []
    elif final_status == "WAIT_BREAKOUT":
        br_status = "WAIT"
        br_result = "WNL / trigger 未突破。突破足を待っています。"
        br_waits = ["wnl_not_broken"]
    else:
        br_status = "WAIT"
        br_result = "ブレイク未確認です。"
        br_waits = ["breakout_not_confirmed"]
    steps.append(_step(
        "breakout_confirmed", "ブレイク確認", br_status, "P0",
        condition_ja="WNL / trigger を実体ブレイクで抜けたこと。",
        result_ja=br_result,
        evidence={
            "breakout_confirmed": breakout_confirmed,
            "trigger_line_id": ep.get("trigger_line_id") or "—",
            "trigger_line_price": _as_float(ep.get("trigger_line_price")),
        },
        wait_reasons=br_waits,
    ))

    # ── 8. retest_confirmed ────────────────────────────────────
    retest_confirmed = bool(ep.get("retest_confirmed"))
    if retest_confirmed:
        rt_status: ProcedureStatus = "PASS"
        rt_result = "リターンムーブを確認しました。"
        rt_waits: list[str] = []
    elif breakout_confirmed:
        rt_status = "WAIT"
        rt_result = "ブレイク後のリターンムーブを待っています。"
        rt_waits = ["awaiting_retest_confirmation"]
    else:
        # まだブレイク前なら、リターンムーブも未確認 (WAIT が自然)
        rt_status = "WAIT"
        rt_result = "ブレイク前のためリターンムーブも未確認。"
        rt_waits = ["awaiting_breakout_first"]
    steps.append(_step(
        "retest_confirmed", "リターンムーブ確認", rt_status, "P0",
        condition_ja=(
            "WNL ブレイク後、戻り足が WNL を支持/抵抗として再認識すること。"
        ),
        result_ja=rt_result,
        evidence={
            "retest_confirmed": retest_confirmed,
            "breakout_confirmed": breakout_confirmed,
        },
        wait_reasons=rt_waits,
    ))

    # ── 9. confirmation_candle ─────────────────────────────────
    confirmation_candle = ep.get("confirmation_candle") or ""
    if confirmation_candle:
        cc_status: ProcedureStatus = "PASS"
        cc_result = f"確認足: {confirmation_candle}"
        cc_waits: list[str] = []
    elif retest_confirmed:
        cc_status = "WAIT"
        cc_result = (
            "リターンムーブ後の確認ローソク (例: pinbar / engulfing) "
            "を待っています。"
        )
        cc_waits = ["awaiting_confirmation_candle"]
    else:
        cc_status = "WAIT"
        cc_result = "リターンムーブ確認後にローソク確認します。"
        cc_waits = ["awaiting_retest_first"]
    steps.append(_step(
        "confirmation_candle", "ローソク足確認", cc_status, "P0",
        condition_ja=(
            "リターンムーブ後、方向に沿った確認ローソクが出ていること。"
            "通常の neckline_retest READY では必須条件。"
        ),
        result_ja=cc_result,
        evidence={
            "confirmation_candle": confirmation_candle or "—",
            "candlestick_anatomy_available": bool(candle),
        },
        wait_reasons=cc_waits,
    ))

    # ── 10. entry_price ─────────────────────────────────────────
    entry_price = _as_float(ep.get("entry_price"))
    if entry_price is not None:
        en_status: ProcedureStatus = "PASS"
        en_result = f"ENTRY = {entry_price:.5f}"
        en_blocks: list[str] = []
        en_waits: list[str] = []
    elif final_status in (
        "WAIT_BREAKOUT", "WAIT_RETEST", "WAIT_EVENT_CLEAR",
        "WAIT_TRIGGER", "WATCH",
    ):
        en_status = "WAIT"
        en_result = "ENTRY 価格は確定待ちです。"
        en_blocks = []
        en_waits = ["entry_price_pending"]
    else:
        en_status = "BLOCK"
        en_result = "ENTRY 価格がありません。"
        en_blocks = ["entry_price_missing"]
        en_waits = []
    steps.append(_step(
        "entry_price", "ENTRY 価格", en_status, "P0",
        condition_ja="エントリー価格が決定していること。",
        result_ja=en_result,
        evidence={"entry_price": entry_price},
        block_reasons=en_blocks,
        wait_reasons=en_waits,
    ))

    # ── 11. stop_price ─────────────────────────────────────────
    stop_price = _as_float(ep.get("stop_price"))
    if stop_price is not None:
        sp_status: ProcedureStatus = "PASS"
        sp_result = f"STOP = {stop_price:.5f}"
        sp_blocks: list[str] = []
    else:
        sp_status = "BLOCK"
        sp_result = "STOP がありません。"
        sp_blocks = ["stop_price_missing"]
    steps.append(_step(
        "stop_price", "STOP 価格", sp_status, "P0",
        condition_ja=(
            "STOP が決定していること (波形外 or ATR ベース)。"
        ),
        result_ja=sp_result,
        evidence={"stop_price": stop_price},
        block_reasons=sp_blocks,
    ))

    # ── 12. target_price ───────────────────────────────────────
    target_price = _as_float(
        ep.get("target_extended_price")
        if ep.get("target_extended_price") is not None
        else ep.get("target_price")
    )
    if target_price is not None:
        tp_status: ProcedureStatus = "PASS"
        tp_result = f"TP = {target_price:.5f}"
        tp_blocks: list[str] = []
    else:
        tp_status = "BLOCK"
        tp_result = "TP がありません。"
        tp_blocks = ["target_price_missing"]
    steps.append(_step(
        "target_price", "TP 価格", tp_status, "P0",
        condition_ja=(
            "TP が決定していること (波形目標 or extended target)。"
        ),
        result_ja=tp_result,
        evidence={"target_price": target_price},
        block_reasons=tp_blocks,
    ))

    # ── 13. rr_ok ──────────────────────────────────────────────
    rr = _as_float(ep.get("rr"))
    if rr is None:
        rr_status: ProcedureStatus = "BLOCK"
        rr_result = "RR が計算されていません。"
        rr_blocks = ["rr_missing"]
    elif rr >= float(min_rr):
        rr_status = "PASS"
        rr_result = f"RR = {rr:.2f} ≥ {float(min_rr):.2f}"
        rr_blocks = []
    else:
        rr_status = "BLOCK"
        rr_result = f"RR = {rr:.2f} < {float(min_rr):.2f}"
        rr_blocks = [f"rr_too_low:{rr:.2f}<{float(min_rr):.2f}"]
    steps.append(_step(
        "rr_ok", "RR (リスクリワード)", rr_status, "P0",
        condition_ja=(
            f"RR が {float(min_rr):.2f} 以上であること。"
        ),
        result_ja=rr_result,
        evidence={"rr": rr, "min_rr": float(min_rr)},
        block_reasons=rr_blocks,
    ))

    # ── 14. event_clear ────────────────────────────────────────
    if event_status == "BLOCK":
        ec_status: ProcedureStatus = "BLOCK"
        ec_result = "重大イベント時間帯のためBLOCK。"
        ec_blocks = ["event_risk_block"]
        ec_cautions: list[str] = []
    elif event_status == "WARNING":
        ec_status = "WARN"
        ec_result = "イベント直前のため注意。"
        ec_blocks = []
        ec_cautions = ["event_risk_warning"]
    elif event_status in ("CLEAR", "PASS", "OK"):
        ec_status = "PASS"
        ec_result = "イベント危険なし。"
        ec_blocks = []
        ec_cautions = []
    else:
        ec_status = "UNKNOWN"
        ec_result = "イベント状態が未提供です。"
        ec_blocks = []
        ec_cautions = []
    steps.append(_step(
        "event_clear", "イベント確認", ec_status, "P0",
        condition_ja=(
            "重大イベントの直前/最中ではないこと。"
        ),
        result_ja=ec_result,
        evidence={"event_risk_status": event_status or "UNKNOWN"},
        block_reasons=ec_blocks,
        cautions=ec_cautions,
    ))

    # ── Aggregate P0 / waits / blocks ────────────────────────────
    p0_missing_or_blocked: list[str] = []
    aggregate_waits: list[str] = []
    aggregate_blocks: list[str] = []

    for s in steps:
        if s.importance == "P0":
            # Any P0 step that is not PASS / WARN means the royal-road
            # procedure has not yet completed. The user's spec
            # explicitly shows WAIT_RETEST → 王道P0: NOT PASS, so
            # WAIT P0 steps also count as missing.
            if s.status in ("BLOCK", "UNKNOWN", "WAIT"):
                if s.key not in p0_missing_or_blocked:
                    p0_missing_or_blocked.append(s.key)
        if s.status == "WAIT":
            for w in (s.wait_reasons or []):
                if w not in aggregate_waits:
                    aggregate_waits.append(w)
        if s.status == "BLOCK":
            for b in (s.block_reasons or []):
                if b not in aggregate_blocks:
                    aggregate_blocks.append(b)

    p0_pass = (len(p0_missing_or_blocked) == 0) and not event_block
    # "ready" is the strictest: every P0 must PASS and there must be
    # no WAIT step blocking the path (WAIT_RETEST etc. all have at
    # least one P0 WAIT).
    ready = p0_pass and not any(s.status == "WAIT" for s in steps)

    summary_ja = _summary_for(final_status, event_block=event_block)

    return RoyalRoadProcedureChecklist(
        schema_version=PROCEDURE_SCHEMA_VERSION,
        side=side,
        final_status=final_status,
        ready=ready,
        summary_ja=summary_ja,
        steps=steps,
        p0_pass=p0_pass,
        p0_missing_or_blocked=p0_missing_or_blocked,
        wait_reasons=aggregate_waits,
        block_reasons=aggregate_blocks,
    )


__all__ = [
    "PROCEDURE_SCHEMA_VERSION",
    "RoyalRoadProcedureStep",
    "RoyalRoadProcedureChecklist",
    "build_royal_road_procedure_checklist",
]
