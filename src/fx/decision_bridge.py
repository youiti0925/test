"""Decision bridge (observation-only).

The audit payload contains a lot of information now (Masterclass
panels, source-pack panels, wave-derived lines, fibonacci lines,
candlestick anatomy, ...). The user's experience showed that this
made it hard to tell which of those pieces actually drove the final
BUY/SELL/HOLD action and which are merely audit displays.

This module classifies every category in the payload into one of:

  USED            - feeds the final action directly (royal_road_v2
                    core, block_reasons, structure_stop_plan)
  PARTIAL         - has some indirect connection
  AUDIT_ONLY      - displayed for the human reviewer but does NOT
                    influence the final action
  NOT_CONNECTED   - referenced in the source pack but real data is
                    not yet wired (macro / position sizing / news)
  UNKNOWN         - state cannot be determined

The bridge itself is observation-only and never modifies any
decision.
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "decision_bridge_v1"


# Status enum constants
USED: Final[str] = "USED"
PARTIAL: Final[str] = "PARTIAL"
AUDIT_ONLY: Final[str] = "AUDIT_ONLY"
NOT_CONNECTED: Final[str] = "NOT_CONNECTED"
UNKNOWN: Final[str] = "UNKNOWN"


STATUS_LABEL_JA: Final[dict[str, str]] = {
    USED:           "判断に使用",
    PARTIAL:        "一部使用",
    AUDIT_ONLY:     "表示のみ",
    NOT_CONNECTED:  "未接続",
    UNKNOWN:        "不明",
}


def _entry(
    *,
    category: str,
    label_ja: str,
    status: str,
    reason_ja: str,
    what_to_check_ja: str = "",
) -> dict:
    return {
        "category": category,
        "label_ja": label_ja,
        "status": status,
        "status_label_ja": STATUS_LABEL_JA.get(status, status),
        "reason_ja": reason_ja,
        "what_to_check_ja": what_to_check_ja,
    }


def build_decision_bridge(payload: dict | None) -> dict:
    """Build the decision_bridge_v1 dict from a visual_audit payload.

    Returns `available=False` when payload is None or has no
    `royal_road_decision_v2` slice.
    """
    if not payload:
        return _empty_panel("missing_payload")
    v2 = payload.get("royal_road_decision_v2")
    if not v2:
        return _empty_panel("missing_v2_slice")

    action = v2.get("action") or "?"
    profile = v2.get("profile") or "royal_road_decision_v2"
    block_reasons = v2.get("block_reasons") or []
    stop_plan = v2.get("structure_stop_plan") or {}
    macro_align = v2.get("macro_alignment") or {}
    macro_score = macro_align.get("macro_score")

    # ── USED / PARTIAL ─────────────────────────────────────────
    used: list[dict] = []
    used.append(_entry(
        category="royal_road_v2_core",
        label_ja="王道v2本体の判断",
        status=USED,
        reason_ja=(
            f"最終 action ({action}) は {profile} の既存ロジックから"
            "出ています。"
        ),
        what_to_check_ja=(
            "現行の royal_road_decision_v2 の判定基準を変えていません。"
            "audit 表示は判断に影響しません。"
        ),
    ))
    if stop_plan.get("stop_price") is not None:
        used.append(_entry(
            category="stop_plan",
            label_ja="損切り / RR (structure_stop_plan)",
            status=PARTIAL,
            reason_ja=(
                "v2 の stop_plan / RR 判定は最終判断に関係します。"
                f"chosen_mode={stop_plan.get('chosen_mode')} "
                f"outcome={stop_plan.get('outcome')}."
            ),
            what_to_check_ja=(
                "結論カードの structure_stop / atr_stop / take_profit / "
                "RR が表示されているか確認してください。"
            ),
        ))
    if block_reasons:
        used.append(_entry(
            category="block_reasons",
            label_ja="ブロック理由 (block_reasons)",
            status=USED if action == "HOLD" else PARTIAL,
            reason_ja=(
                f"v2 が {len(block_reasons)} 件のブロック理由で判定。"
                f"先頭: {block_reasons[0][:60]}"
                if block_reasons[0] else
                f"v2 が {len(block_reasons)} 件のブロック理由で判定。"
            ),
            what_to_check_ja=(
                "block_reasons 一覧を見て、HOLD の根拠が納得できるか"
                "確認してください。"
            ),
        ))
    sr_v2 = v2.get("support_resistance_v2") or {}
    if (
        sr_v2.get("selected_level_zones_top5")
        or sr_v2.get("near_strong_support")
        or sr_v2.get("near_strong_resistance")
    ):
        used.append(_entry(
            category="support_resistance_v2",
            label_ja="サポレジ (support_resistance_v2)",
            status=PARTIAL,
            reason_ja=(
                "v2 の SR 判定は evidence_axes を経由して判断に影響します。"
            ),
            what_to_check_ja=(
                "緑/赤の SR 帯が現在価格と整合しているか確認してください。"
            ),
        ))

    # ── AUDIT_ONLY ─────────────────────────────────────────────
    audit_only: list[dict] = []
    if payload.get("wave_shape_review"):
        audit_only.append(_entry(
            category="wave_shape_review",
            label_ja="波形認識 (waveform shape review)",
            status=AUDIT_ONLY,
            reason_ja=(
                "波形候補は表示されていますが、まだ BUY / SELL / HOLD の"
                "最終判断には使っていません。"
            ),
            what_to_check_ja=(
                "チャート上の波形候補や W ラインが、人間の見方と合って"
                "いるか確認してください。"
            ),
        ))
    if payload.get("wave_derived_lines"):
        audit_only.append(_entry(
            category="wave_derived_lines",
            label_ja="波形由来ライン (WSL/WTP/WNL/WUP/WLOW/WBR)",
            status=AUDIT_ONLY,
            reason_ja=(
                "波形由来の参考線は描画されていますが、"
                "最終判断には未使用です。"
            ),
            what_to_check_ja=(
                "WNL / WSL / WTP / WUP / WLOW / WBR の位置が自然か、"
                "サポレジと重なっているか確認してください。"
            ),
        ))

    masterclass_dict = payload.get("masterclass_panels") or {}
    panels = masterclass_dict.get("panels") or {}
    if masterclass_dict.get("available"):
        audit_only.append(_entry(
            category="masterclass_panels",
            label_ja="Masterclass 監査パネル (19機能)",
            status=AUDIT_ONLY,
            reason_ja=(
                "資料由来のチェックは表示のみです。"
                "現時点では最終 action には未接続です。"
            ),
            what_to_check_ja=(
                "Tier 1 重要サマリ / Tier 2 王道チェック (13軸) で "
                "PASS / WARN / BLOCK を確認してください。"
            ),
        ))
        # Pull out Fibonacci specifically — most user-visible
        fib = panels.get("fibonacci_context_review") or {}
        if fib.get("available"):
            audit_only.append(_entry(
                category="fibonacci_context_review",
                label_ja="フィボナッチ (fibonacci_context_review)",
                status=AUDIT_ONLY,
                reason_ja=(
                    "フィボ線 (WFIB382 / 500 / 618 / 1272 / 1618) は"
                    "表示されていますが、まだ最終判断には使っていません。"
                ),
                what_to_check_ja=(
                    "現在価格が 38.2 / 50 / 61.8% 付近で反応しているか、"
                    "サポート / ローソク足反発と合流しているか確認してください。"
                ),
            ))
        # Pull out Daily Roadmap separately
        roadmap = panels.get("daily_roadmap_review") or {}
        if roadmap.get("available"):
            audit_only.append(_entry(
                category="daily_roadmap_review",
                label_ja="Daily 10k FX Roadmap",
                status=AUDIT_ONLY,
                reason_ja=(
                    "運用チェックリストは表示中です。最終 action には未接続。"
                ),
                what_to_check_ja=(
                    "未接続 (UNKNOWN) 項目はポジションサイジング / "
                    "経済指標カレンダー / ジャーナル。手動で確認してください。"
                ),
            ))

    # ── NOT_CONNECTED ──────────────────────────────────────────
    not_connected: list[dict] = []
    if macro_score is None or float(macro_score or 0.0) == 0.0:
        not_connected.append(_entry(
            category="macro_real_data",
            label_ja="ファンダ / マクロ実データ",
            status=NOT_CONNECTED,
            reason_ja=(
                "USDJPY briefing や DXY / 米金利 / VIX などは資料上重要"
                "ですが、実データには未接続です。macro_score は 0.0 NEUTRAL。"
            ),
            what_to_check_ja=(
                "別チャートで DXY / 米金利 / VIX などを手動で確認して"
                "ください。"
            ),
        ))
    not_connected.append(_entry(
        category="position_sizing",
        label_ja="ポジションサイズ計算",
        status=NOT_CONNECTED,
        reason_ja=(
            "Daily Roadmap 上は重要ですが、ポジションサイジングは未接続です。"
            "口座残高 × 1〜2% を手動で計算してください。"
        ),
    ))
    not_connected.append(_entry(
        category="news_calendar",
        label_ja="経済指標カレンダー",
        status=NOT_CONNECTED,
        reason_ja=(
            "news / event_overlay は未接続です。Forex Factory 等で当日"
            "の重要指標 (CPI / NFP / FOMC など) を手動で確認してください。"
        ),
    ))
    if masterclass_dict.get("available"):
        sb = (panels or {}).get("symbol_macro_briefing_review") or {}
        if sb.get("unavailable_reason") == "macro_briefing_data_missing":
            not_connected.append(_entry(
                category="symbol_macro_briefing",
                label_ja="通貨ペア固有ファンダ実データ",
                status=NOT_CONNECTED,
                reason_ja=(
                    "Tactical Briefing 由来のドライバーリスト (US_yield / "
                    "DXY / JPY_risk_sentiment 等) は表示中ですが、"
                    "実データは未接続です。"
                ),
            ))
        plt_panel = panels.get("parent_bar_lower_tf_anatomy") or {}
        if plt_panel.get("unavailable_reason") in (
            "lower_tf_missing", "too_few_lower_tf_bars",
        ):
            not_connected.append(_entry(
                category="lower_tf_anatomy",
                label_ja="下位足解剖 (lower-TF inside parent bar)",
                status=NOT_CONNECTED,
                reason_ja=(
                    "下位足 (5分 / 15分 等) の DataFrame が visual_audit "
                    "に未接続です。"
                ),
            ))

    # ── Action-specific message ─────────────────────────────────
    if action == "HOLD":
        action_message_ja = (
            "このチャートは入れそうに見えるかもしれません。"
            "ただし最終判断は HOLD です。\n\n"
            "理由:\n"
            "- 王道v2本体で必要条件が不足\n"
            "- W ライン / フィボ / Masterclass パネルは表示のみで、"
            "まだ最終判断には使っていない\n"
            "- ファンダ / position sizing / news は未接続"
        )
    elif action in ("BUY", "SELL"):
        action_message_ja = (
            f"最終判断は {action} です。\n\n"
            "ただし、波形・フィボ・Masterclass パネルは現時点では補助監査"
            "であり、最終判断に直接使っていません。\n"
            "チャート上でそれらが最終判断と同方向か、矛盾していないかを"
            "確認してください。"
        )
    else:
        action_message_ja = f"最終判断: {action}"

    summary_ja = (
        f"最終判断は {action} です。"
        "ただし波形・フィボ・Masterclass パネルは現時点では監査表示のみで、"
        "直接の売買判断には使っていません。"
    )
    plain_answer_ja = (
        "この画面の多くは『判断に使った根拠』ではなく『人間が確認するための"
        "監査情報』です。最終結果と完全にはまだ結びついていません。"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "final_action": action,
        "final_action_source": profile,
        "summary_ja": summary_ja,
        "action_message_ja": action_message_ja,
        "plain_answer_ja": plain_answer_ja,
        "used_for_final_decision": used,
        "audit_only_references": audit_only,
        "unconnected_or_missing": not_connected,
    }


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
        "final_action": None,
        "used_for_final_decision": [],
        "audit_only_references": [],
        "unconnected_or_missing": [],
    }


__all__ = [
    "SCHEMA_VERSION",
    "USED",
    "PARTIAL",
    "AUDIT_ONLY",
    "NOT_CONNECTED",
    "UNKNOWN",
    "STATUS_LABEL_JA",
    "build_decision_bridge",
]
