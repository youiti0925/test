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


def _axis_entry(
    *,
    category: str,
    label_ja: str,
    axis: dict,
    use_block_as_used: bool,
) -> dict:
    """Render one IntegratedEvidenceAxis as a bridge entry.

    `use_block_as_used=True` means BLOCK status is reported as USED
    (the axis actively forced HOLD — which IS using it). This is
    appropriate for required P0 axes. P1/P2 axes use
    `use_block_as_used=False` so a BLOCK still reports as PARTIAL.
    """
    ax_status = (axis.get("status") or "").upper()
    if ax_status == "PASS":
        bridge_status = USED
    elif ax_status == "WARN":
        bridge_status = PARTIAL
    elif ax_status == "BLOCK":
        bridge_status = USED if use_block_as_used else PARTIAL
    else:
        bridge_status = PARTIAL
    side = (axis.get("side") or "NEUTRAL")
    strength = float(axis.get("strength") or 0.0)
    reason_ja = (
        f"axis={axis.get('axis')} side={side} "
        f"status={ax_status} strength={strength:.2f}.\n"
        + (axis.get("reason_ja") or "")
    )
    return _entry(
        category=category,
        label_ja=label_ja,
        status=bridge_status,
        reason_ja=reason_ja,
        what_to_check_ja=(
            f"raw panel そのものではなく、ここから作った evidence axis "
            f"({axis.get('axis')}) が判断に使われています。"
            "axis の status と reason_ja を確認してください。"
        ),
    )


def build_decision_bridge(payload: dict | None) -> dict:
    """Build the decision_bridge_v1 dict from a visual_audit payload.

    Returns `available=False` when payload is None or has no
    `royal_road_decision_v2` slice.

    When the v2 slice represents the integrated profile (i.e.
    `profile == "royal_road_decision_v2_integrated"` OR
    `integrated_decision` is present), wave / W-lines / fib /
    Masterclass / candlestick / dow are reclassified from AUDIT_ONLY
    to USED or PARTIAL based on the integrated_decision axis
    statuses. Otherwise the legacy v2 classification is used.
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

    # Integrated profile detection
    integrated = v2.get("integrated_decision") or {}
    is_integrated = bool(
        profile == "royal_road_decision_v2_integrated"
        or integrated.get("schema_version") == "royal_road_integrated_decision_v1"
    )
    integrated_mode = integrated.get("mode") if is_integrated else None

    # ── USED / PARTIAL ─────────────────────────────────────────
    used: list[dict] = []

    # Pre-extract integrated axis statuses if available — these tell us
    # exactly which audit module actually drove the action.
    axis_status_by_source: dict[str, str] = {}
    axes_by_axis: dict[str, dict] = {}
    if is_integrated:
        for ax in integrated.get("axes") or []:
            src = ax.get("source") or ""
            stat = ax.get("status") or ""
            ax_name = ax.get("axis") or ""
            if src and stat:
                axis_status_by_source[src] = stat
            if ax_name:
                axes_by_axis[ax_name] = ax

    if is_integrated:
        # Integrated profile: build the USED / PARTIAL list per
        # Deep Research P0/P1/P2 spec ordering.
        used.append(_entry(
            category="royal_road_integrated_core",
            label_ja=f"王道統合判断 (integrated profile, mode={integrated_mode})",
            status=USED,
            reason_ja=(
                f"最終 action ({action}) は integrated decision エンジンが"
                "出しています。波形 / Wライン / フィボ / ローソク足 / ダウ / "
                "MA / RSI / BB / MACD / 損切り / RR / マクロ から作った "
                "evidence axes を統合して判定しました。"
            ),
            what_to_check_ja=(
                "下の各 axis の status (PASS/WARN/BLOCK) と reason_ja を見て、"
                "どの根拠が action に寄与したか確認してください。"
            ),
        ))

        # ── P0: 王道の必須要素 (USED) ──
        # 1. 波形認識
        wp = axes_by_axis.get("wave_pattern")
        if wp is not None:
            used.append(_axis_entry(
                category="wave_pattern_axis",
                label_ja="波形認識 (P0 必須軸)",
                axis=wp,
                use_block_as_used=True,
            ))
        # 2. Wライン (WNL / WSL / WTP / WBR)
        wl = axes_by_axis.get("wave_lines")
        if wl is not None:
            used.append(_axis_entry(
                category="wave_lines_axis",
                label_ja="W ライン (WNL/WSL/WTP/WBR — P0 必須軸)",
                axis=wl,
                use_block_as_used=True,
            ))
        # 3. ダウ構造
        dw = axes_by_axis.get("dow_structure")
        if dw is not None:
            used.append(_axis_entry(
                category="dow_structure_axis",
                label_ja="ダウ構造 (P0 必須軸)",
                axis=dw,
                use_block_as_used=True,
            ))
        # 4. インバリデーション / RR
        inv = axes_by_axis.get("invalidation_rr")
        if inv is not None:
            used.append(_axis_entry(
                category="invalidation_rr_axis",
                label_ja="インバリデーション / RR (P0 必須軸)",
                axis=inv,
                use_block_as_used=True,
            ))

        # ── P1: 補助軸 (PARTIAL / USED-on-PASS) ──
        for axis_name, label_ja, category in (
            ("levels", "サポレジ・レベル心理 (P1)", "levels_axis"),
            ("candlestick", "ローソク足解剖 (P1)", "candlestick_axis"),
            ("ma_context", "MA / グランビル (P1)", "ma_context_axis"),
            ("fibonacci", "フィボナッチ位置 (P1)", "fibonacci_axis"),
            ("macro", "マクロ整合 (P1)", "macro_axis"),
            ("daily_roadmap", "Daily Roadmap (P1)", "daily_roadmap_axis"),
            ("symbol_macro_briefing", "Symbol Macro Briefing (P1)",
             "symbol_macro_briefing_axis"),
        ):
            ax = axes_by_axis.get(axis_name)
            if ax is None:
                continue
            used.append(_axis_entry(
                category=category, label_ja=label_ja, axis=ax,
                use_block_as_used=False,
            ))

        # ── P2: 補助のみ (PARTIAL / never-alone) ──
        for axis_name, label_ja, category in (
            ("rsi_regime", "RSI レジーム (P2 補助のみ)", "rsi_axis"),
            ("bollinger", "ボリンジャーバンド (P2 補助のみ)", "bb_axis"),
            ("macd", "MACD (P2 補助のみ)", "macd_axis"),
            ("divergence", "ダイバージェンス (P2 補助のみ)",
             "divergence_axis"),
        ):
            ax = axes_by_axis.get(axis_name)
            if ax is None:
                continue
            used.append(_axis_entry(
                category=category, label_ja=label_ja, axis=ax,
                use_block_as_used=False,
            ))
    else:
        # Legacy v2: existing classification (unchanged)
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

    masterclass_dict = payload.get("masterclass_panels") or {}
    panels = masterclass_dict.get("panels") or {}

    if is_integrated:
        # Integrated profile: raw panel display itself is human-only.
        # The DECISION uses evidence axes derived from these panels
        # (already listed in `used` above). What sits here are the
        # raw human-readable summaries / supplementary panels that
        # don't translate into single-axis status.
        if payload.get("wave_shape_review"):
            audit_only.append(_entry(
                category="wave_shape_review_raw",
                label_ja="波形認識 raw 表示 (人間用)",
                status=AUDIT_ONLY,
                reason_ja=(
                    "波形 raw パネルは人間が確認するための表示です。"
                    "判断には wave_pattern axis (上の USED 一覧) が"
                    "使われています — raw panel そのものではありません。"
                ),
            ))
        if payload.get("wave_derived_lines"):
            audit_only.append(_entry(
                category="wave_derived_lines_raw",
                label_ja="W ライン raw 描画 (人間用)",
                status=AUDIT_ONLY,
                reason_ja=(
                    "W ライン (WNL/WSL/WTP) のチャート描画は人間用の"
                    "視覚化です。判断には wave_lines axis (P0 必須軸) が"
                    "使われています。"
                ),
            ))
        if masterclass_dict.get("available"):
            # Grand Confluence summary
            if panels.get("grand_confluence_v2"):
                audit_only.append(_entry(
                    category="grand_confluence_summary",
                    label_ja="Grand Confluence v2 summary (人間用)",
                    status=AUDIT_ONLY,
                    reason_ja=(
                        "13 軸の集約 summary は人間の確認用です。"
                        "判断は個別 axis (dow / candlestick / levels / "
                        "ma_context / fibonacci / rsi / bb / macd / "
                        "divergence / invalidation_rr) を直接読みます。"
                    ),
                ))
            # Pre-trade checklist summary
            if panels.get("pre_trade_diagnostic_checklist_v1"):
                audit_only.append(_entry(
                    category="pre_trade_checklist_summary",
                    label_ja="Pre-trade checklist summary (人間用)",
                    status=AUDIT_ONLY,
                    reason_ja=(
                        "Pre-trade checklist は人間が手元で確認するための"
                        "チェックリスト表示。判断には axes が使われています。"
                    ),
                ))
            # Generic "raw 19-panel" reference
            audit_only.append(_entry(
                category="masterclass_panels_raw",
                label_ja="Masterclass 19 パネル raw 表示 (人間用)",
                status=AUDIT_ONLY,
                reason_ja=(
                    "raw Masterclass パネルは人間用の監査表示です。"
                    "判断にはここから抽出した evidence axes (上の USED / "
                    "PARTIAL 一覧) が使われています。"
                ),
                what_to_check_ja=(
                    "raw パネルと axis の status が一致しているか確認して"
                    "ください。例: dow_structure_review.trend が UP なら "
                    "dow_structure axis も BUY 側 PASS。"
                ),
            ))
    else:
        # Legacy v2 (audit-only) — existing behaviour preserved
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
        # Legacy v2 only: fib + daily_roadmap appear here as AUDIT_ONLY.
        # Integrated profile already shows them via axis entries
        # (fibonacci_axis / daily_roadmap_axis) in the USED list.
        if not is_integrated:
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
    if is_integrated:
        # Integrated profile: messages reflect that wave / fib /
        # Masterclass actually drove the action.
        if action == "HOLD":
            integrated_block_reasons = integrated.get("block_reasons") or []
            block_summary = (
                ", ".join(integrated_block_reasons[:3])
                if integrated_block_reasons else "条件不足"
            )
            mode_ja = (
                "strict (必須データ未接続なら HOLD)"
                if integrated_mode == "integrated_strict"
                else "balanced (未接続は WARN として扱う)"
            )
            action_message_ja = (
                "この判断は王道統合ロジック (integrated profile) で出ています。"
                "波形 / Wライン / フィボ / ローソク足 / ダウ / MA / RSI / "
                "BB / MACD / 損切り / RR / マクロ を統合して判定し、"
                f"HOLD になりました。\n\nmode: {mode_ja}\nブロック理由: "
                f"{block_summary}"
            )
        elif action in ("BUY", "SELL"):
            mode_ja = (
                "strict" if integrated_mode == "integrated_strict" else "balanced"
            )
            action_message_ja = (
                f"この判断は王道統合ロジック (integrated profile, mode={mode_ja}) "
                f"で出ています。\n波形 / Wライン / フィボ / ローソク足 / ダウ / "
                f"MA / RSI / BB / MACD / 損切り / RR を統合して {action} を決定。\n"
                "axes 表示で各軸の PASS / WARN / BLOCK と判断への寄与を確認できます。"
            )
        else:
            action_message_ja = f"integrated profile: 最終判断 {action}"
        summary_ja = (
            f"最終判断は {action} (integrated profile, mode={integrated_mode})。"
            "波形 / Wライン / フィボ / Masterclass パネルは判断に直接使用されています。"
        )
        plain_answer_ja = (
            "この画面は『判断に使った根拠』そのものです。"
            "axes の status (PASS/WARN/BLOCK) を見れば、どの根拠が"
            "action に寄与したかが分かります。"
        )
    else:
        # Legacy v2 messaging (audit-only)
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
        "integrated_profile_active": bool(is_integrated),
        "integrated_mode": integrated_mode,
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
