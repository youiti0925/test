"""Source-pack Daily 10k FX Roadmap review (observation-only).

Maps the Daily_10k_FX_Roadmap operational checklist onto an audit
panel. The current implementation surfaces the items as a status
list — but several items (notably position sizing, news event
schedule, daily P/L tracking) are NOT yet wired to real data, so
they explicitly emit `status="UNKNOWN"` with a JA note that says
"未接続".

Strict invariants
-----------------
- observation_only=True, used_in_decision=False on the panel and
  every item.
- The panel never gates the v2 final action; it is meant for the
  human reviewer to read on their phone before placing a trade.
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "daily_roadmap_review_v1"


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
        "items": [],
    }


def _item(
    *,
    id_: str,
    label_ja: str,
    status: str,
    reason_ja: str,
    what_to_check_on_chart_ja: str = "",
) -> dict:
    return {
        "id": id_,
        "label_ja": label_ja,
        "status": status,
        "reason_ja": reason_ja,
        "what_to_check_on_chart_ja": what_to_check_on_chart_ja,
        "observation_only": True,
        "used_in_decision": False,
    }


def build_daily_roadmap_review(
    *,
    macro_score: float | None,
    higher_tf_trend: str | None,
    invalidation_review: dict | None = None,
    entry_summary: dict | None = None,
) -> dict:
    """Build the daily_roadmap_review panel.

    The 7 roadmap checklist items map to readily-available signals
    when possible, and explicitly mark "未接続" otherwise.
    """
    items: list[dict] = []

    # 1. Today's bias defined?
    if higher_tf_trend in ("UP", "DOWN"):
        items.append(_item(
            id_="today_bias_defined",
            label_ja="今日の方向感は明確か？",
            status="PASS",
            reason_ja=f"上位足トレンド: {higher_tf_trend}.",
            what_to_check_on_chart_ja=(
                "上位足のチャートで明確なトレンド方向が読み取れるかを"
                "確認してください。"
            ),
        ))
    elif higher_tf_trend in ("RANGE", "MIXED"):
        items.append(_item(
            id_="today_bias_defined",
            label_ja="今日の方向感は明確か？",
            status="WARN",
            reason_ja=f"上位足トレンド: {higher_tf_trend} (方向感が弱い)。",
            what_to_check_on_chart_ja=(
                "レンジ相場では王道は様子見が基本です。"
            ),
        ))
    else:
        macro_label = ""
        if macro_score is not None and abs(macro_score) < 0.2:
            macro_label = "macro bias が NEUTRAL のため方向感は弱いです。"
        items.append(_item(
            id_="today_bias_defined",
            label_ja="今日の方向感は明確か？",
            status="WARN" if macro_score is not None else "UNKNOWN",
            reason_ja=macro_label or "方向感を判定する情報が不足しています。",
        ))

    # 2. Risk per trade defined?
    items.append(_item(
        id_="risk_per_trade_defined",
        label_ja="1回の損失許容が決まっているか？",
        status="UNKNOWN",
        reason_ja="position sizing は未接続です。手動で口座残高 × 1〜2% を設定してください。",
        what_to_check_on_chart_ja=(
            "現在のロットサイズが、口座残高の 1〜2% リスクに収まる損切り"
            "幅と整合しているかを取引画面で確認してください。"
        ),
    ))

    # 3. RR planned?
    rr = (entry_summary or {}).get("rr") if entry_summary else None
    if isinstance(rr, (int, float)):
        if rr >= 2.0:
            items.append(_item(
                id_="rr_planned",
                label_ja="RR は 1:2 以上か？",
                status="PASS",
                reason_ja=f"RR = {rr:.2f} ≥ 2.0.",
            ))
        else:
            items.append(_item(
                id_="rr_planned",
                label_ja="RR は 1:2 以上か？",
                status="NO" if rr >= 0 else "BLOCK",
                reason_ja=f"RR = {rr:.2f} < 2.0.",
            ))
    else:
        items.append(_item(
            id_="rr_planned",
            label_ja="RR は 1:2 以上か？",
            status="UNKNOWN",
            reason_ja="entry_summary.rr が算出されていません。",
        ))

    # 4. Invalidation defined?
    if invalidation_review and invalidation_review.get("available"):
        if invalidation_review.get("is_structure_anchored"):
            items.append(_item(
                id_="invalidation_defined",
                label_ja="損切り (invalidation) は構造ベースで決まっているか？",
                status="PASS",
                reason_ja="構造アンカーベースの損切り設定済み。",
            ))
        else:
            items.append(_item(
                id_="invalidation_defined",
                label_ja="損切り (invalidation) は構造ベースで決まっているか？",
                status="WARN",
                reason_ja="ATR距離損切りで、構造アンカーベースではありません。",
            ))
    else:
        items.append(_item(
            id_="invalidation_defined",
            label_ja="損切り (invalidation) は構造ベースで決まっているか？",
            status="UNKNOWN",
            reason_ja="invalidation 情報が不足しています。",
        ))

    # 5. News event schedule checked? (未接続)
    items.append(_item(
        id_="news_calendar_checked",
        label_ja="今日の経済指標スケジュールを確認したか？",
        status="UNKNOWN",
        reason_ja="news / event_overlay は未接続です。手動でカレンダーを確認してください。",
        what_to_check_on_chart_ja=(
            "Forex Factory / Investing.com 等で当日の重要指標 (CPI / NFP / FOMC など) を"
            "確認してください。"
        ),
    ))

    # 6. Daily P/L tracker (未接続)
    items.append(_item(
        id_="daily_pl_tracking",
        label_ja="今日の累積損益を把握しているか？",
        status="UNKNOWN",
        reason_ja="daily P/L トラッカーは未接続です。",
    ))

    # 7. Trade journal (未接続)
    items.append(_item(
        id_="trade_journal_ready",
        label_ja="トレード日誌は準備できているか？",
        status="UNKNOWN",
        reason_ja="トレード日誌は未接続です。手動で記録してください。",
    ))

    pass_count = sum(1 for i in items if i["status"] == "PASS")
    warn_count = sum(1 for i in items if i["status"] == "WARN")
    unknown_count = sum(1 for i in items if i["status"] == "UNKNOWN")
    no_count = sum(1 for i in items if i["status"] in ("NO", "BLOCK"))
    if no_count > 0:
        verdict_ja = (
            "Daily roadmap で NG 項目があります。エントリー前に解決してください。"
        )
        agg_status = "BLOCK"
    elif unknown_count >= 3:
        verdict_ja = (
            "未接続項目が多いため、ポジションサイジング・経済指標カレンダー・"
            "ジャーナルは手動で確認してください。"
        )
        agg_status = "WARN"
    elif pass_count >= 3:
        verdict_ja = "Daily roadmap の重要項目はクリアしています。"
        agg_status = "PASS"
    else:
        verdict_ja = "Daily roadmap の項目が不足しています。"
        agg_status = "WARN"

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "status": agg_status,
        "items": items,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "unknown_count": unknown_count,
        "no_count": no_count,
        "verdict_ja": verdict_ja,
        "what_to_check_on_chart_ja": (
            "未接続 (UNKNOWN) 項目は手動で確認してください。"
            "特にポジションサイジング・経済指標・ジャーナルは王道では必須です。"
        ),
        "inputs_used": [
            "macro_score (input)",
            "higher_tf_trend (input)",
            "invalidation_engine_v2",
            "entry_summary",
        ],
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_daily_roadmap_review",
]
