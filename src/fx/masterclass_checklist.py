"""Masterclass pre-trade diagnostic checklist v1 (observation-only).

Reduces all the other masterclass panels into a 7-question
yes/no/unknown checklist:

  1. 現在のダウ理論のトレンドは明確か？
  2. 反発を裏付けるラインはあるか？
  3. 上位足と下位足の方向が一致しているか？
  4. MAやBBで環境認識できるか？
  5. RSI/MACDなどに根拠があるか？
  6. 明確な損切りポイントがあるか？
  7. RRは最低1:2以上あるか？

Output keeps the question (`question_ja`), the answer
(`status` ∈ {"YES", "NO", "UNKNOWN"}), and the
`reason_ja` for traceability.

Strict invariants
-----------------
- observation_only=True, used_in_decision=False on every entry.
- The aggregate panel never blocks the v2 final action.
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "pre_trade_diagnostic_checklist_v1"


def _q(question_ja: str, status: str, reason_ja: str) -> dict:
    return {
        "question_ja": question_ja,
        "status": status,
        "reason_ja": reason_ja,
    }


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
        "questions": [],
    }


def build_pre_trade_diagnostic_checklist(
    *,
    dow_review: dict | None,
    overlays: dict | None,
    wave_derived_lines: list[dict] | None,
    mtf_story: dict | None,
    ma_review: dict | None,
    bb_review: dict | None,
    rsi_review: dict | None,
    macd_review: dict | None,
    invalidation_review: dict | None,
    entry_summary: dict | None,
) -> dict:
    """Build the 7-question diagnostic panel."""
    questions: list[dict] = []

    # 1. Dow trend明確か
    trend = (dow_review or {}).get("trend", "UNKNOWN")
    if trend in ("UP", "DOWN"):
        questions.append(_q(
            "現在のダウ理論のトレンドは明確か？", "YES",
            f"トレンド: {trend}.",
        ))
    elif trend in ("RANGE", "MIXED"):
        questions.append(_q(
            "現在のダウ理論のトレンドは明確か？", "NO",
            f"トレンドが不明確 ({trend}).",
        ))
    else:
        questions.append(_q(
            "現在のダウ理論のトレンドは明確か？", "UNKNOWN",
            "ダウ判定に必要なスイングが不足。",
        ))

    # 2. 反発を裏付けるラインはあるか
    sr_sel = len((overlays or {}).get("level_zones_selected") or [])
    tl_sel = len((overlays or {}).get("trendlines_selected") or [])
    wd_total = len(wave_derived_lines or [])
    line_total = sr_sel + tl_sel + wd_total
    if line_total >= 2:
        questions.append(_q(
            "反発を裏付けるラインはあるか？", "YES",
            f"{sr_sel} S/R + {tl_sel} 線 + {wd_total} 波形由来 = "
            f"{line_total} 本のラインで裏付けあり。",
        ))
    elif line_total == 1:
        questions.append(_q(
            "反発を裏付けるラインはあるか？", "UNKNOWN",
            "ライン 1 本のみ。複数の根拠が望ましい。",
        ))
    else:
        questions.append(_q(
            "反発を裏付けるラインはあるか？", "NO",
            "サポレジ/トレンドライン/波形由来ラインがありません。",
        ))

    # 3. 上位足と下位足の方向が一致しているか
    if mtf_story and mtf_story.get("available"):
        if mtf_story.get("tf_aligned"):
            questions.append(_q(
                "上位足と下位足の方向が一致しているか？", "YES",
                f"上位 {mtf_story.get('higher_tf')} と下位 "
                f"{mtf_story.get('lower_tf')} の方向が一致。",
            ))
        else:
            questions.append(_q(
                "上位足と下位足の方向が一致しているか？", "NO",
                "上位足と下位足の方向が一致していません。",
            ))
    else:
        questions.append(_q(
            "上位足と下位足の方向が一致しているか？", "UNKNOWN",
            "マルチタイムフレーム情報不足。",
        ))

    # 4. MAやBBで環境認識できるか
    ma_ok = bool(
        ma_review and (
            ma_review.get("is_uptrend_stack")
            or ma_review.get("is_downtrend_stack")
        )
    )
    bb_ok = bool(
        bb_review and bb_review.get("stage")
        and bb_review.get("stage") != "neutral"
    )
    if ma_ok or bb_ok:
        questions.append(_q(
            "MAやBBで環境認識できるか？", "YES",
            f"MA配列: {'OK' if ma_ok else 'mixed'} / "
            f"BBステージ: {(bb_review or {}).get('stage', 'unknown')}.",
        ))
    elif ma_review or bb_review:
        questions.append(_q(
            "MAやBBで環境認識できるか？", "UNKNOWN",
            "MA / BB の状態は読めますが、環境認識として弱いです。",
        ))
    else:
        questions.append(_q(
            "MAやBBで環境認識できるか？", "UNKNOWN",
            "MA / BB の入力不足。",
        ))

    # 5. RSI/MACDなどに根拠があるか
    rsi_ok = bool(rsi_review and rsi_review.get("usable_signal"))
    macd_ok = bool(
        macd_review
        and macd_review.get("bias") in ("BUY", "SELL")
    )
    if rsi_ok or macd_ok:
        bits = []
        if rsi_ok:
            bits.append(
                f"RSI {rsi_review.get('rsi_value', 0):.1f} を環境的に使える。"
            )
        if macd_ok:
            bits.append(f"MACD bias: {macd_review.get('bias')}.")
        questions.append(_q(
            "RSI/MACDなどに根拠があるか？", "YES",
            " / ".join(bits),
        ))
    elif rsi_review or macd_review:
        questions.append(_q(
            "RSI/MACDなどに根拠があるか？", "UNKNOWN",
            "値はありますが、現在の環境では根拠として弱いです。",
        ))
    else:
        questions.append(_q(
            "RSI/MACDなどに根拠があるか？", "UNKNOWN",
            "オシレータ入力不足。",
        ))

    # 6. 明確な損切りポイントがあるか
    if invalidation_review and invalidation_review.get("available"):
        if invalidation_review.get("is_structure_anchored"):
            questions.append(_q(
                "明確な損切りポイントがあるか？", "YES",
                "構造アンカー (サポレジ・波形) ベースの損切りが設定されています。",
            ))
        else:
            questions.append(_q(
                "明確な損切りポイントがあるか？", "UNKNOWN",
                "ATR距離損切りで、構造アンカーベースではありません。",
            ))
    else:
        questions.append(_q(
            "明確な損切りポイントがあるか？", "NO",
            "損切り設定なし。",
        ))

    # 7. RRは最低1:2以上あるか
    rr = (entry_summary or {}).get("rr")
    if isinstance(rr, (int, float)):
        if rr >= 2.0:
            questions.append(_q(
                "RRは最低1:2以上あるか？", "YES", f"RR = {rr:.2f}.",
            ))
        else:
            questions.append(_q(
                "RRは最低1:2以上あるか？", "NO", f"RR = {rr:.2f} < 2.0.",
            ))
    else:
        questions.append(_q(
            "RRは最低1:2以上あるか？", "UNKNOWN",
            f"RR算出不可 ({(entry_summary or {}).get('rr_unavailable_reason') or 'unknown'}).",
        ))

    yes_count = sum(1 for q in questions if q["status"] == "YES")
    no_count = sum(1 for q in questions if q["status"] == "NO")
    unknown_count = sum(1 for q in questions if q["status"] == "UNKNOWN")
    if yes_count >= 6 and no_count == 0:
        verdict_ja = (
            "事前診断クリア。王道の条件はほぼ整っています "
            "(最終確認は人間が行ってください)。"
        )
    elif no_count >= 1:
        verdict_ja = (
            f"事前診断 NG が {no_count} 項目あります。エントリーは見送り推奨。"
        )
    else:
        verdict_ja = (
            "事前診断は不明確。情報が揃ったら再評価してください。"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "questions": questions,
        "yes_count": yes_count,
        "no_count": no_count,
        "unknown_count": unknown_count,
        "verdict_ja": verdict_ja,
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_pre_trade_diagnostic_checklist",
]
