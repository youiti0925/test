"""Masterclass chart-pattern anatomy v2 (observation-only).

Wraps the existing pattern_shape_review + wave_derived_lines output
into a Masterclass-flavoured panel that lists, for the matched
pattern:

  detected_pattern (kind):
    double_bottom / double_top / head_and_shoulders /
    inverse_head_and_shoulders / bullish_flag / bearish_flag /
    rising_wedge / falling_wedge / ascending_triangle /
    descending_triangle / symmetric_triangle

  status:
    forming / neckline_not_broken / neckline_broken / retested /
    invalidated / not_matched

  parts (id → present? + price):
    B1 / B2 / P1 / P2 / LS / H / RS / NL / BR / RT / SL / TP

`RT` (retest) and the `retested` status are heuristic placeholders
in this v1 — currently set to `False` / not raised because the
underlying matcher only emits forming / neckline_broken / etc.

Strict invariants
-----------------
- Output carries `observation_only = True` and
  `used_in_decision = False`.
- This module does NOT re-run the encoder; it only repackages the
  already-computed `wave_shape_review` + `wave_derived_lines`.
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "chart_pattern_anatomy_v2"


_KIND_LABEL_JA: Final[dict] = {
    "double_bottom": "ダブルボトム",
    "double_top": "ダブルトップ",
    "head_and_shoulders": "三尊",
    "inverse_head_and_shoulders": "逆三尊",
    "bullish_flag": "上昇フラッグ",
    "bearish_flag": "下降フラッグ",
    "rising_wedge": "上昇ウェッジ",
    "falling_wedge": "下降ウェッジ",
    "ascending_triangle": "上昇三角形",
    "descending_triangle": "下降三角形",
    "symmetric_triangle": "対称三角形",
}


_PART_LABEL_BY_KIND: Final[dict] = {
    "double_bottom": ("B1", "B2", "NL", "BR"),
    "double_top": ("P1", "P2", "NL", "BR"),
    "head_and_shoulders": ("LS", "H", "RS", "NL", "BR"),
    "inverse_head_and_shoulders": ("LS", "H", "RS", "NL", "BR"),
    "bullish_flag": ("I0", "I1", "C-H", "C-L", "BR"),
    "bearish_flag": ("I0", "I1", "C-H", "C-L", "BR"),
    "rising_wedge": ("A1", "A2", "A3", "BR"),
    "falling_wedge": ("A1", "A2", "A3", "BR"),
    "ascending_triangle": ("A1", "A2", "A3", "BR"),
    "descending_triangle": ("A1", "A2", "A3", "BR"),
    "symmetric_triangle": ("A1", "A2", "A3", "AP"),
}


_PART_NAME_TO_LABEL: Final[dict] = {
    "first_bottom": "B1",
    "second_bottom": "B2",
    "first_top": "P1",
    "second_top": "P2",
    "left_shoulder": "LS",
    "right_shoulder": "RS",
    "head": "H",
    "neckline_peak": "NL",
    "neckline_trough": "NL",
    "neckline_left_peak": "NL",
    "neckline_right_peak": "NL",
    "neckline_left_trough": "NL",
    "neckline_right_trough": "NL",
    "breakout": "BR",
    "apex": "AP",
    "impulse_start": "I0",
    "impulse_end": "I1",
    "consolidation_high": "C-H",
    "consolidation_low": "C-L",
    "consolidation_mid": "C-M",
    "lower_anchor_1": "A1",
    "upper_anchor_1": "A1",
    "lower_anchor_2": "A2",
    "upper_anchor_2": "A2",
    "lower_anchor_3": "A3",
    "upper_anchor_3": "A3",
}


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
    }


def build_chart_pattern_anatomy_v2(
    *,
    wave_shape_review: dict | None,
    wave_derived_lines: list[dict] | None,
) -> dict:
    """Build the chart_pattern_anatomy_v2 panel.

    Inputs:
      - wave_shape_review: dict from pattern_shape_review
      - wave_derived_lines: list from wave_derived_lines.build_*

    The function relies entirely on already-computed inputs; it
    never reruns the matcher.
    """
    review = wave_shape_review or {}
    best = review.get("best_pattern") or {}
    if not best:
        return _empty_panel("no_best_pattern")
    kind = best.get("kind") or ""
    if kind not in _KIND_LABEL_JA:
        return _empty_panel(f"unknown_kind:{kind}")

    raw_status = best.get("status") or "not_matched"
    matched_parts = best.get("matched_parts") or {}
    has_neckline = any(
        "neckline" in name for name in matched_parts.keys()
    )
    # Status remap to Masterclass labels.
    if raw_status == "forming" and has_neckline:
        status = "neckline_not_broken"
    else:
        status = raw_status
    side_bias = best.get("side_bias") or "NEUTRAL"

    # Parts breakdown
    parts_present: dict[str, dict] = {}
    expected_labels = list(_PART_LABEL_BY_KIND.get(kind, ()))
    for src_part, src_idx in matched_parts.items():
        label = _PART_NAME_TO_LABEL.get(src_part)
        if not label:
            continue
        if src_idx is None:
            continue
        parts_present.setdefault(label, {
            "label": label,
            "src_part_name": src_part,
            "pivot_index": int(src_idx),
        })
    # Cross-check expected list
    expected_with_status = []
    for lbl in expected_labels:
        expected_with_status.append({
            "label": lbl,
            "present": lbl in parts_present,
        })

    # SL / TP / RT pulled from wave_derived_lines
    derived = wave_derived_lines or []
    sl_line = next(
        (l for l in derived if l.get("role") == "stop_candidate"),
        None,
    )
    tp_line = next(
        (l for l in derived if l.get("role") == "target_candidate"),
        None,
    )
    rt_present = False  # retest detection not implemented in v1

    label_ja = _KIND_LABEL_JA.get(kind, kind)
    side_label_ja = {
        "BUY": "買い候補", "SELL": "売り候補", "NEUTRAL": "中立",
    }.get(side_bias, side_bias)
    status_ja_map = {
        "forming": "形成中",
        "neckline_not_broken": "ネックライン未ブレイク (形成中)",
        "neckline_broken": "ネックラインブレイク済み",
        "retested": "リターンムーブ確認済み",
        "invalidated": "形が崩れた",
        "not_matched": "形として弱い",
    }
    status_label_ja = status_ja_map.get(status, status)
    summary_ja = (
        f"{label_ja}候補 ({side_label_ja}) — 状態: {status_label_ja}。"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "detected_pattern": kind,
        "kind_label_ja": label_ja,
        "status": status,
        "status_label_ja": status_label_ja,
        "side_bias": side_bias,
        "shape_score": float(best.get("shape_score") or 0.0),
        "expected_parts": expected_with_status,
        "matched_parts_present": list(parts_present.values()),
        "sl_line": sl_line,
        "tp_line": tp_line,
        "retest_present": bool(rt_present),
        "summary_ja": summary_ja,
        "next_check_ja": _next_check_ja(status, label_ja),
    }


def _next_check_ja(status: str, label_ja: str) -> str:
    if status == "forming":
        return f"{label_ja}は形成中。ネックラインブレイクを待ちます。"
    if status == "neckline_not_broken":
        return f"{label_ja}のネックラインを終値で抜けるかを確認してください。"
    if status == "neckline_broken":
        return (
            f"{label_ja}のネックラインはブレイクしています。"
            "リターンムーブ (押し戻し再確認) を確認してください。"
        )
    if status == "retested":
        return (
            f"{label_ja}のリターンムーブも確認できています。"
            "RR と上位足整合を最終確認してください。"
        )
    if status == "invalidated":
        return f"{label_ja}は崩れました。形を根拠としたエントリーは見送りです。"
    return f"{label_ja}は形として弱いため、形を理由にしたエントリーは避けます。"


__all__ = [
    "SCHEMA_VERSION",
    "build_chart_pattern_anatomy_v2",
]
