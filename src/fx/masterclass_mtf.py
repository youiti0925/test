"""Masterclass multi-timeframe story (observation-only).

Combines higher / middle / lower timeframe readings into one
human story:

  「上位足は上昇。中位足は押し目。下位足でダブルボトムを作り、
   ネックラインを上抜けたため、買い候補です。」

The function takes:
  - higher_tf_trend (UP / DOWN / RANGE / UNKNOWN) from
    higher_timeframe context
  - middle_tf_state (pullback / rebound / breakout / continuation /
    range / unknown) from dow_structure_review or v2 setups
  - lower_tf_anatomy from masterclass_candlestick.parent_bar_lower_tf
  - waveform best_pattern (already classified by
    pattern_shape_review)

Returns observation-only audit panel with `used_in_decision=False`.
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "multi_timeframe_story_v1"


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
    }


def _trend_label_ja(trend: str | None) -> str:
    return {
        "UP": "上昇", "DOWN": "下降",
        "RANGE": "レンジ", "MIXED": "もみ合い",
        "UNKNOWN": "不明",
    }.get((trend or "UNKNOWN").upper(), "不明")


def _middle_state_ja(state: str | None) -> str:
    return {
        "pullback": "押し目", "rebound": "戻り",
        "breakout": "ブレイク", "continuation": "継続",
        "range": "レンジ", "unknown": "不明",
    }.get((state or "unknown").lower(), "不明")


def _lower_wave_ja(kind: str | None, status: str | None) -> str:
    if not kind:
        return "下位足は明確な波形なし"
    kind_label = {
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
    }.get(kind, kind)
    status_label = {
        "forming": "形成中",
        "neckline_broken": "ネックラインブレイク済み",
        "retested": "リターンムーブ確認済み",
        "invalidated": "形が崩れた",
        "not_matched": "形は弱い",
    }.get(status or "", status or "")
    return f"{kind_label} ({status_label})"


def build_multi_timeframe_story(
    *,
    higher_tf_trend: str | None,
    middle_tf_state: str | None = None,
    lower_tf_kind: str | None = None,
    lower_tf_status: str | None = None,
    lower_tf_side_bias: str | None = None,
) -> dict:
    """Build the multi_timeframe_story panel.

    All inputs may be None — the story still renders with the
    available pieces. Returns `available=True` when at least the
    higher_tf_trend is supplied.
    """
    if not higher_tf_trend:
        return _empty_panel("missing_higher_tf_trend")

    higher_label = _trend_label_ja(higher_tf_trend)
    middle_label = _middle_state_ja(middle_tf_state)
    lower_label = _lower_wave_ja(lower_tf_kind, lower_tf_status)

    aligned = False
    side = None
    if higher_tf_trend.upper() == "UP" and lower_tf_side_bias == "BUY":
        aligned = True
        side = "BUY"
    elif higher_tf_trend.upper() == "DOWN" and lower_tf_side_bias == "SELL":
        aligned = True
        side = "SELL"

    parts = [f"上位足は{higher_label}"]
    if middle_tf_state:
        parts.append(f"中位足は{middle_label}")
    if lower_tf_kind:
        parts.append(f"下位足は{lower_label}")
    parts.append(
        "上位足と下位足の方向が一致しており、王道の押し目・戻りシナリオです。"
        if aligned else
        "上位足と下位足の方向が一致していないため、王道としては慎重な局面です。"
    )
    story_ja = "。".join(parts) + "。"

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "higher_tf": higher_tf_trend.upper(),
        "middle_tf": middle_tf_state,
        "lower_tf": lower_tf_kind,
        "lower_tf_status": lower_tf_status,
        "tf_aligned": bool(aligned),
        "alignment_side": side,
        "story_ja": story_ja,
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_multi_timeframe_story",
]
