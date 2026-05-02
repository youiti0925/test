"""Masterclass candlestick anatomy review (observation-only).

Two panels:

  candlestick_anatomy_review
    Classifies the parent_bar's body / wick anatomy into one of:
      large_bull / large_bear / power_move
      bullish_pinbar / bearish_pinbar
      bullish_engulfing / bearish_engulfing
      bullish_harami / bearish_harami
      neutral_doji / range_bar / unclassified
    Each carries a JA `meaning_ja`, `context_ja`, and `warning_ja`
    so the audit reader can verify the system's reading against
    their own.

  parent_bar_lower_tf_anatomy
    Takes the LOWER-timeframe bars that fall inside the parent_bar's
    wall-clock window and encodes their wave skeleton, then reports
    a Japanese interpretation ("上位足の陽線内部で、下位足がダブル
    ボトムを作って反発しています"). When the lower-timeframe df is
    not available, returns `available=False` with
    `unavailable_reason="lower_tf_missing"`.

Strict invariants
-----------------
- Both panels carry `observation_only = True` and
  `used_in_decision = False`.
- The candlestick classifier never reads bars after parent_bar; the
  lower-tf anatomy never reads bars after parent_bar's end ts.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


CANDLESTICK_SCHEMA: Final[str] = "candlestick_anatomy_review_v1"
LOWER_TF_SCHEMA: Final[str] = "parent_bar_lower_tf_anatomy_v1"


def _empty_panel(schema_version: str, reason: str) -> dict:
    return {
        "schema_version": schema_version,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
    }


def _classify_anatomy(
    *,
    parent: pd.Series,
    prev: pd.Series | None,
    atr_value: float | None,
) -> tuple[str, str, str | None]:
    """Return (bar_type, direction, special_marker).

    bar_type is the primary anatomy label.
    direction is "BUY" / "SELL" / "NEUTRAL".
    special_marker may be "power_move" when the body dwarfs ATR.
    """
    o, h, l, c = (
        float(parent["open"]),
        float(parent["high"]),
        float(parent["low"]),
        float(parent["close"]),
    )
    rng = h - l
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if rng <= 0:
        return "range_bar", "NEUTRAL", None
    body_frac = body / rng
    is_bull = c > o
    is_bear = c < o
    atr = float(atr_value) if (atr_value is not None and atr_value > 0) else None

    # 1. Pinbar: long opposite-side wick relative to body
    if body_frac < 0.35:
        if lower_wick >= 2.0 * body and lower_wick > upper_wick:
            return "bullish_pinbar", "BUY", None
        if upper_wick >= 2.0 * body and upper_wick > lower_wick:
            return "bearish_pinbar", "SELL", None

    # 2. Engulfing: needs prev bar
    if prev is not None:
        po, pc = float(prev["open"]), float(prev["close"])
        prev_bull = pc > po
        prev_bear = pc < po
        prev_body_low = min(po, pc)
        prev_body_high = max(po, pc)
        cur_body_low = min(o, c)
        cur_body_high = max(o, c)
        if (
            is_bull and prev_bear
            and cur_body_low < prev_body_low
            and cur_body_high > prev_body_high
        ):
            return "bullish_engulfing", "BUY", None
        if (
            is_bear and prev_bull
            and cur_body_low < prev_body_low
            and cur_body_high > prev_body_high
        ):
            return "bearish_engulfing", "SELL", None
        # Harami: current body inside prev body
        if (
            is_bull and prev_bear
            and cur_body_low > prev_body_low
            and cur_body_high < prev_body_high
        ):
            return "bullish_harami", "BUY", None
        if (
            is_bear and prev_bull
            and cur_body_low > prev_body_low
            and cur_body_high < prev_body_high
        ):
            return "bearish_harami", "SELL", None

    # 3. Doji: tiny body relative to range
    if body_frac < 0.10:
        return "neutral_doji", "NEUTRAL", None

    # 4. Large-body classification
    if body_frac > 0.70:
        special = None
        if atr is not None and body >= 1.5 * atr:
            special = "power_move"
        if is_bull:
            return "large_bull", "BUY", special
        if is_bear:
            return "large_bear", "SELL", special

    if is_bull:
        return "small_bull", "BUY", None
    if is_bear:
        return "small_bear", "SELL", None
    return "unclassified", "NEUTRAL", None


_BAR_TYPE_JA: Final[dict] = {
    "large_bull": "大陽線",
    "large_bear": "大陰線",
    "small_bull": "小陽線",
    "small_bear": "小陰線",
    "bullish_pinbar": "下ヒゲピンバー (Bullish Pinbar)",
    "bearish_pinbar": "上ヒゲピンバー (Bearish Pinbar)",
    "bullish_engulfing": "陽の包み足 (Bullish Engulfing)",
    "bearish_engulfing": "陰の包み足 (Bearish Engulfing)",
    "bullish_harami": "陽のはらみ足 (Bullish Harami)",
    "bearish_harami": "陰のはらみ足 (Bearish Harami)",
    "neutral_doji": "十字線 (Doji)",
    "range_bar": "レンジバー",
    "unclassified": "判定不能",
}


def _meaning_ja(bar_type: str, direction: str, special: str | None) -> str:
    label = _BAR_TYPE_JA.get(bar_type, bar_type)
    if bar_type in ("bullish_pinbar",):
        base = "下ヒゲの長いピンバー。下方向を試したが戻された反発サインです。"
    elif bar_type == "bearish_pinbar":
        base = "上ヒゲの長いピンバー。上方向を試したが戻された反発サインです。"
    elif bar_type == "bullish_engulfing":
        base = (
            "前の陰線を完全に呑み込んだ陽線。買い圧力の急増を示唆します。"
        )
    elif bar_type == "bearish_engulfing":
        base = (
            "前の陽線を完全に呑み込んだ陰線。売り圧力の急増を示唆します。"
        )
    elif bar_type == "bullish_harami":
        base = (
            "前の陰線の中に小さな陽線が収まる形。下落の勢いが弱まり、"
            "反発の可能性を示唆します。"
        )
    elif bar_type == "bearish_harami":
        base = (
            "前の陽線の中に小さな陰線が収まる形。上昇の勢いが弱まり、"
            "反落の可能性を示唆します。"
        )
    elif bar_type == "large_bull":
        base = (
            "実体が大きい陽線。買いの勢いを示します。"
        )
    elif bar_type == "large_bear":
        base = (
            "実体が大きい陰線。売りの勢いを示します。"
        )
    elif bar_type == "neutral_doji":
        base = (
            "実体がほぼゼロの十字線。買い手と売り手が拮抗しており、"
            "方向感を欠いています。"
        )
    elif bar_type == "range_bar":
        base = "高値と安値が同値で値動きがない足。"
    else:
        base = f"{label}。"
    if special == "power_move":
        base += " ATR比で大きい (パワームーブ)。"
    return f"{label}。{base}"


def _context_ja(
    bar_type: str,
    direction: str,
    *,
    near_support: bool,
    near_resistance: bool,
) -> str:
    if direction == "BUY":
        if near_support:
            return "サポート帯付近で出ているため、反発根拠になり得ます。"
        return "サポート帯から離れている位置のため、信頼度はやや下がります。"
    if direction == "SELL":
        if near_resistance:
            return "レジスタンス帯付近で出ているため、反落根拠になり得ます。"
        return "レジスタンス帯から離れている位置のため、信頼度はやや下がります。"
    return "方向の偏りが弱いため、単独では判断材料になりません。"


def _warning_ja(bar_type: str, direction: str) -> str:
    if bar_type in ("bullish_pinbar", "bearish_pinbar"):
        return (
            "レンジ中央や上位足逆行で出たピンバーは騙しになりやすいです。"
        )
    if bar_type in ("bullish_engulfing", "bearish_engulfing"):
        return (
            "包み足は単独で十分ではなく、サポレジ・トレンド方向との整合が必要です。"
        )
    if bar_type in ("bullish_harami", "bearish_harami"):
        return (
            "はらみ足は『反転候補』であり、追随する確認バーが必要です。"
        )
    if bar_type == "neutral_doji":
        return "十字線は方向不明。次の足で方向を確認してください。"
    return ""


def build_candlestick_anatomy_review(
    *,
    visible_df: pd.DataFrame | None,
    atr_value: float | None,
    near_support: bool = False,
    near_resistance: bool = False,
) -> dict:
    """Classify the latest bar of `visible_df` (the parent bar) into
    a candlestick anatomy label and emit a JA audit panel.

    `visible_df` MUST already be truncated to bars at or before
    parent_bar_ts. We never reach beyond `iloc[-1]`.
    """
    if visible_df is None or len(visible_df) == 0:
        return _empty_panel(CANDLESTICK_SCHEMA, "no_visible_bars")
    parent = visible_df.iloc[-1]
    prev = visible_df.iloc[-2] if len(visible_df) >= 2 else None
    bar_type, direction, special = _classify_anatomy(
        parent=parent, prev=prev, atr_value=atr_value,
    )
    return {
        "schema_version": CANDLESTICK_SCHEMA,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "bar_type": bar_type,
        "direction": direction,
        "special_marker": special,
        "open": float(parent["open"]),
        "high": float(parent["high"]),
        "low": float(parent["low"]),
        "close": float(parent["close"]),
        "atr_value": (
            float(atr_value) if (atr_value is not None and atr_value > 0)
            else None
        ),
        "near_support": bool(near_support),
        "near_resistance": bool(near_resistance),
        "meaning_ja": _meaning_ja(bar_type, direction, special),
        "context_ja": _context_ja(
            bar_type, direction,
            near_support=near_support, near_resistance=near_resistance,
        ),
        "warning_ja": _warning_ja(bar_type, direction),
    }


# ---------------------------------------------------------------------------
# Lower-TF anatomy of the parent bar
# ---------------------------------------------------------------------------


def build_parent_bar_lower_tf_anatomy(
    *,
    parent_bar_ts: pd.Timestamp | str | None,
    parent_bar_open_ts: pd.Timestamp | str | None = None,
    df_lower_tf: pd.DataFrame | None = None,
    higher_tf_interval_minutes: int = 60,
) -> dict:
    """Encode the lower-tf bars that fall inside the parent bar.

    `df_lower_tf` is the lower-timeframe DataFrame (e.g. 5-minute
    bars when the parent_bar is 1-hour). We slice it to the bars
    whose ts is within `[parent_open_ts, parent_close_ts]`, encode
    a wave skeleton, run shape matching, and report the best match.

    When `df_lower_tf` is None or empty: returns
    `available=False, unavailable_reason="lower_tf_missing"`.
    """
    if df_lower_tf is None or len(df_lower_tf) == 0:
        return _empty_panel(LOWER_TF_SCHEMA, "lower_tf_missing")
    if parent_bar_ts is None:
        return _empty_panel(LOWER_TF_SCHEMA, "missing_parent_bar_ts")
    parent_close = pd.Timestamp(parent_bar_ts)
    if parent_bar_open_ts is None:
        parent_open = parent_close - pd.Timedelta(
            minutes=higher_tf_interval_minutes,
        )
    else:
        parent_open = pd.Timestamp(parent_bar_open_ts)
    if parent_open >= parent_close:
        return _empty_panel(LOWER_TF_SCHEMA, "invalid_parent_window")
    inside = df_lower_tf[
        (df_lower_tf.index > parent_open)
        & (df_lower_tf.index <= parent_close)
    ]
    if len(inside) < 6:
        return {
            "schema_version": LOWER_TF_SCHEMA,
            "available": False,
            "observation_only": True,
            "used_in_decision": False,
            "unavailable_reason": "too_few_lower_tf_bars",
            "lower_tf_bars_inside": int(len(inside)),
        }

    # Encode the lower-tf skeleton; observation-only.
    from .pattern_shape_matcher import match_skeleton
    from .waveform_encoder import encode_wave_skeleton

    skel = encode_wave_skeleton(inside, scale="lower_tf_inside_parent")
    if not skel.pivots:
        return {
            "schema_version": LOWER_TF_SCHEMA,
            "available": True,
            "observation_only": True,
            "used_in_decision": False,
            "parent_bar_open_ts": parent_open.isoformat(),
            "parent_bar_close_ts": parent_close.isoformat(),
            "lower_tf_bars_inside": int(len(inside)),
            "lower_tf_wave": None,
            "lower_tf_status": None,
            "shape_score": 0.0,
            "meaning_ja": (
                "下位足では明確な波形は出ていません。"
                "上位足の中身は方向感が弱い動きです。"
            ),
        }
    matches = match_skeleton(skel)
    best = matches[0] if matches else None
    direction = (
        "陽線" if float(inside["close"].iloc[-1]) > float(inside["close"].iloc[0])
        else "陰線"
    )
    if best is None or best.shape_score < 0.70:
        meaning = (
            f"上位足の{direction}内部で、下位足は明確な波形を作っていません。"
        )
        return {
            "schema_version": LOWER_TF_SCHEMA,
            "available": True,
            "observation_only": True,
            "used_in_decision": False,
            "parent_bar_open_ts": parent_open.isoformat(),
            "parent_bar_close_ts": parent_close.isoformat(),
            "lower_tf_bars_inside": int(len(inside)),
            "lower_tf_wave": None,
            "lower_tf_status": "not_matched",
            "shape_score": float(best.shape_score) if best else 0.0,
            "meaning_ja": meaning,
        }
    status_ja_map = {
        "forming": "形成中",
        "neckline_broken": "ネックラインブレイク済み",
        "retested": "リターンムーブ確認済み",
        "invalidated": "形が崩れた",
    }
    status_ja = status_ja_map.get(best.status, best.status)
    meaning = (
        f"上位足の{direction}内部で、下位足が{best.human_label}を作り、"
        f"現在は{status_ja}です。"
    )
    return {
        "schema_version": LOWER_TF_SCHEMA,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "parent_bar_open_ts": parent_open.isoformat(),
        "parent_bar_close_ts": parent_close.isoformat(),
        "lower_tf_bars_inside": int(len(inside)),
        "lower_tf_wave": best.kind,
        "lower_tf_status": best.status,
        "shape_score": float(best.shape_score),
        "side_bias": best.side_bias,
        "meaning_ja": meaning,
    }


__all__ = [
    "CANDLESTICK_SCHEMA",
    "LOWER_TF_SCHEMA",
    "build_candlestick_anatomy_review",
    "build_parent_bar_lower_tf_anatomy",
]
