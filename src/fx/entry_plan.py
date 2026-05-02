"""entry_plan_v1 — discriminate READY / WAIT_BREAKOUT / WAIT_RETEST /
HOLD entry states from a recognized pattern.

Royal-road trading distinguishes between:

  WAIT_BREAKOUT   — pattern recognized but trigger line (NL/BR) not
                    broken yet.
  WAIT_RETEST     — trigger line broken; waiting for the price to
                    return to the line and produce a confirmation
                    candle (return-move / pullback).
  READY           — retest confirmed AND a confirmation candle prints
                    in the trade direction; stop / target / RR≥2 all
                    present. The integrated decision emits BUY/SELL.
  HOLD            — any blocker (no stop, no target, RR<2, missing
                    pattern, conflicting Dow, etc.).

This module is observation-only. It does NOT consult future bars.
The retest detection looks at the visible window AFTER the breakout
bar and checks whether price has come back to the trigger line then
produced a directional candle.

Output schema (as used in audit and bridge HTML):

  {
    "schema_version": "entry_plan_v1",
    "side": "BUY" / "SELL" / "NEUTRAL",
    "entry_type": "breakout" / "retest" / "none",
    "entry_status": "READY" / "WAIT_BREAKOUT" / "WAIT_RETEST" / "HOLD",
    "trigger_line_id": str | None,
    "trigger_line_price": float | None,
    "entry_price": float | None,         # READY: actual entry; else None
    "stop_price": float | None,
    "target_price": float | None,        # 1× projected (chart target)
    "target_extended_price": float | None,  # 2× projected (RR target)
    "rr": float | None,                  # at extended target
    "breakout_confirmed": bool,
    "retest_confirmed": bool,
    "confirmation_candle": str,          # "bullish_pinbar" / ...
    "reason_ja": str,
    "what_to_wait_for_ja": str,
    "block_reasons": list[str],
  }
"""
from __future__ import annotations

from typing import Final

import pandas as pd


SCHEMA_VERSION: Final[str] = "entry_plan_v1"


_BULLISH_BARS: Final[frozenset] = frozenset({
    "bullish_pinbar", "bullish_engulfing", "bullish_harami",
    "large_bull",
})
_BEARISH_BARS: Final[frozenset] = frozenset({
    "bearish_pinbar", "bearish_engulfing", "bearish_harami",
    "large_bear",
})


def _empty_plan(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "side": "NEUTRAL",
        "entry_type": "none",
        "entry_status": "HOLD",
        "trigger_line_id": None,
        "trigger_line_price": None,
        "entry_price": None,
        "stop_price": None,
        "target_price": None,
        "target_extended_price": None,
        "rr": None,
        "breakout_confirmed": False,
        "retest_confirmed": False,
        "confirmation_candle": "",
        "reason_ja": "エントリープランの根拠が不足しています。",
        "what_to_wait_for_ja": "波形の認識を待ちます。",
        "block_reasons": [reason],
    }


def _find_first_breakout_bar(
    *, side: str, df_window: pd.DataFrame, trigger_price: float,
    start_index: int = 0,
) -> int | None:
    """Locate the FIRST bar (by close) that crosses the trigger in
    the breakout direction. Different from skeleton's BR pivot
    (which is the last extreme). Used to define the post-breakout
    retest window."""
    if df_window is None or len(df_window) == 0:
        return None
    closes = df_window["close"].to_numpy()
    n = len(closes)
    start = max(0, int(start_index))
    if side == "BUY":
        for i in range(start, n):
            if closes[i] > trigger_price:
                return i
    elif side == "SELL":
        for i in range(start, n):
            if closes[i] < trigger_price:
                return i
    return None


def _detect_retest_and_confirm(
    *,
    side: str,
    df_window: pd.DataFrame | None,
    breakout_bar_index: int | None,
    trigger_price: float,
    candle_review: dict | None,
    atr: float | None,
) -> tuple[bool, str]:
    """Heuristic retest detection.

    A retest is considered confirmed when, AFTER the FIRST breakout
    bar (close crosses trigger), at some later bar the price comes
    back to the trigger line within ATR×0.5, AND the most recent
    classified bar is bullish (BUY) / bearish (SELL).

    The "FIRST breakout bar" is recomputed from df even if
    `breakout_bar_index` (from skeleton) points to a later peak —
    this is the only way to surface retests on synthetic curves
    where the skeleton anchors BR to the end of the post-breakout
    move.

    Returns (retest_confirmed, confirmation_candle_label).
    """
    if df_window is None or len(df_window) == 0:
        return False, ""

    # Locate true first breakout bar. start_index = the pre-breakout
    # anchor (B2/P2/RS) so we don't pick a much-earlier crossover.
    first_break = _find_first_breakout_bar(
        side=side, df_window=df_window, trigger_price=trigger_price,
        start_index=int(breakout_bar_index) if breakout_bar_index is not None else 0,
    )
    if first_break is None:
        return False, ""
    if first_break >= len(df_window) - 1:
        return False, ""
    after = df_window.iloc[first_break + 1:]
    if len(after) == 0:
        return False, ""
    a = float(atr) if atr is not None and atr > 0 else max(
        float(after["high"].max() - after["low"].min()) * 0.05, 1e-9,
    )
    band = a * 0.5
    if side == "BUY":
        # Retest: post-breakout, price came back DOWN to within band of trigger
        touched = bool(((after["low"] <= trigger_price + band)
                        & (after["low"] >= trigger_price - 2 * band)).any())
    elif side == "SELL":
        # Retest: post-breakout, price came back UP to within band of trigger
        touched = bool(((after["high"] >= trigger_price - band)
                        & (after["high"] <= trigger_price + 2 * band)).any())
    else:
        return False, ""
    if not touched:
        return False, ""

    # Look at most-recent candle anatomy
    bar_type = (candle_review or {}).get("bar_type") or ""
    direction = (candle_review or {}).get("direction") or ""
    if side == "BUY":
        if bar_type in _BULLISH_BARS or direction == "BUY":
            return True, bar_type or "bullish_candle"
    elif side == "SELL":
        if bar_type in _BEARISH_BARS or direction == "SELL":
            return True, bar_type or "bearish_candle"
    return False, ""


def build_entry_plan(
    *,
    pattern_levels: dict | None,
    candle_review: dict | None,
    df_window: pd.DataFrame | None,
    last_close: float | None,
    atr_value: float | None,
    min_rr: float = 2.0,
) -> dict:
    """Build the entry_plan_v1 dict from pattern_levels + recent bars.

    The function is the single source of truth for action gating in
    the integrated decision: if entry_status != READY, the integrated
    profile MUST emit HOLD with a clear reason in the audit.
    """
    if not pattern_levels or not pattern_levels.get("available"):
        return _empty_plan("missing_pattern_levels")

    side = (pattern_levels.get("side") or "NEUTRAL").upper()
    if side not in ("BUY", "SELL"):
        return _empty_plan("non_directional_side")

    trigger_id = pattern_levels.get("trigger_line_id")
    trigger_price = pattern_levels.get("trigger_line_price")
    stop_price = pattern_levels.get("stop_price")
    target_price = pattern_levels.get("target_price")
    target_ext = pattern_levels.get("target_extended_price")
    rr_extended = pattern_levels.get("rr_at_extended_target")
    breakout_confirmed = bool(pattern_levels.get("breakout_confirmed"))
    parts = pattern_levels.get("parts") or {}
    breakout_bar_index = (parts.get("BR") or {}).get("index")
    # Anchor the post-breakout retest search to the LAST pattern
    # constituent BEFORE the breakout (B2 / P2 / RS). The skeleton's
    # BR pivot can be the last-extreme of the post-breakout move
    # which leaves no bars after it; using the pre-BR anchor lets
    # us find the FIRST close that crosses the trigger.
    pre_break_anchor = None
    for key in ("B2", "P2", "RS"):
        v = parts.get(key)
        if v and v.get("index") is not None:
            pre_break_anchor = int(v["index"])
            break

    block_reasons: list[str] = []

    if trigger_price is None:
        block_reasons.append("missing_trigger_line")
    if stop_price is None:
        block_reasons.append("missing_stop")
    if target_price is None:
        block_reasons.append("missing_target")
    if rr_extended is None or float(rr_extended) < min_rr:
        block_reasons.append(f"rr_below_{min_rr}")

    # State machine (in order: HOLD on missing → WAIT_BREAKOUT →
    # WAIT_RETEST → READY).
    if (
        trigger_price is None
        or stop_price is None
        or target_price is None
    ):
        return {
            "schema_version": SCHEMA_VERSION,
            "side": side,
            "entry_type": "none",
            "entry_status": "HOLD",
            "trigger_line_id": trigger_id,
            "trigger_line_price": trigger_price,
            "entry_price": None,
            "stop_price": stop_price,
            "target_price": target_price,
            "target_extended_price": target_ext,
            "rr": float(rr_extended) if rr_extended is not None else None,
            "breakout_confirmed": breakout_confirmed,
            "retest_confirmed": False,
            "confirmation_candle": "",
            "reason_ja": (
                "波形の主要部位 (NL/SL/TP) が揃っていません。HOLD。"
            ),
            "what_to_wait_for_ja": (
                "波形が完成し、ネックライン / 損切り / 利確が引けるまで待ちます。"
            ),
            "block_reasons": block_reasons,
        }

    if not breakout_confirmed:
        return {
            "schema_version": SCHEMA_VERSION,
            "side": side,
            "entry_type": "breakout",
            "entry_status": "WAIT_BREAKOUT",
            "trigger_line_id": trigger_id,
            "trigger_line_price": trigger_price,
            "entry_price": None,
            "stop_price": stop_price,
            "target_price": target_price,
            "target_extended_price": target_ext,
            "rr": float(rr_extended) if rr_extended is not None else None,
            "breakout_confirmed": False,
            "retest_confirmed": False,
            "confirmation_candle": "",
            "reason_ja": (
                f"波形 ({pattern_levels.get('pattern_kind')}) を認識中ですが、"
                f"トリガーライン {trigger_id} ({trigger_price:.5f}) は未突破です。"
            ),
            "what_to_wait_for_ja": (
                f"トリガーライン {trigger_id} ({trigger_price:.5f}) の"
                f"{'上抜け' if side == 'BUY' else '下抜け'}と"
                f"確定足を待ちます。"
            ),
            "block_reasons": block_reasons + ["wnl_not_broken"],
        }

    # Breakout confirmed — check retest
    retest_ok, conf_candle = _detect_retest_and_confirm(
        side=side,
        df_window=df_window,
        breakout_bar_index=pre_break_anchor or breakout_bar_index,
        trigger_price=float(trigger_price),
        candle_review=candle_review,
        atr=atr_value,
    )

    if not retest_ok:
        return {
            "schema_version": SCHEMA_VERSION,
            "side": side,
            "entry_type": "retest",
            "entry_status": "WAIT_RETEST",
            "trigger_line_id": trigger_id,
            "trigger_line_price": trigger_price,
            "entry_price": None,
            "stop_price": stop_price,
            "target_price": target_price,
            "target_extended_price": target_ext,
            "rr": float(rr_extended) if rr_extended is not None else None,
            "breakout_confirmed": True,
            "retest_confirmed": False,
            "confirmation_candle": "",
            "reason_ja": (
                f"トリガーライン {trigger_id} は突破済みですが、"
                "リターンムーブ (押し戻し) と確定足が確認できていません。"
            ),
            "what_to_wait_for_ja": (
                f"{trigger_id} 付近への戻りと"
                f"{'反発' if side == 'BUY' else '反落'}の確定足を待ちます。"
            ),
            "block_reasons": block_reasons + ["awaiting_retest_confirmation"],
        }

    if rr_extended is None or float(rr_extended) < min_rr:
        return {
            "schema_version": SCHEMA_VERSION,
            "side": side,
            "entry_type": "retest",
            "entry_status": "HOLD",
            "trigger_line_id": trigger_id,
            "trigger_line_price": trigger_price,
            "entry_price": None,
            "stop_price": stop_price,
            "target_price": target_price,
            "target_extended_price": target_ext,
            "rr": float(rr_extended) if rr_extended is not None else None,
            "breakout_confirmed": True,
            "retest_confirmed": True,
            "confirmation_candle": conf_candle,
            "reason_ja": (
                f"リターンムーブと確定足は確認できましたが、"
                f"RR={rr_extended:.2f} が必要値 {min_rr} 未満。HOLD。"
                if rr_extended is not None else
                f"RR が計算できません。HOLD。"
            ),
            "what_to_wait_for_ja": (
                "より深い押し戻し / 戻り (RR が改善する局面) を待ちます。"
            ),
            "block_reasons": block_reasons,
        }

    # READY
    entry_price = (
        float(last_close) if last_close is not None
        else float(trigger_price)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "side": side,
        "entry_type": "retest",
        "entry_status": "READY",
        "trigger_line_id": trigger_id,
        "trigger_line_price": trigger_price,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "target_extended_price": target_ext,
        "rr": float(rr_extended),
        "breakout_confirmed": True,
        "retest_confirmed": True,
        "confirmation_candle": conf_candle,
        "reason_ja": (
            f"トリガーライン {trigger_id} 突破 + リターンムーブ確認 + "
            f"確定足 ({conf_candle}) + RR={rr_extended:.2f}。"
            f"{side} エントリー条件を満たしています。"
        ),
        "what_to_wait_for_ja": (
            "条件は揃っています。実取引前に最終確認 "
            "(レンジ拘束 / 経済指標 / セッション) を済ませてください。"
        ),
        "block_reasons": [],
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_entry_plan",
]
