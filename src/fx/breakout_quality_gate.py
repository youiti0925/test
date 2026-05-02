"""breakout_quality_gate_v1 — 3-condition breakout quality check.

The High_Probability_Breakout_Blueprint resource specifies three
ingredients for a high-probability breakout:

  1. Build-up      — pre-breakout price compression near the trigger
                     line (BUY: lows climb up to the line; SELL: highs
                     climb down to the line).
  2. HTF alignment — higher-timeframe trend confirms the breakout
                     direction.
  3. Stop-loss accumulation
                   — equal highs / equal lows BEYOND the line
                     suggesting trapped stops on the wrong side.

The integrated decision uses this gate as a P0 component: a BLOCK
status forces HOLD even if every other condition passes (no chasing
breakouts without quality).

Output schema:

  {
    "schema_version": "breakout_quality_gate_v1",
    "side": "BUY" / "SELL" / "NEUTRAL",
    "status": "PASS" | "WARN" | "BLOCK",
    "build_up_status": "PASS" / "WARN" / "BLOCK",
    "trend_alignment_status": "PASS" / "WARN" / "BLOCK",
    "stop_loss_accumulation_status": "PASS" / "WARN" / "BLOCK",
    "reason_ja": str,
    "what_to_check_on_chart_ja": str,
  }
"""
from __future__ import annotations

from typing import Final

import pandas as pd


SCHEMA_VERSION: Final[str] = "breakout_quality_gate_v1"


def _empty_gate(side: str = "NEUTRAL", reason: str = "no_pattern") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "side": side,
        "status": "WARN",
        "build_up_status": "WARN",
        "trend_alignment_status": "WARN",
        "stop_loss_accumulation_status": "WARN",
        "reason_ja": "ブレイクアウト品質を判定する材料が不足しています。",
        "what_to_check_on_chart_ja": "波形の認識と上位足のトレンドを再確認してください。",
        "unavailable_reason": reason,
    }


def _classify_build_up(
    *,
    side: str,
    df_window: pd.DataFrame | None,
    trigger_price: float | None,
    breakout_bar_index: int | None,
    atr: float | None,
    lookback: int = 12,
) -> tuple[str, str]:
    """Detect compression near the trigger line BEFORE the breakout.

    BUY build-up: in the `lookback` bars before breakout, the lows
    should climb (slope > 0) AND the highs should be capped near the
    trigger line (max high within ATR×0.5 of trigger).

    SELL build-up: lows are supported near trigger, highs descend.
    """
    if (
        df_window is None or len(df_window) == 0
        or trigger_price is None or breakout_bar_index is None
    ):
        return "WARN", "ビルドアップ評価に必要な情報が不足。"
    end = max(0, int(breakout_bar_index))
    start = max(0, end - lookback)
    pre = df_window.iloc[start:end]
    if len(pre) < 4:
        return "WARN", "ブレイク前の足数が少なくビルドアップ判定不可。"
    a = float(atr) if atr is not None and atr > 0 else max(
        float(pre["high"].max() - pre["low"].min()) * 0.05, 1e-9,
    )
    band = a * 0.5

    highs = pre["high"].to_numpy()
    lows = pre["low"].to_numpy()
    n = len(pre)

    if side == "BUY":
        first_lows = lows[: n // 2].mean() if n // 2 > 0 else lows[0]
        last_lows = lows[n // 2:].mean() if n - n // 2 > 0 else lows[-1]
        lows_climbing = bool(last_lows > first_lows)
        capped_near_trigger = bool(
            highs.max() <= trigger_price + band
            and highs.max() >= trigger_price - 2 * band
        )
        if lows_climbing and capped_near_trigger:
            return "PASS", (
                "WNL/WBR 直下で安値が切り上がり、高値はライン付近で抑えられています。"
            )
        if capped_near_trigger:
            return "WARN", (
                "高値はライン付近で抑えられていますが、安値の切り上げが弱いです。"
            )
        return "WARN", "明確な圧縮 (ビルドアップ) は確認できません。"

    if side == "SELL":
        first_highs = highs[: n // 2].mean() if n // 2 > 0 else highs[0]
        last_highs = highs[n // 2:].mean() if n - n // 2 > 0 else highs[-1]
        highs_descending = bool(last_highs < first_highs)
        capped_near_trigger = bool(
            lows.min() >= trigger_price - band
            and lows.min() <= trigger_price + 2 * band
        )
        if highs_descending and capped_near_trigger:
            return "PASS", (
                "WNL/WBR 直上で高値が切り下がり、安値はライン付近で支えられています。"
            )
        if capped_near_trigger:
            return "WARN", (
                "安値はライン付近で支えられていますが、高値の切り下げが弱いです。"
            )
        return "WARN", "明確な圧縮 (ビルドアップ) は確認できません。"

    return "WARN", "side=NEUTRAL のため判定保留。"


def _classify_trend_alignment(
    *, side: str, higher_tf_trend: str | None,
) -> tuple[str, str]:
    """BUY needs HTF UP; SELL needs HTF DOWN. RANGE / TRANSITION is
    WARN. Counter-trend (BUY against DOWN) is BLOCK."""
    htf = (higher_tf_trend or "UNKNOWN").upper()
    if side == "BUY":
        if htf in ("UP", "UPTREND"):
            return "PASS", f"上位足 {htf} と買い方向が一致。"
        if htf in ("DOWN", "DOWNTREND"):
            return "BLOCK", (
                f"上位足 {htf} に逆らう買いです。転換確認なしの逆行はブロック。"
            )
        return "WARN", f"上位足 {htf}。trend が明確でなく caution。"
    if side == "SELL":
        if htf in ("DOWN", "DOWNTREND"):
            return "PASS", f"上位足 {htf} と売り方向が一致。"
        if htf in ("UP", "UPTREND"):
            return "BLOCK", (
                f"上位足 {htf} に逆らう売りです。転換確認なしの逆行はブロック。"
            )
        return "WARN", f"上位足 {htf}。trend が明確でなく caution。"
    return "WARN", "side=NEUTRAL のため判定保留。"


def _classify_stop_loss_accumulation(
    *,
    side: str,
    df_window: pd.DataFrame | None,
    trigger_price: float | None,
    breakout_bar_index: int | None,
    atr: float | None,
    lookback: int = 60,
) -> tuple[str, str]:
    """Estimate stop-loss accumulation by counting equal highs (BUY)
    or equal lows (SELL) in the period before breakout.

    BUY: equal highs near the trigger line suggest sellers' stops sit
    just above. PASS when ≥3 highs cluster within ATR×0.3 of the
    trigger.

    SELL: equal lows clustering below the trigger.
    """
    if (
        df_window is None or len(df_window) == 0
        or trigger_price is None or breakout_bar_index is None
    ):
        return "WARN", "損切り蓄積の構造推定に必要な足数がありません。"
    end = max(0, int(breakout_bar_index))
    start = max(0, end - lookback)
    pre = df_window.iloc[start:end]
    if len(pre) < 8:
        return "WARN", "ブレイク前の足数が少なく構造推定不可。"
    a = float(atr) if atr is not None and atr > 0 else max(
        float(pre["high"].max() - pre["low"].min()) * 0.05, 1e-9,
    )
    cluster_band = a * 0.3
    highs = pre["high"].to_numpy()
    lows = pre["low"].to_numpy()

    if side == "BUY":
        # Count highs that touched the trigger ± cluster_band
        touched = sum(
            1 for h in highs
            if (trigger_price - cluster_band) <= h <= (trigger_price + cluster_band)
        )
        if touched >= 3:
            return "PASS", (
                f"WNL 付近で {touched} 回上昇を抑えられています。"
                "売り方のストップが上に蓄積されている可能性があります。"
            )
        if touched >= 2:
            return "WARN", "WNL 付近で2回抑えられた程度。蓄積はやや弱い。"
        return "WARN", (
            "WNL 付近で抑えられた回数が少なく、損切り蓄積の構造推定は弱い。"
        )

    if side == "SELL":
        touched = sum(
            1 for l in lows
            if (trigger_price - cluster_band) <= l <= (trigger_price + cluster_band)
        )
        if touched >= 3:
            return "PASS", (
                f"WNL 付近で {touched} 回下落を支えられています。"
                "買い方のストップが下に蓄積されている可能性があります。"
            )
        if touched >= 2:
            return "WARN", "WNL 付近で2回支えられた程度。蓄積はやや弱い。"
        return "WARN", "WNL 付近で支えられた回数が少なく、構造推定は弱い。"

    return "WARN", "side=NEUTRAL のため判定保留。"


def _aggregate(
    build_up: str, trend_align: str, stop_loss: str,
) -> str:
    """Aggregate the three sub-statuses.

    BLOCK if any sub is BLOCK (any single condition can disqualify).
    PASS if all three are PASS.
    WARN otherwise.
    """
    statuses = (build_up, trend_align, stop_loss)
    if any(s == "BLOCK" for s in statuses):
        return "BLOCK"
    if all(s == "PASS" for s in statuses):
        return "PASS"
    return "WARN"


def build_breakout_quality_gate(
    *,
    side: str,
    pattern_levels: dict | None,
    df_window: pd.DataFrame | None,
    higher_tf_trend: str | None,
    atr_value: float | None,
) -> dict:
    """Build the breakout_quality_gate_v1 panel."""
    if not pattern_levels or not pattern_levels.get("available"):
        return _empty_gate(side=side, reason="missing_pattern_levels")
    if side not in ("BUY", "SELL"):
        return _empty_gate(side=side, reason="non_directional_side")

    trigger_price = pattern_levels.get("trigger_line_price")
    breakout_bar_index = (pattern_levels.get("parts") or {}).get("BR", {}).get("index")

    bu_status, bu_reason = _classify_build_up(
        side=side, df_window=df_window,
        trigger_price=trigger_price,
        breakout_bar_index=breakout_bar_index,
        atr=atr_value,
    )
    ta_status, ta_reason = _classify_trend_alignment(
        side=side, higher_tf_trend=higher_tf_trend,
    )
    sl_status, sl_reason = _classify_stop_loss_accumulation(
        side=side, df_window=df_window,
        trigger_price=trigger_price,
        breakout_bar_index=breakout_bar_index,
        atr=atr_value,
    )
    overall = _aggregate(bu_status, ta_status, sl_status)

    if overall == "PASS":
        reason_ja = (
            f"3条件 (build-up / 上位足一致 / 損切り蓄積) すべて PASS。{side} 高確率ブレイクアウト。"
        )
    elif overall == "BLOCK":
        reason_ja = (
            f"3条件のうち一部が BLOCK。high-probability breakout の条件を"
            f"満たしません。"
        )
    else:
        reason_ja = (
            f"3条件のうち一部が WARN。条件不十分のため caution として扱います。"
        )

    what_to_check_ja = (
        f"build-up: {bu_reason}\n"
        f"trend alignment: {ta_reason}\n"
        f"stop-loss accumulation: {sl_reason}"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "side": side,
        "status": overall,
        "build_up_status": bu_status,
        "build_up_reason_ja": bu_reason,
        "trend_alignment_status": ta_status,
        "trend_alignment_reason_ja": ta_reason,
        "stop_loss_accumulation_status": sl_status,
        "stop_loss_accumulation_reason_ja": sl_reason,
        "reason_ja": reason_ja,
        "what_to_check_on_chart_ja": what_to_check_ja,
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_breakout_quality_gate",
]
