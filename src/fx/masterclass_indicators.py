"""Masterclass indicator panels (observation-only).

Seven panels in one file (related concerns: how to use indicators
in different market regimes):

  indicator_environment_router  — TREND vs RANGE → preferred /
                                  deprioritized indicator list
  ma_context_review              — SMA/EMA slope, position relative
                                  to price, pullback / 戻り売り /
                                  乖離
  granville_entry_review         — Granville's eight rules
                                  (uptrend pullback / downtrend
                                  rebound / trap warnings)
  bollinger_lifecycle_review     — Squeeze / Expansion / Band Walk
                                  / Reversal risk
  rsi_regime_filter              — overbought / oversold gated by
                                  regime (Masterclass: RSI traps
                                  in trending markets)
  divergence_review              — bullish / bearish / hidden
                                  divergence — single-source
                                  signal NOT for entry alone
  macd_architecture_review       — MACD / Signal / Histogram /
                                  Zero / Golden Cross / Dead Cross

Each panel returns a dict with:
  schema_version, available, observation_only=True,
  used_in_decision=False, + panel-specific fields and JA text.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


ENV_ROUTER_SCHEMA: Final[str] = "indicator_environment_router_v1"
MA_CONTEXT_SCHEMA: Final[str] = "ma_context_review_v1"
GRANVILLE_SCHEMA: Final[str] = "granville_entry_review_v1"
BB_SCHEMA: Final[str] = "bollinger_lifecycle_review_v1"
RSI_SCHEMA: Final[str] = "rsi_regime_filter_v1"
DIVERGENCE_SCHEMA: Final[str] = "divergence_review_v1"
MACD_SCHEMA: Final[str] = "macd_architecture_review_v1"


def _empty(schema: str, reason: str) -> dict:
    return {
        "schema_version": schema,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
    }


# ---------------------------------------------------------------------------
# 1. indicator_environment_router
# ---------------------------------------------------------------------------


def build_indicator_environment_router(
    *,
    market_regime: str | None,
    bb_squeeze: bool = False,
    bb_expansion: bool = False,
) -> dict:
    """Map the current regime to preferred / deprioritized indicators.

    Masterclass intuition:
      - TREND → MA / MACD are reliable; RSI counter-signals are
        traps because price can stay overbought/oversold for long.
      - RANGE → RSI / Stochastics oscillator counter-signals work;
        MA crosses are noisy because of repeated whips.
    """
    if not market_regime:
        return _empty(ENV_ROUTER_SCHEMA, "missing_market_regime")
    regime_upper = str(market_regime).upper()
    is_trend = regime_upper.startswith("TREND")
    is_range = regime_upper.startswith("RANGE")
    if is_trend:
        preferred = ["MA_slope", "MA_pullback", "MACD_zero_cross"]
        deprioritized = ["RSI_reverse", "Stoch_reverse"]
        reason = (
            "トレンド環境のため、RSIの逆張りよりMA/MACDを優先します。"
        )
    elif is_range:
        preferred = ["RSI_overbought_oversold", "Stoch_OB_OS",
                     "BB_band_revert"]
        deprioritized = ["MA_cross", "MACD_cross_only"]
        reason = (
            "レンジ環境のため、MAクロスよりRSI / Stoch / BBの逆張りを"
            "優先します。"
        )
    else:
        preferred = []
        deprioritized = []
        reason = (
            "市場環境が判定できないため、特定インジケーターを優先しません。"
        )
    if bb_squeeze:
        reason += " BBスクイーズ中はブレイク方向待ちが王道です。"
    if bb_expansion:
        reason += " BBエクスパンション中はトレンド追随が王道です。"
    return {
        "schema_version": ENV_ROUTER_SCHEMA,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "market_regime": regime_upper,
        "is_trend_regime": bool(is_trend),
        "is_range_regime": bool(is_range),
        "preferred_indicators": preferred,
        "deprioritized_indicators": deprioritized,
        "reason_ja": reason,
    }


# ---------------------------------------------------------------------------
# 2. ma_context_review
# ---------------------------------------------------------------------------


def _slope_label_ja(slope_pct: float | None) -> str:
    if slope_pct is None:
        return "傾き不明"
    if slope_pct > 0.05:
        return "上向き (UP)"
    if slope_pct < -0.05:
        return "下向き (DOWN)"
    return "横ばい (FLAT)"


def build_ma_context_review(
    *,
    last_close: float | None,
    sma_20: float | None,
    sma_50: float | None,
    ema_12: float | None = None,
    sma_20_prev: float | None = None,
    sma_50_prev: float | None = None,
    atr_value: float | None = None,
) -> dict:
    """Slope / position / pullback / 戻り売り / 乖離 review for MA pair.

    Slopes are computed from `prev` snapshots when provided; absent
    that, the panel still describes price-vs-MA position.
    """
    if last_close is None or sma_20 is None or sma_50 is None:
        return _empty(MA_CONTEXT_SCHEMA, "missing_ma_inputs")

    sma20_slope = (
        ((sma_20 - sma_20_prev) / sma_20_prev * 100.0)
        if (sma_20_prev not in (None, 0)) else None
    )
    sma50_slope = (
        ((sma_50 - sma_50_prev) / sma_50_prev * 100.0)
        if (sma_50_prev not in (None, 0)) else None
    )
    price_above_20 = last_close > sma_20
    price_above_50 = last_close > sma_50
    sma20_above_sma50 = sma_20 > sma_50
    is_uptrend_stack = price_above_20 and price_above_50 and sma20_above_sma50
    is_downtrend_stack = (
        not price_above_20 and not price_above_50 and not sma20_above_sma50
    )
    pullback_to_20 = False
    rebound_to_20 = False
    if atr_value is not None and atr_value > 0:
        dist_20 = abs(last_close - sma_20) / atr_value
        if is_uptrend_stack and dist_20 <= 0.5 and last_close >= sma_20:
            pullback_to_20 = True
        if is_downtrend_stack and dist_20 <= 0.5 and last_close <= sma_20:
            rebound_to_20 = True
    # 乖離 (deviation): too far from MA → reversion candidate
    deviation_pct = (
        abs(last_close - sma_20) / sma_20 * 100.0 if sma_20 > 0 else 0.0
    )
    over_extended = deviation_pct > 1.5  # heuristic

    if is_uptrend_stack:
        stack_ja = "上昇配列 (Price > SMA20 > SMA50)"
    elif is_downtrend_stack:
        stack_ja = "下降配列 (Price < SMA20 < SMA50)"
    else:
        stack_ja = "MA配列が混在 (もみ合い)"
    summary_lines = [
        f"配列: {stack_ja}",
        f"SMA20傾き: {_slope_label_ja(sma20_slope)}",
        f"SMA50傾き: {_slope_label_ja(sma50_slope)}",
    ]
    if pullback_to_20:
        summary_lines.append(
            "SMA20への押し目状態 (上昇トレンドの押し目買い候補)"
        )
    elif rebound_to_20:
        summary_lines.append(
            "SMA20への戻り売り状態 (下降トレンドの戻り売り候補)"
        )
    if over_extended:
        summary_lines.append(
            f"乖離率 {deviation_pct:.2f}% は大きく、平均回帰リスクあり"
        )
    summary_ja = "; ".join(summary_lines)

    return {
        "schema_version": MA_CONTEXT_SCHEMA,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "last_close": float(last_close),
        "sma_20": float(sma_20),
        "sma_50": float(sma_50),
        "ema_12": (float(ema_12) if ema_12 is not None else None),
        "sma_20_slope_pct": (
            float(sma20_slope) if sma20_slope is not None else None
        ),
        "sma_50_slope_pct": (
            float(sma50_slope) if sma50_slope is not None else None
        ),
        "price_above_sma_20": bool(price_above_20),
        "price_above_sma_50": bool(price_above_50),
        "sma_20_above_sma_50": bool(sma20_above_sma50),
        "is_uptrend_stack": bool(is_uptrend_stack),
        "is_downtrend_stack": bool(is_downtrend_stack),
        "pullback_to_sma_20": bool(pullback_to_20),
        "rebound_to_sma_20": bool(rebound_to_20),
        "deviation_pct": float(deviation_pct),
        "over_extended": bool(over_extended),
        "summary_ja": summary_ja,
    }


# ---------------------------------------------------------------------------
# 3. granville_entry_review
# ---------------------------------------------------------------------------


def build_granville_entry_review(
    *,
    last_close: float | None,
    sma_20: float | None,
    sma_20_prev: float | None = None,
    atr_value: float | None = None,
) -> dict:
    """Granville's eight rules — distilled to the four cleanest cases.

      trend_pullback_buy:    SMA up + price comes back to / touches SMA
      trend_pullback_sell:   SMA down + price comes back to / touches SMA
      reversal_buy:          flat→up SMA + price crosses above
      reversal_sell:         flat→down SMA + price crosses below
      trap_warning:          SMA is sloped opposite to direction of touch
    """
    if last_close is None or sma_20 is None:
        return _empty(GRANVILLE_SCHEMA, "missing_ma_inputs")
    slope = None
    if sma_20_prev not in (None, 0):
        slope = (sma_20 - sma_20_prev) / sma_20_prev
    direction_up = slope is not None and slope > 0.0005
    direction_down = slope is not None and slope < -0.0005
    direction_flat = not direction_up and not direction_down
    above = last_close >= sma_20
    below = last_close <= sma_20
    touching = False
    if atr_value is not None and atr_value > 0:
        touching = abs(last_close - sma_20) / atr_value <= 0.3

    pattern = "neutral"
    meaning_ja = ""
    trap_ja = ""
    ma_dir = (
        "UP" if direction_up else
        "DOWN" if direction_down else "FLAT"
    )
    price_vs_ma = (
        "above" if above and not touching else
        "below" if below and not touching else
        "touching" if touching else "neutral"
    )
    if direction_up and touching:
        pattern = "trend_pullback_buy"
        meaning_ja = (
            "上向きMAへの押し目であり、買い候補 (Granville 第2則)。"
        )
        trap_ja = (
            "実際にはMAが下を向いていたら罠です。傾きを必ず確認してください。"
        )
    elif direction_down and touching:
        pattern = "trend_pullback_sell"
        meaning_ja = (
            "下向きMAへの戻りであり、戻り売り候補 (Granville 第6則)。"
        )
        trap_ja = (
            "実際にはMAが上を向いていたら罠です。傾きを必ず確認してください。"
        )
    elif direction_up and below:
        pattern = "trend_pullback_buy_deep"
        meaning_ja = (
            "上向きMAから乖離した深い押し。割安候補ですが、"
            "急騰後の反落リスクもあります。"
        )
    elif direction_down and above:
        pattern = "trend_pullback_sell_deep"
        meaning_ja = (
            "下向きMAから乖離した深い戻り。割高候補ですが、"
            "急落後の反発リスクもあります。"
        )
    elif direction_flat:
        pattern = "range_no_bias"
        meaning_ja = "MAが横ばいのため、Granville 法則は不安定です。"
    else:
        pattern = "neutral"
        meaning_ja = "MAとの位置関係から決定的な Granville パターンは出ていません。"

    return {
        "schema_version": GRANVILLE_SCHEMA,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "ma_direction": ma_dir,
        "price_vs_ma": price_vs_ma,
        "pattern": pattern,
        "meaning_ja": meaning_ja,
        "trap_warning_ja": trap_ja,
    }


# ---------------------------------------------------------------------------
# 4. bollinger_lifecycle_review
# ---------------------------------------------------------------------------


def build_bollinger_lifecycle_review(
    *,
    bb_squeeze: bool | None,
    bb_expansion: bool | None,
    bb_band_walk: bool | None,
    bb_position: float | None,
) -> dict:
    """Bollinger Band lifecycle stage."""
    if (
        bb_squeeze is None and bb_expansion is None
        and bb_band_walk is None
    ):
        return _empty(BB_SCHEMA, "missing_bb_inputs")
    stage = "neutral"
    meaning_ja = "BB ステージ判定は明確ではありません。"
    reversal_risk = False
    if bb_squeeze:
        stage = "squeeze"
        meaning_ja = (
            "Squeeze (バンド収縮)。レンジ末期、ブレイク待ちの段階です。"
            "ブレイクの方向待ちが王道です。"
        )
    elif bb_band_walk:
        stage = "band_walk"
        meaning_ja = (
            "Band Walk (バンド沿い歩き)。トレンド継続中の強い動きです。"
            "押し目買い・戻り売りが機能しやすい段階です。"
        )
    elif bb_expansion:
        stage = "expansion"
        meaning_ja = (
            "Expansion (バンド拡大)。スクイーズからのブレイク後で、"
            "勢いのある段階です。"
        )
    if bb_position is not None:
        if bb_position >= 0.95 or bb_position <= 0.05:
            reversal_risk = True
            meaning_ja += (
                " 価格がバンド端に達しており、平均回帰の可能性があります。"
            )
    return {
        "schema_version": BB_SCHEMA,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "stage": stage,
        "bb_squeeze": bool(bb_squeeze) if bb_squeeze is not None else None,
        "bb_expansion": (
            bool(bb_expansion) if bb_expansion is not None else None
        ),
        "bb_band_walk": (
            bool(bb_band_walk) if bb_band_walk is not None else None
        ),
        "bb_position": (
            float(bb_position) if bb_position is not None else None
        ),
        "reversal_risk": bool(reversal_risk),
        "meaning_ja": meaning_ja,
    }


# ---------------------------------------------------------------------------
# 5. rsi_regime_filter
# ---------------------------------------------------------------------------


def build_rsi_regime_filter(
    *,
    rsi_value: float | None,
    market_regime: str | None,
) -> dict:
    """RSI overbought / oversold gated by market regime.

    The Masterclass insight: in TREND, RSI overbought / oversold
    is a TRAP (price can stay extreme); only in RANGE does the
    counter-signal work.
    """
    if rsi_value is None:
        return _empty(RSI_SCHEMA, "missing_rsi")
    rsi = float(rsi_value)
    if rsi >= 70:
        raw_signal = "overbought"
    elif rsi <= 30:
        raw_signal = "oversold"
    else:
        raw_signal = "neutral"
    regime_upper = (
        str(market_regime).upper() if market_regime else "UNKNOWN"
    )
    is_trend = regime_upper.startswith("TREND")
    is_range = regime_upper.startswith("RANGE")
    if raw_signal == "neutral":
        usable = False
        trap_ja = ""
    elif is_trend:
        usable = False
        trap_ja = (
            "トレンド環境ではRSI買われすぎ/売られすぎ単独でのカウンターは"
            "危険です。"
        )
    elif is_range:
        usable = True
        trap_ja = ""
    else:
        usable = False
        trap_ja = (
            "市場環境が不明のため、RSI 逆張り単独は採用しません。"
        )
    return {
        "schema_version": RSI_SCHEMA,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "rsi_value": rsi,
        "regime": regime_upper,
        "raw_signal": raw_signal,
        "usable_signal": bool(usable),
        "trap_reason_ja": trap_ja,
    }


# ---------------------------------------------------------------------------
# 6. divergence_review
# ---------------------------------------------------------------------------


def build_divergence_review(
    *,
    rsi_bearish_divergence: bool | None = None,
    rsi_bullish_divergence: bool | None = None,
    macd_bearish_divergence: bool | None = None,
    macd_bullish_divergence: bool | None = None,
) -> dict:
    """Divergence summary panel.

    Masterclass: divergence alone is NOT an entry signal — pair
    with a structure / pattern confirmation.
    """
    flags = {
        "rsi_bullish": bool(rsi_bullish_divergence),
        "rsi_bearish": bool(rsi_bearish_divergence),
        "macd_bullish": bool(macd_bullish_divergence),
        "macd_bearish": bool(macd_bearish_divergence),
    }
    any_div = any(flags.values())
    if not any_div:
        return {
            "schema_version": DIVERGENCE_SCHEMA,
            "available": True,
            "observation_only": True,
            "used_in_decision": False,
            "any_divergence": False,
            **flags,
            "meaning_ja": "現在ダイバージェンスは検出されていません。",
            "warning_ja": (
                "ダイバージェンス単体ではエントリー根拠にしません。"
            ),
        }
    if flags["rsi_bullish"] or flags["macd_bullish"]:
        meaning = (
            "強気ダイバージェンス検出 (価格は安値更新だがオシレータは更新せず)。"
            "下落の勢いが弱まっている可能性があります。"
        )
    else:
        meaning = (
            "弱気ダイバージェンス検出 (価格は高値更新だがオシレータは更新せず)。"
            "上昇の勢いが弱まっている可能性があります。"
        )
    return {
        "schema_version": DIVERGENCE_SCHEMA,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "any_divergence": True,
        **flags,
        "meaning_ja": meaning,
        "warning_ja": (
            "ダイバージェンス単体ではエントリー根拠にしません。"
            "形・構造・サポレジでの確認が必要です。"
        ),
    }


# ---------------------------------------------------------------------------
# 7. macd_architecture_review
# ---------------------------------------------------------------------------


def build_macd_architecture_review(
    *,
    macd: float | None,
    macd_signal: float | None,
    macd_hist: float | None,
    macd_prev: float | None = None,
    macd_signal_prev: float | None = None,
    macd_hist_prev: float | None = None,
) -> dict:
    """MACD architecture: line vs signal, zero-line cross, hist sign,
    golden / dead cross detection.
    """
    if macd is None or macd_signal is None or macd_hist is None:
        return _empty(MACD_SCHEMA, "missing_macd")
    above_signal = macd > macd_signal
    above_zero = macd > 0
    hist_sign = (
        "positive" if macd_hist > 0 else
        "negative" if macd_hist < 0 else "zero"
    )
    cross = "none"
    if (
        macd_prev is not None and macd_signal_prev is not None
    ):
        prev_above = macd_prev > macd_signal_prev
        if above_signal and not prev_above:
            cross = "golden_cross"
        elif not above_signal and prev_above:
            cross = "dead_cross"
    momentum = "weakening"
    if macd_hist_prev is not None:
        if abs(macd_hist) > abs(macd_hist_prev):
            momentum = "strengthening"
        elif abs(macd_hist) < abs(macd_hist_prev):
            momentum = "weakening"
        else:
            momentum = "flat"
    bias = (
        "BUY" if (above_signal and above_zero and hist_sign == "positive")
        else "SELL" if (
            not above_signal and not above_zero and hist_sign == "negative"
        )
        else "NEUTRAL"
    )
    summary_parts: list[str] = []
    if cross == "golden_cross":
        summary_parts.append("ゴールデンクロス (MACDがシグナルを上抜け)")
    elif cross == "dead_cross":
        summary_parts.append("デッドクロス (MACDがシグナルを下抜け)")
    summary_parts.append(
        "ゼロライン上" if above_zero else "ゼロライン下"
    )
    summary_parts.append(
        f"ヒストグラム: {hist_sign}, 勢い: {momentum}"
    )
    summary_ja = " / ".join(summary_parts)

    return {
        "schema_version": MACD_SCHEMA,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "macd": float(macd),
        "macd_signal": float(macd_signal),
        "macd_hist": float(macd_hist),
        "above_signal": bool(above_signal),
        "above_zero": bool(above_zero),
        "hist_sign": hist_sign,
        "cross_event": cross,
        "momentum": momentum,
        "bias": bias,
        "summary_ja": summary_ja,
    }


__all__ = [
    "ENV_ROUTER_SCHEMA",
    "MA_CONTEXT_SCHEMA",
    "GRANVILLE_SCHEMA",
    "BB_SCHEMA",
    "RSI_SCHEMA",
    "DIVERGENCE_SCHEMA",
    "MACD_SCHEMA",
    "build_indicator_environment_router",
    "build_ma_context_review",
    "build_granville_entry_review",
    "build_bollinger_lifecycle_review",
    "build_rsi_regime_filter",
    "build_divergence_review",
    "build_macd_architecture_review",
]
