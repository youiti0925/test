"""Masterclass grand confluence v2 (observation-only).

Re-frames the existing v2 grand_confluence_checklist into the
9-axis Masterclass set:

  Dow            — dow_structure_review.trend
  Line           — selected SR / trendline / wave-derived line counts
  MA             — ma_context_review (slope + position + pullback)
  Oscillator     — rsi_regime_filter + macd_architecture_review
  Price Action   — candlestick_anatomy_review.bar_type
  Pattern        — chart_pattern_anatomy_v2.status
  Risk Reward    — entry_summary.rr (>= 2 is PASS)
  Invalidation   — entry_summary.stop_price + structure or atr
  Macro          — macro_alignment.macro_score (treated as
                   未実装/不明 by default — see masterclass_invalidation
                   description for the same disclaimer)

Each axis returns one of {"PASS", "WARN", "BLOCK", "UNKNOWN"} with
a Japanese reason. Output is observation-only and never gates the
final action.
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "grand_confluence_v2"


_AXES: Final[tuple[str, ...]] = (
    "dow", "line", "ma", "oscillator", "price_action",
    "pattern", "risk_reward", "invalidation", "macro",
)


def _axis(name: str, status: str, reason_ja: str) -> dict:
    return {
        "axis": name,
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
        "axes": [],
    }


def build_grand_confluence_v2(
    *,
    dow_review: dict | None,
    overlays: dict | None,
    wave_derived_lines: list[dict] | None,
    ma_review: dict | None,
    rsi_review: dict | None,
    macd_review: dict | None,
    candlestick_review: dict | None,
    pattern_review: dict | None,
    entry_summary: dict | None,
    macro_score: float | None = None,
) -> dict:
    """Aggregate the 9 axes into a single panel."""
    axes: list[dict] = []

    # Dow
    if not dow_review or not dow_review.get("available"):
        axes.append(_axis("dow", "UNKNOWN", "ダウ判定不可。"))
    else:
        trend = dow_review.get("trend", "UNKNOWN")
        if trend == "UP":
            axes.append(_axis("dow", "PASS", "上昇トレンド明確。"))
        elif trend == "DOWN":
            axes.append(_axis("dow", "PASS", "下降トレンド明確。"))
        elif trend in ("RANGE", "MIXED"):
            axes.append(_axis("dow", "WARN", f"トレンドが不明確 ({trend})。"))
        else:
            axes.append(_axis("dow", "UNKNOWN", "ダウ構造不足。"))

    # Line
    sr_sel = len((overlays or {}).get("level_zones_selected") or [])
    tl_sel = len((overlays or {}).get("trendlines_selected") or [])
    wd_total = len(wave_derived_lines or [])
    line_total = sr_sel + tl_sel + wd_total
    if line_total >= 3:
        axes.append(_axis(
            "line", "PASS",
            f"線が十分 ({sr_sel} S/R + {tl_sel} 線 + {wd_total} 波形由来)。",
        ))
    elif line_total >= 1:
        axes.append(_axis(
            "line", "WARN",
            f"線が少なめ ({line_total} 本)。",
        ))
    else:
        axes.append(_axis("line", "BLOCK", "線が引かれていません。"))

    # MA
    if not ma_review or not ma_review.get("available"):
        axes.append(_axis("ma", "UNKNOWN", "MA判定不可。"))
    elif ma_review.get("is_uptrend_stack") or ma_review.get("is_downtrend_stack"):
        axes.append(_axis("ma", "PASS", "MA配列が整っています。"))
    elif ma_review.get("over_extended"):
        axes.append(_axis(
            "ma", "WARN", "MA乖離が大きく、平均回帰リスクあり。",
        ))
    else:
        axes.append(_axis("ma", "WARN", "MA配列が混在しています。"))

    # Oscillator (RSI + MACD)
    osc_pass = False
    osc_warn = False
    osc_reasons: list[str] = []
    if rsi_review and rsi_review.get("available"):
        if rsi_review.get("usable_signal"):
            osc_pass = True
            osc_reasons.append(
                f"RSI {rsi_review.get('rsi_value'):.1f} を逆張りに使える環境。"
            )
        elif rsi_review.get("raw_signal") in ("overbought", "oversold"):
            osc_warn = True
            osc_reasons.append(
                rsi_review.get("trap_reason_ja") or "RSI 罠の可能性。"
            )
    if macd_review and macd_review.get("available"):
        bias = macd_review.get("bias")
        if bias in ("BUY", "SELL"):
            osc_pass = True
            osc_reasons.append(
                f"MACD は {bias} 寄り ({macd_review.get('summary_ja')})。"
            )
        else:
            osc_reasons.append("MACD はニュートラル。")
    if osc_pass and not osc_warn:
        axes.append(_axis(
            "oscillator", "PASS", " / ".join(osc_reasons) or "オシレータ整合。",
        ))
    elif osc_warn:
        axes.append(_axis(
            "oscillator", "WARN", " / ".join(osc_reasons) or "オシレータ警告。",
        ))
    else:
        axes.append(_axis(
            "oscillator", "UNKNOWN", " / ".join(osc_reasons) or "オシレータ判定不可。",
        ))

    # Price Action (candlestick)
    if not candlestick_review or not candlestick_review.get("available"):
        axes.append(_axis("price_action", "UNKNOWN", "ローソク足判定不可。"))
    else:
        bar_type = candlestick_review.get("bar_type", "")
        direction = candlestick_review.get("direction", "")
        if bar_type in ("bullish_pinbar", "bearish_pinbar",
                        "bullish_engulfing", "bearish_engulfing",
                        "large_bull", "large_bear"):
            axes.append(_axis(
                "price_action", "PASS",
                candlestick_review.get("meaning_ja", ""),
            ))
        elif bar_type in ("bullish_harami", "bearish_harami"):
            axes.append(_axis(
                "price_action", "WARN",
                candlestick_review.get("meaning_ja", ""),
            ))
        else:
            axes.append(_axis(
                "price_action", "UNKNOWN",
                candlestick_review.get("meaning_ja", ""),
            ))

    # Pattern
    if not pattern_review or not pattern_review.get("available"):
        axes.append(_axis("pattern", "UNKNOWN", "波形パターン判定不可。"))
    else:
        status = pattern_review.get("status", "")
        if status == "neckline_broken":
            axes.append(_axis(
                "pattern", "PASS",
                pattern_review.get("summary_ja", "ブレイク済み。"),
            ))
        elif status in ("forming", "neckline_not_broken"):
            axes.append(_axis(
                "pattern", "WARN",
                pattern_review.get("summary_ja", "形成中。"),
            ))
        elif status == "invalidated":
            axes.append(_axis(
                "pattern", "BLOCK",
                pattern_review.get("summary_ja", "形が崩れた。"),
            ))
        else:
            axes.append(_axis("pattern", "UNKNOWN", "形は弱い。"))

    # Risk Reward
    es = entry_summary or {}
    rr = es.get("rr")
    if rr is None:
        axes.append(_axis(
            "risk_reward", "UNKNOWN",
            f"RR算出不可 ({es.get('rr_unavailable_reason') or 'unknown'})。",
        ))
    elif rr >= 2.0:
        axes.append(_axis("risk_reward", "PASS", f"RR {rr:.2f} ≥ 2.0。"))
    elif rr >= 1.0:
        axes.append(_axis(
            "risk_reward", "WARN", f"RR {rr:.2f} は最低2に届きません。",
        ))
    else:
        axes.append(_axis(
            "risk_reward", "BLOCK", f"RR {rr:.2f} は損益比逆転。",
        ))

    # Invalidation
    if not es or es.get("stop_price") is None:
        axes.append(_axis(
            "invalidation", "UNKNOWN", "損切り設定なし。",
        ))
    else:
        if es.get("structure_stop_price") is not None:
            axes.append(_axis(
                "invalidation", "PASS",
                "構造損切り (structure_stop) が設定されています。",
            ))
        elif es.get("atr_stop_price") is not None:
            axes.append(_axis(
                "invalidation", "WARN",
                "ATR損切り (構造損切りなし) のため、波形根拠が崩れる場所"
                "ではなく距離ベースです。",
            ))
        else:
            axes.append(_axis(
                "invalidation", "WARN", "損切り根拠が不明。",
            ))

    # Macro — masterclass disclaimer: macro is observation-only and
    # currently 未実装/不明 by default.
    if macro_score is None:
        axes.append(_axis(
            "macro", "UNKNOWN",
            "ファンダ・マクロは未実装/不明。方向の根拠としては弱いです。",
        ))
    elif abs(macro_score) >= 0.5:
        axes.append(_axis(
            "macro", "PASS",
            f"マクロスコア {macro_score:+.2f} が方向側に整合。",
        ))
    elif abs(macro_score) >= 0.2:
        axes.append(_axis(
            "macro", "WARN",
            f"マクロスコア {macro_score:+.2f} は弱い。",
        ))
    else:
        axes.append(_axis(
            "macro", "UNKNOWN",
            f"マクロスコア {macro_score:+.2f} (NEUTRAL)。",
        ))

    # Aggregate
    pass_count = sum(1 for a in axes if a["status"] == "PASS")
    warn_count = sum(1 for a in axes if a["status"] == "WARN")
    block_count = sum(1 for a in axes if a["status"] == "BLOCK")
    unknown_count = sum(1 for a in axes if a["status"] == "UNKNOWN")
    label = (
        "STRONG_CONFLUENCE" if pass_count >= 6 and block_count == 0
        else "MODERATE_CONFLUENCE" if pass_count >= 3 and block_count == 0
        else "WEAK_CONFLUENCE" if block_count == 0
        else "BLOCKED"
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "axes": axes,
        "pass_count": pass_count,
        "warn_count": warn_count,
        "block_count": block_count,
        "unknown_count": unknown_count,
        "label": label,
        "summary_ja": (
            f"PASS {pass_count} / WARN {warn_count} / "
            f"BLOCK {block_count} / UNKNOWN {unknown_count}"
        ),
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_grand_confluence_v2",
]
