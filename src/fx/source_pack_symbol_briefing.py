"""Source-pack symbol macro briefing review (observation-only).

Maps the per-symbol Tactical Briefing concepts (especially the
USDJPY Tactical Briefing PDFs) onto an audit panel that lists the
relevant macro drivers. The current implementation surfaces:

  - macro_drivers: the PDFs' "watch this" list per symbol
  - bias: NEUTRAL by default (computed macro bias is NOT used in
    the v2 final action; it is observation-only here)
  - meaning_ja / warning_ja: JA explanation

For symbols whose briefing is not in the source pack (i.e. not
USDJPY/EURUSD/GBPUSD), a generic placeholder is emitted so the
audit reader knows the per-symbol fundamentals are not yet wired.

Strict invariants
-----------------
- observation_only=True, used_in_decision=False.
- The panel never feeds the v2 final action.
- For symbols with no real macro briefing data, returns
  `available=False, unavailable_reason="macro_briefing_data_missing"`
  with the JA notice.
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "symbol_macro_briefing_review_v1"


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
    }


# Per-symbol macro driver lists distilled from the source pack.
_SYMBOL_BRIEFING: Final[dict[str, dict]] = {
    "USDJPY=X": {
        "drivers": ["US_yield", "DXY", "JPY_risk_sentiment"],
        "summary_ja": (
            "USDJPYでは米10年金利・DXY (ドル指数)・"
            "リスク選好/回避 (株/VIX) が重要です。"
        ),
        "what_to_check_on_chart_ja": (
            "DXY と相関しているか、米金利と一緒に動いているか、"
            "リスクオフ局面で円高に動く構造か、を別チャートで確認してください。"
        ),
    },
    "USDJPY": {
        "drivers": ["US_yield", "DXY", "JPY_risk_sentiment"],
        "summary_ja": (
            "USDJPYでは米10年金利・DXY (ドル指数)・"
            "リスク選好/回避 (株/VIX) が重要です。"
        ),
        "what_to_check_on_chart_ja": (
            "DXY と相関しているか、米金利と一緒に動いているか、"
            "リスクオフ局面で円高に動く構造か、を別チャートで確認してください。"
        ),
    },
    "EURUSD=X": {
        "drivers": ["DXY", "EUR-USD_yield_spread", "ECB_policy"],
        "summary_ja": (
            "EURUSDでは DXY、EUR-USD金利差、ECB政策 が重要です。"
        ),
        "what_to_check_on_chart_ja": (
            "DXY と逆相関で動いているか、独米金利差の方向と整合しているかを"
            "別チャートで確認してください。"
        ),
    },
    "GBPUSD=X": {
        "drivers": ["DXY", "BOE_policy", "UK_inflation"],
        "summary_ja": (
            "GBPUSDでは DXY、英中銀政策、英CPI が重要です。"
        ),
        "what_to_check_on_chart_ja": (
            "DXYと逆相関で動いているか、英中銀のタカ/ハト姿勢が直近で変化していないかを"
            "確認してください。"
        ),
    },
}


def build_symbol_macro_briefing_review(
    *,
    symbol: str | None,
    data_available: bool = False,
) -> dict:
    """Build the symbol_macro_briefing_review panel.

    `symbol` is the FX pair label (e.g. "USDJPY=X" or "EURUSD=X").
    `data_available` indicates whether real macro/fundamental data
    is wired (currently always False in this codebase, so the panel
    explicitly emits `unavailable_reason="macro_briefing_data_missing"`).
    """
    if not symbol:
        return _empty_panel("missing_symbol")
    sym_upper = symbol.upper()
    briefing = _SYMBOL_BRIEFING.get(sym_upper)
    if briefing is None:
        # Generic placeholder for symbols not in the pack
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "observation_only": True,
            "used_in_decision": False,
            "status": "UNKNOWN",
            "symbol": symbol,
            "macro_drivers": [],
            "bias": "NEUTRAL",
            "unavailable_reason": "no_briefing_for_symbol",
            "meaning_ja": (
                f"{symbol} 用の専用マクロブリーフィングは資料パックに"
                "含まれていません。汎用ファンダ確認 (DXY / 金利 / リスク選好) を"
                "手動で行ってください。"
            ),
            "warning_ja": (
                "マクロブリーフィングが無い通貨ペアは、Tactical Briefing 由来の"
                "確認軸が無いため audit としては弱い状態です。"
            ),
            "what_to_check_on_chart_ja": (
                "通貨ペアごとの主要ドライバーを手動で別チャート確認してください。"
            ),
            "inputs_used": ["symbol (input)"],
        }

    if not data_available:
        # Briefing exists in source pack, but real macro data is not
        # wired yet — explicitly emit "macro_briefing_data_missing".
        return {
            "schema_version": SCHEMA_VERSION,
            "available": False,
            "observation_only": True,
            "used_in_decision": False,
            "status": "UNKNOWN",
            "symbol": symbol,
            "macro_drivers": briefing["drivers"],
            "bias": "NEUTRAL",
            "unavailable_reason": "macro_briefing_data_missing",
            "meaning_ja": (
                f"{symbol} 用のファンダ情報は資料上は重要ですが、"
                "現在の実データには未接続です。"
            ),
            "warning_ja": (
                "現状は Tactical Briefing の内容を audit 表示する段階で、"
                "売買判断には使っていません。"
            ),
            "what_to_check_on_chart_ja": briefing[
                "what_to_check_on_chart_ja"
            ],
            "inputs_used": ["symbol (input)"],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "status": "PASS",
        "symbol": symbol,
        "macro_drivers": briefing["drivers"],
        "bias": "NEUTRAL",
        "meaning_ja": briefing["summary_ja"],
        "warning_ja": (
            "マクロ判定はあくまで観測ですので、売買判断には使っていません。"
        ),
        "what_to_check_on_chart_ja": briefing["what_to_check_on_chart_ja"],
        "inputs_used": ["symbol (input)", "macro_briefing_pack"],
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_symbol_macro_briefing_review",
]
