"""Masterclass invalidation engine v2 (observation-only).

Reframes the existing structure_stop_plan + entry_summary into a
Masterclass-flavoured "loss is where the basis fully breaks"
audit panel:

  setup_basis: list of strings — what the trade is built on
               (e.g. "double_bottom" + "support_zone" + "ma_pullback")
  invalidation_price: the price whose close-break invalidates ALL
                      of the basis simultaneously (currently the
                      v2 selected stop_price; for non-structure
                      stops, the panel notes that ATR-distance is
                      a workaround, not a true structural anchor).
  why_invalidates_ja: Japanese reason string.
  rr: reward / risk ratio.
  rr_pass: True when rr >= 2.0 (Masterclass minimum).

Strict invariants
-----------------
- observation_only=True, used_in_decision=False.
- The panel does NOT change the v2 stop selection — it only
  re-explains what the existing selection means in Masterclass
  terms.
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "invalidation_engine_v2"


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
    }


def build_invalidation_engine_v2(
    *,
    entry_summary: dict | None,
    pattern_review: dict | None,
    overlays: dict | None,
    dow_review: dict | None,
    ma_review: dict | None,
    invalidation_explanation: dict | None = None,
) -> dict:
    """Aggregate the inputs into a single invalidation panel."""
    es = entry_summary or {}
    if (
        es.get("entry_price") is None
        or es.get("stop_price") is None
    ):
        return _empty_panel("missing_entry_or_stop")

    setup_basis: list[str] = []
    if pattern_review and pattern_review.get("available"):
        kind = pattern_review.get("detected_pattern")
        if kind:
            setup_basis.append(kind)
    if (overlays or {}).get("level_zones_selected"):
        kinds = {
            l.get("kind")
            for l in (overlays or {}).get("level_zones_selected") or []
        }
        if "support" in kinds:
            setup_basis.append("support_zone")
        if "resistance" in kinds:
            setup_basis.append("resistance_zone")
    if dow_review and dow_review.get("trend") in ("UP", "DOWN"):
        setup_basis.append(f"dow_{dow_review['trend'].lower()}")
    if ma_review and (
        ma_review.get("pullback_to_sma_20")
        or ma_review.get("rebound_to_sma_20")
    ):
        setup_basis.append("ma_pullback_or_rebound")
    if not setup_basis:
        setup_basis.append("unspecified_basis")

    stop_price = float(es["stop_price"])
    structure_stop = es.get("structure_stop_price")
    atr_stop = es.get("atr_stop_price")
    is_structure = structure_stop is not None and abs(
        float(structure_stop) - stop_price
    ) < 1e-9

    why_lines: list[str] = []
    if is_structure:
        if "support_zone" in setup_basis:
            why_lines.append(
                f"{stop_price:.5f} を終値で下抜けると、サポート帯の根拠が崩れます。"
            )
        if "resistance_zone" in setup_basis:
            why_lines.append(
                f"{stop_price:.5f} を終値で上抜けると、レジスタンス帯の根拠が崩れます。"
            )
        if any(b in setup_basis for b in (
            "double_bottom", "head_and_shoulders",
            "inverse_head_and_shoulders", "double_top",
        )):
            why_lines.append("波形パターンの構造点を割るため、形そのものが無効化されます。")
        if any(b.startswith("dow_") for b in setup_basis):
            why_lines.append(
                "ダウ理論の押し安値 / 戻り高値を割るため、トレンド構造が崩れます。"
            )
    else:
        why_lines.append(
            "ATR距離の損切りであり、波形・サポレジが完全に崩れる場所ではなく"
            "距離ベースの workaround です。"
        )

    if invalidation_explanation and invalidation_explanation.get("available"):
        injected = invalidation_explanation.get(
            "why_this_stop_invalidates_the_setup"
        )
        if injected and injected not in why_lines:
            why_lines.append(injected)

    why_ja = "; ".join(why_lines) or (
        "損切り位置の意味付けが不明確です。"
    )

    rr = es.get("rr")
    rr_pass = isinstance(rr, (int, float)) and rr >= 2.0

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "setup_basis": setup_basis,
        "entry_price": float(es["entry_price"]),
        "stop_price": stop_price,
        "structure_stop_price": (
            float(structure_stop) if structure_stop is not None else None
        ),
        "atr_stop_price": (
            float(atr_stop) if atr_stop is not None else None
        ),
        "take_profit_price": (
            float(es["take_profit_price"])
            if es.get("take_profit_price") is not None else None
        ),
        "is_structure_anchored": bool(is_structure),
        "rr": (float(rr) if isinstance(rr, (int, float)) else None),
        "rr_pass": bool(rr_pass),
        "rr_minimum_required": 2.0,
        "why_invalidates_ja": why_ja,
        "philosophy_ja": (
            "損切りは、金額やATRではなく、根拠が完全に崩れる場所に置きます。"
            "RRは最低 1:2 です。"
        ),
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_invalidation_engine_v2",
]
