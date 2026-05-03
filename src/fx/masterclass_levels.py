"""Masterclass horizontal-level psychology review (observation-only).

Maps the existing support_resistance_v2 selected zones to a
Masterclass "level psychology" panel that names the current phase
relative to each level:

  initial_test   — first time price is touching the level
  breakout       — price has closed beyond the level
  pullback_bounce  — after a breakout, price came back and bounced
                    (role-reversal candidate)
  failed_breakout — price broke beyond but came back inside
  approaching    — price is near but hasn't tested
  far            — price is well away from the level

Each level emits:
  - level_id (e.g. "S1", "R1", "S2", ...)
  - phase
  - psychology_ja: 「一度上抜けた水平線に戻り、反発しているため、
                     レジサポ転換候補です。」
  - order_flow_ja: 「売り方の損切りと新規買いが入りやすい場所です。」

Strict invariants
-----------------
- observation_only=True, used_in_decision=False on every entry.
- Reads only the visible df (caller is responsible for slicing).
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "level_psychology_review_v1"


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
        "levels": [],
    }


def _classify_phase(
    *,
    last_close: float,
    zone_low: float,
    zone_high: float,
    body_break_count: int,
    role_reversal_count: int,
    false_breakout_count: int,
    touch_count: int,
    distance_to_close_atr: float | None,
    kind: str,  # "support" or "resistance"
) -> str:
    """Heuristic classifier of the current price's relationship to
    the level."""
    inside = zone_low <= last_close <= zone_high
    above = last_close > zone_high
    below = last_close < zone_low
    if inside:
        return "initial_test" if touch_count <= 1 else "retest"
    # Role-reversal / pullback
    if role_reversal_count > 0:
        if (kind == "support" and below) or (kind == "resistance" and above):
            return "failed_breakout"
        return "pullback_bounce"
    # Breakout
    if body_break_count > 0:
        if (kind == "support" and below) or (
            kind == "resistance" and above
        ):
            return "breakout"
        return "failed_breakout"
    # Approaching vs far
    if distance_to_close_atr is not None and distance_to_close_atr <= 1.0:
        return "approaching"
    return "far"


def _psychology_ja(
    phase: str, kind: str,
) -> tuple[str, str]:
    """Return (psychology_ja, order_flow_ja)."""
    if phase == "initial_test":
        if kind == "support":
            return (
                "サポート帯に初めて触れている状態です。"
                "反発するか割れるかを次の足で確認します。",
                "買い指値・売り損切りが集中しやすい価格帯です。",
            )
        return (
            "レジスタンス帯に初めて触れている状態です。"
            "反落するか上抜けるかを次の足で確認します。",
            "売り指値・買い損切りが集中しやすい価格帯です。",
        )
    if phase == "retest":
        if kind == "support":
            return (
                "サポート帯に複数回触れています。サポートとしての強度が"
                "増しているか、それとも崩壊間近かを見極める段階です。",
                "繰り返しの試しは、いずれ抜けたときに大きく動く傾向があります。",
            )
        return (
            "レジスタンス帯に複数回触れています。同様に強度確認の段階です。",
            "繰り返しの試しは、いずれ抜けたときに大きく動く傾向があります。",
        )
    if phase == "breakout":
        if kind == "support":
            return (
                "サポート帯を終値で下抜けました。レジサポ転換 (元サポートが"
                "新しいレジスタンス) の候補です。",
                "買い方の損切りが入り、戻り売りが入る場面です。",
            )
        return (
            "レジスタンス帯を終値で上抜けました。レジサポ転換 (元レジが"
            "新しいサポート) の候補です。",
            "売り方の損切りが入り、押し目買いが入る場面です。",
        )
    if phase == "pullback_bounce":
        if kind == "support":
            return (
                "一度下抜けた水平線に戻り、反発しているため、"
                "レジサポ転換候補です。",
                "売り方の損切りと新規買いが入りやすい場所です。",
            )
        return (
            "一度上抜けた水平線に戻り、反発しているため、"
            "レジサポ転換候補です。",
            "買い方の損切りと新規売りが入りやすい場所です。",
        )
    if phase == "failed_breakout":
        return (
            "一度抜けたが戻ってきています。騙し (failed breakout) の"
            "可能性があり、抜けた方向と逆向きに動きやすい場面です。",
            "ブレイク追随した参加者の損切りが連鎖しやすい価格帯です。",
        )
    if phase == "approaching":
        return (
            "水平線に接近中です。タッチ前後の値動きで強度を確認してください。",
            "ストップ・指値の集積帯に向けて流動性が薄くなる傾向があります。",
        )
    return (
        "水平線から離れているため、現時点では直接の根拠にしにくい位置です。",
        "ストップ集積帯から距離があり、急変リスクは相対的に低い場面です。",
    )


def build_level_psychology_review(
    *,
    overlays: dict | None,
    last_close: float | None,
) -> dict:
    """Build the level psychology panel from the existing
    `support_resistance_v2` selected zones (already in
    payload['overlays']).
    """
    ov = overlays or {}
    zones = list(ov.get("level_zones_selected") or [])
    if not zones:
        return _empty_panel("no_selected_zones")
    if last_close is None:
        return _empty_panel("missing_last_close")

    levels: list[dict] = []
    sup_count = 0
    res_count = 0
    for lvl in zones:
        kind = lvl.get("kind") or "level"
        if kind == "support":
            sup_count += 1
            level_id = f"S{sup_count}"
        elif kind == "resistance":
            res_count += 1
            level_id = f"R{res_count}"
        else:
            level_id = f"L{len(levels) + 1}"
        zlow = lvl.get("zone_low")
        zhigh = lvl.get("zone_high")
        if zlow is None or zhigh is None:
            continue
        phase = _classify_phase(
            last_close=last_close,
            zone_low=float(zlow),
            zone_high=float(zhigh),
            body_break_count=int(lvl.get("body_break_count") or 0),
            role_reversal_count=int(lvl.get("role_reversal_count") or 0),
            false_breakout_count=int(
                lvl.get("false_breakout_count") or 0
            ),
            touch_count=int(lvl.get("touch_count") or 0),
            distance_to_close_atr=lvl.get("distance_to_close_atr"),
            kind=kind,
        )
        psych_ja, order_ja = _psychology_ja(phase, kind)
        levels.append({
            "level_id": level_id,
            "kind": kind,
            "zone_low": float(zlow),
            "zone_high": float(zhigh),
            "price": lvl.get("price"),
            "touch_count": int(lvl.get("touch_count") or 0),
            "phase": phase,
            "psychology_ja": psych_ja,
            "order_flow_ja": order_ja,
            "observation_only": True,
            "used_in_decision": False,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "n_levels": len(levels),
        "levels": levels,
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_level_psychology_review",
]
