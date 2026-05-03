"""Source-pack Fibonacci context review (observation-only).

Maps the Fibonacci_Trading_Blueprint concepts onto the existing
wave skeleton:

  - Pick the most recent two opposite-kind pivots as the anchor
    swing (UP swing: low → high; DOWN swing: high → low).
  - Compute retracement levels (38.2% / 50.0% / 61.8%) and
    extension levels (127.2% / 161.8%) from that swing.
  - Classify the current price's position into one of:
      "0.0-38.2"  / "38.2-50.0"  / "50.0-61.8"  / "61.8-100"
      "above_anchor_high"  / "below_anchor_low"
      "approaching_127.2"  / "past_127.2"
      "approaching_161.8"  / "past_161.8"
  - Emit JA `meaning_ja` and `what_to_check_on_chart_ja`.

Companion helper `build_fib_wave_lines(...)` returns dict entries
in the same shape as `wave_derived_lines` so the audit's chart
overlay shows them as `WFIB382 / WFIB500 / WFIB618 / WFIB1272 /
WFIB1618`. Each line carries `used_in_decision = False`.

Strict invariants
-----------------
- observation_only=True, used_in_decision=False on every output.
- Reads only the supplied skeleton + last_close. Never peeks
  past `parent_bar_ts` (the skeleton encoder is already
  future-leak safe).
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "fibonacci_context_review_v1"


# Fibonacci levels used in this implementation.
RETRACEMENT_LEVELS: Final[tuple[float, ...]] = (0.382, 0.500, 0.618)
EXTENSION_LEVELS: Final[tuple[float, ...]] = (1.272, 1.618)


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "observation_only": True,
        "used_in_decision": False,
        "unavailable_reason": reason,
    }


def _pick_anchor_swing(skeleton: dict | None) -> tuple[dict, dict] | None:
    """Pick the two most recent opposite-kind pivots from the
    skeleton; returns (from_pivot, to_pivot) or None if unavailable.
    """
    pivots = (skeleton or {}).get("pivots") or []
    if len(pivots) < 2:
        return None
    # Use the last two pivots if they are opposite kinds; otherwise
    # walk back until we find an opposite-kind pair.
    last = pivots[-1]
    for p in reversed(pivots[:-1]):
        if p.get("kind") != last.get("kind"):
            return p, last
    return None


def _classify_zone(
    *, retracement_pct: float, has_extension: bool, side: str,
) -> str:
    """Classify current price position into a fibonacci zone label."""
    rp = retracement_pct
    if rp < 0.0:
        # Price moved beyond the anchor extreme → extension territory
        if abs(rp) >= 0.618:
            if abs(rp) >= 0.618:
                return (
                    "approaching_161.8" if abs(rp) < 0.618
                    else "past_161.8" if abs(rp) >= 0.618 else
                    "past_127.2"
                )
        return "past_127.2" if abs(rp) >= 0.272 else "approaching_127.2"
    if rp > 1.0:
        return (
            "below_anchor_low" if side == "DOWN"
            else "above_anchor_high"
        )
    if rp <= 0.382:
        return "0.0-38.2"
    if rp <= 0.500:
        return "38.2-50.0"
    if rp <= 0.618:
        return "50.0-61.8"
    return "61.8-100"


def build_fibonacci_context_review(
    *,
    skeleton: dict | None,
    last_close: float | None,
) -> dict:
    """Build the fibonacci_context_review panel.

    Returns `available=False` when no anchor swing is found or
    `last_close` is missing.
    """
    if last_close is None:
        return _empty_panel("missing_last_close")
    anchor = _pick_anchor_swing(skeleton)
    if anchor is None:
        return _empty_panel("no_anchor_swing")
    from_p, to_p = anchor
    try:
        from_price = float(from_p.get("price"))
        to_price = float(to_p.get("price"))
    except (TypeError, ValueError):
        return _empty_panel("invalid_anchor_swing")
    if from_price == to_price:
        return _empty_panel("flat_anchor_swing")
    side = "UP" if to_price > from_price else "DOWN"

    # retracement_pct: 0.0 = at the swing top (UP) / bottom (DOWN);
    # 1.0 = at the swing start; >1.0 = past the start; <0.0 = past
    # the swing extreme (extension territory).
    swing_size = to_price - from_price
    if side == "UP":
        retracement_pct = (to_price - float(last_close)) / swing_size
    else:
        # DOWN swing: from_price > to_price, swing_size < 0
        retracement_pct = (to_price - float(last_close)) / swing_size

    retracement_levels: list[dict] = []
    for r in RETRACEMENT_LEVELS:
        if side == "UP":
            price = to_price - r * abs(swing_size)
        else:
            price = to_price + r * abs(swing_size)
        retracement_levels.append({
            "level": f"{r * 100:.1f}",
            "price": float(price),
        })
    extension_levels: list[dict] = []
    for e in EXTENSION_LEVELS:
        if side == "UP":
            # Extension UP: price = to_price + (e - 1.0) * swing_size
            price = to_price + (e - 1.0) * abs(swing_size)
        else:
            price = to_price - (e - 1.0) * abs(swing_size)
        extension_levels.append({
            "level": f"{e * 100:.1f}",
            "price": float(price),
        })

    zone = _classify_zone(
        retracement_pct=retracement_pct,
        has_extension=True, side=side,
    )

    side_ja = "上昇波" if side == "UP" else "下降波"
    if zone in ("0.0-38.2", "38.2-50.0"):
        meaning_ja = (
            f"{side_ja}の浅い押し戻し ({zone}%)。"
            "押し目買い・戻り売りの初動候補です。"
        )
    elif zone == "50.0-61.8":
        meaning_ja = (
            f"{side_ja}の{int(retracement_pct * 100)}% 押し付近です。"
            "サポートやローソク足反発と合流すれば押し目候補です。"
        )
    elif zone == "61.8-100":
        meaning_ja = (
            f"{side_ja}の深い押し ({zone}%)。"
            "61.8% を割って深く戻ると、波形が崩れる可能性があります。"
        )
    elif zone in ("approaching_127.2", "approaching_161.8"):
        meaning_ja = (
            f"{side_ja}のブレイク後、エクステンションターゲットに接近中。"
        )
    elif zone in ("past_127.2", "past_161.8"):
        meaning_ja = (
            f"{side_ja}のエクステンションターゲットを既に超過しています。"
            "利確・反転リスクを意識する局面です。"
        )
    elif zone == "above_anchor_high" and side == "UP":
        meaning_ja = "上昇波のスタート以上に戻っており、波が無効化されています。"
    elif zone == "below_anchor_low" and side == "DOWN":
        meaning_ja = "下降波のスタート以下に戻っており、波が無効化されています。"
    else:
        meaning_ja = f"フィボ位置: {zone}。"

    what_to_check = (
        f"フィボ {zone} 付近で、サポレジ帯 / ローソク足反発 / MA との"
        "重なりがあるかを確認してください。"
    )

    # Status:
    # PASS — at 50.0-61.8 (王道の押し目買い / 戻り売りゾーン)
    # WARN — extension territory (already extended) or 0.0-38.2 (early)
    # BLOCK — past_anchor_extreme (wave invalidated)
    # UNKNOWN — otherwise
    if zone == "50.0-61.8":
        status = "PASS"
    elif zone in ("38.2-50.0", "61.8-100", "approaching_127.2",
                  "approaching_161.8"):
        status = "WARN"
    elif zone in ("above_anchor_high", "below_anchor_low",
                  "past_161.8"):
        status = "BLOCK"
    else:
        status = "UNKNOWN"

    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "observation_only": True,
        "used_in_decision": False,
        "status": status,
        "anchor_swing": {
            "from": {
                "price": from_price,
                "ts": from_p.get("ts"),
                "kind": from_p.get("kind"),
            },
            "to": {
                "price": to_price,
                "ts": to_p.get("ts"),
                "kind": to_p.get("kind"),
            },
            "side": side,
        },
        "current_close": float(last_close),
        "current_retracement_pct": float(retracement_pct),
        "retracement_zone": zone,
        "retracement_levels": retracement_levels,
        "extension_targets": extension_levels,
        "meaning_ja": meaning_ja,
        "what_to_check_on_chart_ja": what_to_check,
        "inputs_used": [
            "wave_skeleton.pivots (last 2 opposite-kind)",
            "visible_df.close (last)",
        ],
    }


def build_fib_wave_lines(
    *,
    fibonacci_review: dict | None,
) -> list[dict]:
    """Generate WFIB* horizontal lines compatible with the existing
    wave_derived_lines list (so they render on the chart with the
    same overlay machinery).

    Each line carries:
      id     = "WFIB382" / "WFIB500" / "WFIB618" / "WFIB1272" /
               "WFIB1618"
      kind   = "fibonacci_retracement" or "fibonacci_extension"
      role   = "fibonacci_zone"
      used_in_decision = False
    """
    if not fibonacci_review or not fibonacci_review.get("available"):
        return []
    out: list[dict] = []
    for lvl in fibonacci_review.get("retracement_levels") or []:
        try:
            level_num = float(lvl["level"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "id": f"WFIB{int(round(level_num * 10))}",
            "kind": "fibonacci_retracement",
            "source_pattern": "fibonacci_blueprint",
            "price": float(lvl["price"]),
            "zone_low": None,
            "zone_high": None,
            "role": "fibonacci_zone",
            "used_in_decision": False,
            "reason_ja": f"フィボナッチ {lvl['level']}% リトレースメント。",
        })
    for lvl in fibonacci_review.get("extension_targets") or []:
        try:
            level_num = float(lvl["level"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "id": f"WFIB{int(round(level_num * 10))}",
            "kind": "fibonacci_extension",
            "source_pattern": "fibonacci_blueprint",
            "price": float(lvl["price"]),
            "zone_low": None,
            "zone_high": None,
            "role": "fibonacci_zone",
            "used_in_decision": False,
            "reason_ja": f"フィボナッチ {lvl['level']}% エクステンション。",
        })
    return out


__all__ = [
    "SCHEMA_VERSION",
    "RETRACEMENT_LEVELS",
    "EXTENSION_LEVELS",
    "build_fibonacci_context_review",
    "build_fib_wave_lines",
]
