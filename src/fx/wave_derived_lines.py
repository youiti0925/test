"""Derive entry / stop / target / pattern-part lines from a matched
waveform pattern (observation-only).

These are the "lines a human would draw when they see this pattern" —
they live alongside (NOT inside) the existing `support_resistance_v2`
and `trendline_context` outputs so the audit reviewer can compare:

  既存サポレジ / トレンドライン  vs  波形認識から引いた線

Strict invariants
-----------------
- Every line carries `used_in_decision = False`. This module is
  observation-only; the lines never feed `royal_road_decision_v2`'s
  final BUY/SELL/HOLD logic.
- Heuristic projections (target = NL ± pattern_height) are flagged
  via the `reason_ja` field so the user can judge whether the line
  is reasonable.

Output
------
`build_wave_derived_lines(...)` returns a list of dicts:
    {
      "id": "WNL1" / "WB1" / "WB2" / "WSL1" / "WTP1" / ...,
      "kind": "neckline" / "pivot_low" / "pivot_high" / "shoulder"
              / "head" / "pattern_invalidation" / "pattern_target"
              / "pattern_upper" / "pattern_lower" / "pattern_breakout",
      "source_pattern": "double_bottom" / "double_top" / ...,
      "price": float,
      "zone_low": float | None,
      "zone_high": float | None,
      "role": "entry_confirmation_line" / "pattern_part" /
              "stop_candidate" / "target_candidate" /
              "pattern_boundary" / "breakout_line",
      "used_in_decision": False,
      "reason_ja": str,
    }
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "wave_derived_lines_v1"


def _zone(price: float, atr: float, frac: float = 0.20) -> tuple[float | None, float | None]:
    if atr is None or atr <= 0:
        return None, None
    half = atr * frac
    return float(price) - half, float(price) + half


def _pivot_by_index(skeleton: dict) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in (skeleton or {}).get("pivots") or []:
        try:
            out[int(p.get("index"))] = p
        except (TypeError, ValueError):
            continue
    return out


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def build_wave_derived_lines(
    *,
    best_pattern: dict | None,
    skeleton: dict | None,
    atr_value: float | None = None,
) -> list[dict]:
    """Build the list of wave-derived lines for a single matched
    pattern.

    Returns [] when no best_pattern is available, when its skeleton
    has no pivots, or when the pattern kind is unknown.
    """
    if not best_pattern or not skeleton:
        return []
    pivots = (skeleton or {}).get("pivots") or []
    if not pivots:
        return []
    matched = best_pattern.get("matched_parts") or {}
    if not matched:
        return []
    kind = best_pattern.get("kind") or ""
    if not kind:
        return []
    by_idx = _pivot_by_index(skeleton)
    atr = float(atr_value or 0.0)
    if atr <= 0:
        atr = float(skeleton.get("atr_value") or 0.0)

    lines: list[dict] = []

    def _part_price(part_name: str) -> float | None:
        idx = matched.get(part_name)
        if idx is None:
            return None
        try:
            piv = by_idx.get(int(idx))
        except (TypeError, ValueError):
            return None
        if piv is None:
            return None
        return _safe_float(piv.get("price"))

    def _add_part(
        id_: str, kind_str: str, role: str,
        price: float, zone: tuple[float | None, float | None],
        reason_ja: str,
    ) -> None:
        lines.append({
            "id": id_,
            "kind": kind_str,
            "source_pattern": kind,
            "price": float(price),
            "zone_low": zone[0],
            "zone_high": zone[1],
            "role": role,
            "used_in_decision": False,
            "reason_ja": reason_ja,
        })

    # ------------------------------------------------------------------
    # double_bottom: WB1 / WB2 / WNL1 / WSL1 / WTP1
    # ------------------------------------------------------------------
    if kind == "double_bottom":
        b1 = _part_price("first_bottom")
        b2 = _part_price("second_bottom")
        nl = _part_price("neckline_peak")
        if b1 is not None:
            _add_part(
                "WB1", "pivot_low", "pattern_part",
                b1, _zone(b1, atr, 0.15),
                "ダブルボトム候補の1回目の底 (B1)。波形上の支持点。",
            )
        if b2 is not None:
            _add_part(
                "WB2", "pivot_low", "pattern_part",
                b2, _zone(b2, atr, 0.15),
                "ダブルボトム候補の2回目の底 (B2)。形が確定するための再試し。",
            )
        if nl is not None:
            _add_part(
                "WNL1", "neckline", "entry_confirmation_line",
                nl, _zone(nl, atr, 0.20),
                "ダブルボトム候補の中間高値から作ったネックラインです。"
                "ここを上抜けると買い候補が強まります。",
            )
        if b2 is not None:
            stop_offset = max(atr * 0.30, 0.0)
            stop_price = b2 - stop_offset
            _add_part(
                "WSL1", "pattern_invalidation", "stop_candidate",
                stop_price, (None, None),
                "2回目の底 (B2) を下抜けるとダブルボトム形状が崩れるため、"
                "波形由来の損切り候補です。",
            )
        if nl is not None and (b1 is not None or b2 is not None):
            base_low = min([v for v in (b1, b2) if v is not None])
            tp = nl + (nl - base_low)
            _add_part(
                "WTP1", "pattern_target", "target_candidate",
                tp, (None, None),
                "ネックライン突破後の目標候補。ネックラインから底までの幅を"
                "上方向に投影しています (heuristic projection)。",
            )

    # ------------------------------------------------------------------
    # double_top: WP1 / WP2 / WNL1 / WSL1 / WTP1
    # ------------------------------------------------------------------
    elif kind == "double_top":
        p1 = _part_price("first_top")
        p2 = _part_price("second_top")
        nl = _part_price("neckline_trough")
        if p1 is not None:
            _add_part(
                "WP1", "pivot_high", "pattern_part",
                p1, _zone(p1, atr, 0.15),
                "ダブルトップ候補の1回目の天井 (P1)。波形上の抵抗点。",
            )
        if p2 is not None:
            _add_part(
                "WP2", "pivot_high", "pattern_part",
                p2, _zone(p2, atr, 0.15),
                "ダブルトップ候補の2回目の天井 (P2)。形が確定するための再試し。",
            )
        if nl is not None:
            _add_part(
                "WNL1", "neckline", "entry_confirmation_line",
                nl, _zone(nl, atr, 0.20),
                "ダブルトップ候補の中間安値から作ったネックラインです。"
                "ここを下抜けると売り候補が強まります。",
            )
        if p2 is not None:
            stop_price = p2 + max(atr * 0.30, 0.0)
            _add_part(
                "WSL1", "pattern_invalidation", "stop_candidate",
                stop_price, (None, None),
                "2回目の天井 (P2) を上抜けるとダブルトップ形状が崩れるため、"
                "波形由来の損切り候補です。",
            )
        if nl is not None and (p1 is not None or p2 is not None):
            base_high = max([v for v in (p1, p2) if v is not None])
            tp = nl - (base_high - nl)
            _add_part(
                "WTP1", "pattern_target", "target_candidate",
                tp, (None, None),
                "ネックライン下抜け後の目標候補。天井からネックラインまでの幅を"
                "下方向に投影しています (heuristic projection)。",
            )

    # ------------------------------------------------------------------
    # head_and_shoulders: WLS / WH / WRS / WNL / WSL / WTP
    # ------------------------------------------------------------------
    elif kind == "head_and_shoulders":
        ls = _part_price("left_shoulder")
        head = _part_price("head")
        rs = _part_price("right_shoulder")
        nl_l = _part_price("neckline_left_trough")
        nl_r = _part_price("neckline_right_trough")
        nl = nl_l if nl_l is not None else nl_r
        if ls is not None:
            _add_part(
                "WLS", "shoulder", "pattern_part",
                ls, _zone(ls, atr, 0.15),
                "三尊候補の左肩 (LS)。",
            )
        if head is not None:
            _add_part(
                "WH", "head", "pattern_part",
                head, _zone(head, atr, 0.15),
                "三尊候補の頭 (H)。最も高いピーク。",
            )
        if rs is not None:
            _add_part(
                "WRS", "shoulder", "pattern_part",
                rs, _zone(rs, atr, 0.15),
                "三尊候補の右肩 (RS)。",
            )
        if nl is not None:
            _add_part(
                "WNL1", "neckline", "entry_confirmation_line",
                nl, _zone(nl, atr, 0.20),
                "三尊候補のネックライン (左右の谷を結んだ線の代表値)。"
                "ここを下抜けると売り候補が強まります。",
            )
        # Stop: above right shoulder (or above head if RS missing)
        stop_anchor = rs if rs is not None else head
        if stop_anchor is not None:
            stop_price = stop_anchor + max(atr * 0.30, 0.0)
            _add_part(
                "WSL1", "pattern_invalidation", "stop_candidate",
                stop_price, (None, None),
                "右肩を上抜けると三尊形状が崩れるため、波形由来の損切り候補です。",
            )
        # Target: NL - (head - NL)
        if nl is not None and head is not None:
            tp = nl - (head - nl)
            _add_part(
                "WTP1", "pattern_target", "target_candidate",
                tp, (None, None),
                "ネックライン割れ後の目標候補。頭からネックラインまでの幅を"
                "下方向に投影しています (heuristic projection)。",
            )

    # ------------------------------------------------------------------
    # inverse_head_and_shoulders: WLS / WH / WRS / WNL / WSL / WTP
    # ------------------------------------------------------------------
    elif kind == "inverse_head_and_shoulders":
        ls = _part_price("left_shoulder")
        head = _part_price("head")
        rs = _part_price("right_shoulder")
        nl_l = _part_price("neckline_left_peak")
        nl_r = _part_price("neckline_right_peak")
        nl = nl_l if nl_l is not None else nl_r
        if ls is not None:
            _add_part(
                "WLS", "shoulder", "pattern_part",
                ls, _zone(ls, atr, 0.15),
                "逆三尊候補の左肩 (LS)。",
            )
        if head is not None:
            _add_part(
                "WH", "head", "pattern_part",
                head, _zone(head, atr, 0.15),
                "逆三尊候補の頭 (H)。最も深い谷。",
            )
        if rs is not None:
            _add_part(
                "WRS", "shoulder", "pattern_part",
                rs, _zone(rs, atr, 0.15),
                "逆三尊候補の右肩 (RS)。",
            )
        if nl is not None:
            _add_part(
                "WNL1", "neckline", "entry_confirmation_line",
                nl, _zone(nl, atr, 0.20),
                "逆三尊候補のネックライン (左右のピークを結んだ線の代表値)。"
                "ここを上抜けると買い候補が強まります。",
            )
        stop_anchor = rs if rs is not None else head
        if stop_anchor is not None:
            stop_price = stop_anchor - max(atr * 0.30, 0.0)
            _add_part(
                "WSL1", "pattern_invalidation", "stop_candidate",
                stop_price, (None, None),
                "右肩を下抜けると逆三尊形状が崩れるため、波形由来の損切り候補です。",
            )
        if nl is not None and head is not None:
            tp = nl + (nl - head)
            _add_part(
                "WTP1", "pattern_target", "target_candidate",
                tp, (None, None),
                "ネックライン上抜け後の目標候補。頭からネックラインまでの幅を"
                "上方向に投影しています (heuristic projection)。",
            )

    # ------------------------------------------------------------------
    # flag / wedge / triangle: WUP / WLOW / WBR / WSL / WTP
    # ------------------------------------------------------------------
    elif kind in (
        "bullish_flag", "bearish_flag",
        "rising_wedge", "falling_wedge",
        "ascending_triangle", "descending_triangle", "symmetric_triangle",
    ):
        # Take the upper-anchor max and lower-anchor min as the
        # boundary lines; use breakout part as WBR; stop = back-inside
        # the channel; target = breakout +/- channel_width.
        upper_prices = []
        lower_prices = []
        for name in ("upper_anchor_1", "upper_anchor_2", "upper_anchor_3"):
            v = _part_price(name)
            if v is not None:
                upper_prices.append(v)
        for name in ("lower_anchor_1", "lower_anchor_2", "lower_anchor_3"):
            v = _part_price(name)
            if v is not None:
                lower_prices.append(v)
        upper_max = max(upper_prices) if upper_prices else None
        lower_min = min(lower_prices) if lower_prices else None
        if upper_max is not None:
            _add_part(
                "WUP", "pattern_upper", "pattern_boundary",
                upper_max, _zone(upper_max, atr, 0.15),
                f"{kind} の上限線 (上側アンカー高値)。",
            )
        if lower_min is not None:
            _add_part(
                "WLOW", "pattern_lower", "pattern_boundary",
                lower_min, _zone(lower_min, atr, 0.15),
                f"{kind} の下限線 (下側アンカー安値)。",
            )
        br = _part_price("breakout")
        if br is not None:
            _add_part(
                "WBR", "pattern_breakout", "breakout_line",
                br, _zone(br, atr, 0.15),
                f"{kind} のブレイクライン。ここを抜けると形のシナリオ通りに進みます。",
            )
        # Stop = back inside the channel (mid-point of upper/lower)
        if upper_max is not None and lower_min is not None:
            mid = (upper_max + lower_min) / 2.0
            _add_part(
                "WSL1", "pattern_invalidation", "stop_candidate",
                mid, (None, None),
                "ブレイク後にパターン内側 (上下限の中点) まで戻ると形が崩れるため、"
                "波形由来の損切り候補です。",
            )
            channel_width = upper_max - lower_min
            side = best_pattern.get("side_bias", "")
            if side == "BUY" and br is not None:
                tp = br + channel_width
                _add_part(
                    "WTP1", "pattern_target", "target_candidate",
                    tp, (None, None),
                    "ブレイク後の目標候補。チャネル幅を上方向に投影しています "
                    "(heuristic projection)。",
                )
            elif side == "SELL" and br is not None:
                tp = br - channel_width
                _add_part(
                    "WTP1", "pattern_target", "target_candidate",
                    tp, (None, None),
                    "ブレイク後の目標候補。チャネル幅を下方向に投影しています "
                    "(heuristic projection)。",
                )

    return lines


def line_count_summary(
    *,
    overlays: dict | None,
    wave_derived: list[dict] | None,
) -> dict:
    """Count existing SR / trendline candidates and wave-derived lines
    for the audit summary panel.
    """
    ov = overlays or {}
    wd = wave_derived or []
    sr_sel = len(ov.get("level_zones_selected") or [])
    sr_rej = len(ov.get("level_zones_rejected") or [])
    tl_sel = len(ov.get("trendlines_selected") or [])
    tl_rej = len(ov.get("trendlines_rejected") or [])
    wd_total = len(wd)
    wd_neckline = sum(1 for l in wd if l.get("kind") == "neckline")
    wd_stop = sum(1 for l in wd if l.get("role") == "stop_candidate")
    wd_target = sum(1 for l in wd if l.get("role") == "target_candidate")
    return {
        "sr_selected": int(sr_sel),
        "sr_rejected": int(sr_rej),
        "trendline_selected": int(tl_sel),
        "trendline_rejected": int(tl_rej),
        "wave_derived_total": int(wd_total),
        "wave_derived_neckline": int(wd_neckline),
        "wave_derived_stop": int(wd_stop),
        "wave_derived_target": int(wd_target),
    }


__all__ = [
    "SCHEMA_VERSION",
    "build_wave_derived_lines",
    "line_count_summary",
]
