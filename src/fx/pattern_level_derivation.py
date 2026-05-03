"""pattern_level_derivation_v1 — derive entry / stop / target / RR
from a recognized wave-shape pattern.

The integrated decision needs canonical part labels (B1, B2, NL, etc)
plus actionable price levels for entry / stop / take-profit / RR.
This module bridges:

  wave_shape_review.best_pattern (kind / side_bias / status /
                                  matched_parts: {role: pivot_index})
  +
  wave_derived_lines (list of {id, role, price, ...})

into a single `pattern_levels` dict the integrated engine can consume.

Strict invariants:
  - Observation-only. `used_in_decision` is set True ONLY when the
    integrated profile reads the dict; the dict itself never modifies
    any decision.
  - All prices are floats; missing levels become None (caller MUST
    treat None as "data unavailable" rather than guess).
  - This module does NOT consult future bars. Inputs are derived
    from wave_shape_review which is built from the visible window.

Output schema:
  {
    "schema_version": "pattern_levels_v1",
    "available": bool,
    "pattern_kind": str,           # double_bottom / head_and_shoulders / ...
    "side": "BUY" / "SELL" / "NEUTRAL",
    "status": str,                 # forming / neckline_broken / retested / invalidated
    "parts": {
        "B1" / "B2" / "P1" / "P2" / "LS" / "H" / "RS" / "NL" / "BR": {
            "price": float, "index": int,
        }
    },
    "trigger_line_id": str | None,   # WNL1 / WBR1
    "trigger_line_price": float | None,
    "stop_price": float | None,      # from WSL1
    "target_price": float | None,    # from WTP1
    "rr_at_reference": float | None, # |target - reference| / |reference - stop|
    "breakout_confirmed": bool,
    "retest_confirmed": bool,
    "reason_ja": str,
    "unavailable_reason": str | None,
  }
"""
from __future__ import annotations

from typing import Final


SCHEMA_VERSION: Final[str] = "pattern_levels_v1"


# Map pattern_kind → expected matched_parts roles + canonical part labels.
# `wave_shape_review.matched_parts` uses template-side keys like
# "prior_high" / "first_bottom" / "neckline_peak" / "second_bottom" /
# "breakout" / "left_shoulder" / "head" / "right_shoulder" / "neckline".
# We re-label them into the user-facing canonical scheme.
_PART_ALIAS: Final[dict[str, dict[str, str]]] = {
    "double_bottom": {
        "first_bottom": "B1",
        "second_bottom": "B2",
        "neckline_peak": "NL",
        "breakout": "BR",
        "prior_high": "PH",
    },
    "double_top": {
        "first_top": "P1",
        "second_top": "P2",
        "neckline_trough": "NL",
        "breakout": "BR",
        "prior_low": "PL",
    },
    "head_and_shoulders": {
        "left_shoulder": "LS",
        "head": "H",
        "right_shoulder": "RS",
        "neckline": "NL",
        "breakout": "BR",
    },
    "inverse_head_and_shoulders": {
        "left_shoulder": "LS",
        "head": "H",
        "right_shoulder": "RS",
        "neckline": "NL",
        "breakout": "BR",
    },
}


def _empty_panel(reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "available": False,
        "pattern_kind": None,
        "side": "NEUTRAL",
        "status": None,
        "parts": {},
        "trigger_line_id": None,
        "trigger_line_price": None,
        "stop_price": None,
        "target_price": None,
        "target_extended_price": None,
        "rr_at_reference": None,
        "rr_at_extended_target": None,
        "breakout_confirmed": False,
        "retest_confirmed": False,
        "pattern_height": None,
        # Provenance for STOP / TP. None when the panel is empty.
        # See derive_pattern_levels for the populated case.
        "level_source": None,
        "fallback_used": False,
        "fallback_reason": None,
        "reason_ja": "パターン情報が不足しています。",
        "unavailable_reason": reason,
    }


def _line_by_role(
    wave_lines: list[dict] | None, role: str,
) -> dict | None:
    if not wave_lines:
        return None
    for ln in wave_lines:
        if ln.get("role") == role:
            return ln
    return None


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _pivot_price_by_index(skeleton: dict | None, index: int | None) -> float | None:
    if skeleton is None or index is None:
        return None
    pivots = (skeleton or {}).get("pivots") or []
    for p in pivots:
        try:
            if int(p.get("index")) == int(index):
                return _safe_float(p.get("price"))
        except (TypeError, ValueError):
            continue
    return None


def _extract_parts_from_skeleton_sequence(
    *, kind: str, skeleton: dict | None,
) -> dict[str, dict]:
    """Derive canonical parts directly from the skeleton's L/H pivot
    sequence. Used as the authoritative path because matched_parts
    can be degenerate (multiple template parts pointing at the same
    pivot when pivot count < template parts).

    For double_bottom (sequence ...L H L H...):
        B1 = first L,  NL = first H following B1,
        B2 = first L following NL,  BR = first H following B2

    For double_top (sequence ...H L H L...):
        P1 = first H,  NL = first L following P1,
        P2 = first H following NL,  BR = first L following P2

    For head_and_shoulders (HLHLHL — 6 pivots ideal):
        LS = first H, then L (NL anchor),
        H  = next H (must be > LS),
        RS = next H (lower than H),
        BR = breakout below NL.

    For inverse_head_and_shoulders (LHLHLH):
        LS = first L, H = lowest L below LS, RS = L higher than H,
        NL = highs between, BR = above NL.
    """
    pivots = (skeleton or {}).get("pivots") or []
    if not pivots:
        return {}
    parts: dict[str, dict] = {}

    def _piv_dict(p: dict) -> dict:
        return {
            "price": float(p.get("price"))
            if p.get("price") is not None else None,
            "index": int(p.get("index"))
            if p.get("index") is not None else None,
        }

    if kind == "double_bottom":
        first_low = next((p for p in pivots if p.get("kind") == "L"), None)
        if first_low is None:
            return {}
        f_idx = int(first_low.get("index"))
        # NL = first H AFTER first_low
        first_high = next(
            (p for p in pivots
             if p.get("kind") == "H" and int(p.get("index")) > f_idx),
            None,
        )
        # B2 = first L AFTER first_high (or last L if no H found yet)
        second_low = None
        if first_high is not None:
            h_idx = int(first_high.get("index"))
            second_low = next(
                (p for p in pivots
                 if p.get("kind") == "L" and int(p.get("index")) > h_idx),
                None,
            )
        # BR = last H if it's after second_low (or after first_high)
        breakout = None
        anchor_idx = (
            int(second_low.get("index")) if second_low is not None
            else (int(first_high.get("index")) if first_high is not None else f_idx)
        )
        breakout = next(
            (p for p in reversed(pivots)
             if p.get("kind") == "H" and int(p.get("index")) > anchor_idx),
            None,
        )
        parts["B1"] = _piv_dict(first_low)
        if first_high is not None:
            parts["NL"] = _piv_dict(first_high)
        if second_low is not None:
            parts["B2"] = _piv_dict(second_low)
        if breakout is not None and breakout is not first_high:
            parts["BR"] = _piv_dict(breakout)

    elif kind == "double_top":
        first_high = next((p for p in pivots if p.get("kind") == "H"), None)
        if first_high is None:
            return {}
        h_idx = int(first_high.get("index"))
        first_low = next(
            (p for p in pivots
             if p.get("kind") == "L" and int(p.get("index")) > h_idx),
            None,
        )
        second_high = None
        if first_low is not None:
            l_idx = int(first_low.get("index"))
            second_high = next(
                (p for p in pivots
                 if p.get("kind") == "H" and int(p.get("index")) > l_idx),
                None,
            )
        anchor_idx = (
            int(second_high.get("index")) if second_high is not None
            else (int(first_low.get("index")) if first_low is not None else h_idx)
        )
        breakout = next(
            (p for p in reversed(pivots)
             if p.get("kind") == "L" and int(p.get("index")) > anchor_idx),
            None,
        )
        parts["P1"] = _piv_dict(first_high)
        if first_low is not None:
            parts["NL"] = _piv_dict(first_low)
        if second_high is not None:
            parts["P2"] = _piv_dict(second_high)
        if breakout is not None and breakout is not first_low:
            parts["BR"] = _piv_dict(breakout)

    elif kind == "head_and_shoulders":
        # SELL pattern. Sequence ...H L H L H L...
        highs = [p for p in pivots if p.get("kind") == "H"]
        lows = [p for p in pivots if p.get("kind") == "L"]
        if len(highs) >= 3 and len(lows) >= 2:
            ls, head, rs = highs[0], highs[1], highs[2]
            # head must be the highest of the three for canonical H&S
            if (
                float(head.get("price") or 0) > float(ls.get("price") or 0)
                and float(head.get("price") or 0) > float(rs.get("price") or 0)
            ):
                parts["LS"] = _piv_dict(ls)
                parts["H"] = _piv_dict(head)
                parts["RS"] = _piv_dict(rs)
                # NL anchor: the average of the two lows between shoulders
                neck_lows = [p for p in lows
                             if int(ls.get("index"))
                             < int(p.get("index"))
                             < int(rs.get("index"))]
                if neck_lows:
                    nl_price = sum(float(p.get("price")) for p in neck_lows) / len(neck_lows)
                    parts["NL"] = {"price": float(nl_price),
                                   "index": int(neck_lows[-1].get("index"))}
        # Breakout = first L AFTER right shoulder that's below NL
        if "RS" in parts and "NL" in parts:
            rs_idx = parts["RS"]["index"]
            nl_p = parts["NL"]["price"]
            br = next(
                (p for p in lows
                 if int(p.get("index")) > rs_idx
                 and float(p.get("price")) < nl_p),
                None,
            )
            if br is not None:
                parts["BR"] = _piv_dict(br)

    elif kind == "inverse_head_and_shoulders":
        # BUY pattern. Sequence ...L H L H L H...
        highs = [p for p in pivots if p.get("kind") == "H"]
        lows = [p for p in pivots if p.get("kind") == "L"]
        if len(lows) >= 3 and len(highs) >= 2:
            ls, head, rs = lows[0], lows[1], lows[2]
            if (
                float(head.get("price") or 0) < float(ls.get("price") or 0)
                and float(head.get("price") or 0) < float(rs.get("price") or 0)
            ):
                parts["LS"] = _piv_dict(ls)
                parts["H"] = _piv_dict(head)
                parts["RS"] = _piv_dict(rs)
                neck_highs = [p for p in highs
                              if int(ls.get("index"))
                              < int(p.get("index"))
                              < int(rs.get("index"))]
                if neck_highs:
                    nl_price = sum(float(p.get("price")) for p in neck_highs) / len(neck_highs)
                    parts["NL"] = {"price": float(nl_price),
                                   "index": int(neck_highs[-1].get("index"))}
        if "RS" in parts and "NL" in parts:
            rs_idx = parts["RS"]["index"]
            nl_p = parts["NL"]["price"]
            br = next(
                (p for p in highs
                 if int(p.get("index")) > rs_idx
                 and float(p.get("price")) > nl_p),
                None,
            )
            if br is not None:
                parts["BR"] = _piv_dict(br)

    return parts


def derive_pattern_levels(
    *,
    wave_shape_review: dict | None,
    wave_derived_lines: list[dict] | None,
    skeleton: dict | None,
    last_close: float | None,
) -> dict:
    """Build the pattern_levels_v1 panel.

    Reads wave_shape_review's best_pattern (kind / side_bias / status)
    and derives canonical parts directly from skeleton's L/H pivot
    sequence. The wave_derived_lines list provides actionable WSL/WTP
    prices for stop / target.
    """
    if not wave_shape_review:
        return _empty_panel("missing_wave_shape_review")
    best = (wave_shape_review or {}).get("best_pattern") or {}
    if not best:
        return _empty_panel("no_best_pattern")
    kind = (best.get("kind") or "").lower()
    side = (best.get("side_bias") or "NEUTRAL").upper()
    status = best.get("status") or "forming"

    parts = _extract_parts_from_skeleton_sequence(
        kind=kind, skeleton=skeleton,
    )

    # Trigger line. Priority order:
    # 1. parts.NL (from skeleton L/H sequence — authoritative)
    # 2. wave_derived_lines.role == "entry_confirmation_line" (legacy WNL)
    # 3. wave_derived_lines.role == "breakout_line" (flag/wedge)
    #
    # The skeleton-derived NL is preferred because matched_parts can be
    # degenerate when pivot count < template parts (NL was being mapped
    # to a Low pivot in the original implementation).
    trigger_id: str | None = None
    trigger_price: float | None = None
    if parts.get("NL", {}).get("price") is not None:
        trigger_price = float(parts["NL"]["price"])
        trigger_id = "WNL_derived"
    if trigger_price is None:
        nl_line = _line_by_role(wave_derived_lines, "entry_confirmation_line")
        if nl_line is not None:
            trigger_id = nl_line.get("id")
            trigger_price = _safe_float(nl_line.get("price"))
    if trigger_price is None:
        br_line = _line_by_role(wave_derived_lines, "breakout_line")
        if br_line is None:
            for ln in (wave_derived_lines or []):
                if ln.get("kind") == "pattern_breakout":
                    br_line = ln
                    break
        if br_line is not None:
            trigger_id = br_line.get("id")
            trigger_price = _safe_float(br_line.get("price"))

    # Stop / target from wave_derived_lines (WSL / WTP).
    stop_line = _line_by_role(wave_derived_lines, "stop_candidate")
    target_line = _line_by_role(wave_derived_lines, "target_candidate")
    stop_price = _safe_float(stop_line.get("price") if stop_line else None)
    target_price = _safe_float(target_line.get("price") if target_line else None)
    # Track which source ultimately supplied STOP / TP so the audit
    # report can warn the user when the wave_derived_lines were missing
    # or wrong-sided and a skeleton-based fallback recomputed the levels.
    stop_from_wave = stop_price is not None
    target_from_wave = target_price is not None
    stop_recomputed_from_skeleton = False
    target_recomputed_from_skeleton = False

    # When wave_derived_lines stop / target are missing or sit on
    # the wrong side of the trigger, recompute from skeleton-derived
    # parts so the integrated decision gets a usable structural plan.
    nl_price_struct = (parts.get("NL") or {}).get("price")
    pattern_height: float | None = None
    target_extended_price: float | None = None
    if side == "BUY":
        bottoms = [
            (parts.get("B1") or {}).get("price"),
            (parts.get("B2") or {}).get("price"),
            (parts.get("H") or {}).get("price"),  # IHS
        ]
        bottoms = [b for b in bottoms if b is not None]
        if bottoms and nl_price_struct is not None:
            lowest = float(min(bottoms))
            pattern_height = float(nl_price_struct) - lowest
            if pattern_height > 0:
                if stop_price is None or stop_price >= (trigger_price or nl_price_struct or 0.0):
                    stop_price = lowest - 0.30 * pattern_height
                    stop_recomputed_from_skeleton = True
                if target_price is None or target_price <= (trigger_price or nl_price_struct or 0.0):
                    target_price = float(nl_price_struct) + pattern_height
                    target_recomputed_from_skeleton = True
                # Extended target = 2× measured move (classic breakout
                # extension, used for RR gate so retest entries are
                # actionable; chart draws the 1× target).
                target_extended_price = float(nl_price_struct) + 2.0 * pattern_height
    elif side == "SELL":
        tops = [
            (parts.get("P1") or {}).get("price"),
            (parts.get("P2") or {}).get("price"),
            (parts.get("H") or {}).get("price"),  # H&S
        ]
        tops = [t for t in tops if t is not None]
        if tops and nl_price_struct is not None:
            highest = float(max(tops))
            pattern_height = highest - float(nl_price_struct)
            if pattern_height > 0:
                if stop_price is None or stop_price <= (trigger_price or nl_price_struct or 0.0):
                    stop_price = highest + 0.30 * pattern_height
                    stop_recomputed_from_skeleton = True
                if target_price is None or target_price >= (trigger_price or nl_price_struct or 0.0):
                    target_price = float(nl_price_struct) - pattern_height
                    target_recomputed_from_skeleton = True
                target_extended_price = float(nl_price_struct) - 2.0 * pattern_height

    # Provenance: which source ultimately supplied STOP / TP?
    if stop_recomputed_from_skeleton or target_recomputed_from_skeleton:
        fallback_used = True
        if not stop_from_wave and not target_from_wave:
            fallback_reason = "wave_derived_lines_missing"
        else:
            fallback_reason = "wave_derived_lines_missing_or_wrong_side"
        level_source = "skeleton_fallback"
    elif stop_price is not None and target_price is not None:
        fallback_used = False
        fallback_reason = None
        level_source = "wave_derived_lines"
    else:
        fallback_used = False
        fallback_reason = None
        level_source = None

    # RR: classic (1× projection) and extended (2×). The integrated
    # gate uses rr_extended for action gating because that reflects
    # retest-entry economics (deeper entry → larger reward / risk).
    reference = trigger_price
    if reference is None and last_close is not None:
        reference = float(last_close)
    rr: float | None = None
    rr_extended: float | None = None
    if (
        reference is not None
        and stop_price is not None
        and target_price is not None
        and reference != stop_price
    ):
        rr = abs(target_price - reference) / abs(reference - stop_price)
    if (
        reference is not None
        and stop_price is not None
        and target_extended_price is not None
        and reference != stop_price
    ):
        rr_extended = abs(target_extended_price - reference) / abs(reference - stop_price)

    breakout_confirmed = status in ("neckline_broken", "retested")
    retest_confirmed = status == "retested"

    # Reason text
    if not parts:
        reason_ja = (
            f"{kind} の主要部位を skeleton に紐付けられませんでした。"
        )
    elif trigger_price is None:
        reason_ja = (
            f"{kind} の部位は認識されていますが、トリガーライン (NL/BR) "
            "が引けていません。"
        )
    elif breakout_confirmed:
        reason_ja = (
            f"{kind} を認識し、トリガーライン {trigger_id} ({trigger_price:.5f}) を"
            f"既に突破しています。"
        )
    else:
        reason_ja = (
            f"{kind} を認識中。トリガーライン {trigger_id} "
            f"({trigger_price:.5f}) を未突破。"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "available": bool(parts),
        "pattern_kind": kind,
        "side": side,
        "status": status,
        "parts": parts,
        "trigger_line_id": trigger_id,
        "trigger_line_price": trigger_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "target_extended_price": target_extended_price,
        "rr_at_reference": float(rr) if rr is not None else None,
        "rr_at_extended_target": float(rr_extended) if rr_extended is not None else None,
        "breakout_confirmed": breakout_confirmed,
        "retest_confirmed": retest_confirmed,
        "pattern_height": float(pattern_height) if pattern_height is not None else None,
        # Provenance: where STOP / TP came from. Surfaced in the audit
        # so the user can tell when wave_derived_lines were unusable
        # and the skeleton-based fallback was applied instead.
        "level_source": level_source,           # "wave_derived_lines" / "skeleton_fallback" / None
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,     # None when fallback_used=False
        "reason_ja": reason_ja,
        "unavailable_reason": None,
    }


__all__ = [
    "SCHEMA_VERSION",
    "derive_pattern_levels",
]
