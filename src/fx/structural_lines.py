"""Structural lines — wave-context-derived royal-road lines.

The integrated profile already has a numeric trendline detector
(`detect_trendlines` / `trendline_context`) that finds candidate
trendlines statistically (slope / touch count / ATR distance / score).
That is genuine and we do not remove it.

But "the numbers say there is a line" and "the royal-road wave
structure says there is a line" are not the same thing. A user
following the royal road wants to know:

  - WNL came from the DT neckline, not from a regression
  - WSL came from P2 / above-the-second-top
  - WTP came from the wave target
  - T1/T2/T3 are the *numeric* trendlines
  - STL1 etc. are the *structural* trendlines
  - Whether numeric and structural lines agree

This module builds **structural lines** — lines that exist BECAUSE
of the recognised wave / pattern / dow structure, with `anchor_parts`
recording which pieces of the wave they were derived from. The
existing numeric layer is left untouched; the structural snapshot
just lives next to it and `numeric_alignment` annotates how the
two compare.

Phase I follow-up: observation only. Never changes entry_plan_v1
READY conditions, the candidate selector, or the final action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


STRUCTURAL_LINES_SCHEMA_VERSION = "structural_lines_v1"


StructuralLineKind = Literal[
    "structural_neckline",
    "structural_trendline",
    "structural_channel",
    "structural_support_resistance",
    "structural_invalidation",
    "structural_target",
]

# Roles include the canonical royal-road roles plus a few wave-shape
# specific roles (top/bottom structure line) so the source intent is
# preserved end-to-end.
StructuralLineRole = Literal[
    "entry_trigger",
    "stop_candidate",
    "target_candidate",
    "uptrend_support",
    "downtrend_resistance",
    "channel_upper",
    "channel_lower",
    "support",
    "resistance",
    "retest_zone",
    "breakout_line",
    "bottom_structure_line",
    "top_structure_line",
]

StructuralLineSource = Literal[
    "pattern_levels",
    "wave_derived_lines",
    "multi_scale_chart",
    "dow_structure_review",
    "support_resistance_v2",
]

NumericAlignment = Literal["MATCH", "NEAR", "CONFLICT", "NONE", "UNKNOWN"]


@dataclass(frozen=True)
class StructuralLine:
    schema_version: str
    id: str
    kind: StructuralLineKind
    role: StructuralLineRole
    side: str

    # For horizontal line
    price: float | None = None

    # For diagonal line
    x1: float | None = None
    y1: float | None = None
    x2: float | None = None
    y2: float | None = None
    slope: float | None = None

    source: StructuralLineSource = "pattern_levels"
    anchor_parts: list[str] = field(default_factory=list)

    # Human explanation
    reason_ja: str = ""
    used_in_royal_road: bool = True

    # Matching / comparison with numeric lines
    matched_numeric_line_id: str | None = None
    numeric_alignment: NumericAlignment = "UNKNOWN"

    confidence: float = 0.0
    cautions: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "kind": self.kind,
            "role": self.role,
            "side": self.side,
            "price": self.price,
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "slope": self.slope,
            "source": self.source,
            "anchor_parts": list(self.anchor_parts),
            "reason_ja": self.reason_ja,
            "used_in_royal_road": self.used_in_royal_road,
            "matched_numeric_line_id": self.matched_numeric_line_id,
            "numeric_alignment": self.numeric_alignment,
            "confidence": self.confidence,
            "cautions": list(self.cautions),
            "debug": dict(self.debug),
        }


@dataclass(frozen=True)
class StructuralLinesSnapshot:
    schema_version: str
    lines: list[StructuralLine]
    summary_ja: str
    counts: dict[str, int]
    cautions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lines": [line.to_dict() for line in self.lines],
            "summary_ja": self.summary_ja,
            "counts": dict(self.counts),
            "cautions": list(self.cautions),
        }


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _line(
    *,
    line_id: str,
    kind: StructuralLineKind,
    role: StructuralLineRole,
    side: str,
    source: StructuralLineSource,
    anchor_parts: list[str],
    reason_ja: str,
    price: float | None = None,
    slope: float | None = None,
    x1: float | None = None, y1: float | None = None,
    x2: float | None = None, y2: float | None = None,
    confidence: float = 0.7,
    cautions: list[str] | None = None,
    debug: dict[str, Any] | None = None,
) -> StructuralLine:
    return StructuralLine(
        schema_version=STRUCTURAL_LINES_SCHEMA_VERSION,
        id=line_id,
        kind=kind,
        role=role,
        side=side,
        price=price,
        x1=x1, y1=y1, x2=x2, y2=y2,
        slope=slope,
        source=source,
        anchor_parts=list(anchor_parts),
        reason_ja=reason_ja,
        used_in_royal_road=True,
        matched_numeric_line_id=None,
        numeric_alignment="UNKNOWN",
        confidence=confidence,
        cautions=list(cautions or []),
        debug=dict(debug or {}),
    )


# ─────────────────────────────────────────────────────────────────
# Builders per source
# ─────────────────────────────────────────────────────────────────


def _build_from_wave_derived_lines(
    *,
    wave_derived_lines: list[dict[str, Any]],
    side: str,
    counters: dict[str, int],
) -> list[StructuralLine]:
    """Lift each role-tagged wave-derived line into a structural line.

    role=entry_confirmation_line → structural_neckline / entry_trigger
    role=stop_candidate          → structural_invalidation / stop_candidate
    role=target_candidate        → structural_target / target_candidate

    Anchor labels come from the source line id (e.g. WNL1 → ['NL']).
    """
    out: list[StructuralLine] = []
    for line in wave_derived_lines or []:
        if not isinstance(line, dict):
            continue
        role_raw = str(line.get("role") or "")
        price = _as_float(line.get("price"))
        if price is None:
            continue
        src_id = str(line.get("id") or "")
        debug_payload = {
            "source_line_id": src_id,
            "source_kind": line.get("kind") or "",
        }
        if role_raw == "entry_confirmation_line":
            counters["snl"] += 1
            out.append(_line(
                line_id=f"SNL{counters['snl']}",
                kind="structural_neckline",
                role="entry_trigger",
                side=side,
                source="wave_derived_lines",
                anchor_parts=["NL"],
                price=price,
                reason_ja=(
                    "波形から導いたネックラインです。"
                    "WNLとしてENTRY確認に使います。"
                ),
                confidence=0.85,
                debug=debug_payload,
            ))
        elif role_raw == "stop_candidate":
            counters["sil"] += 1
            anchors = (
                ["P2"] if side == "SELL" else (
                    ["B2"] if side == "BUY" else []
                )
            )
            out.append(_line(
                line_id=f"SIL{counters['sil']}",
                kind="structural_invalidation",
                role="stop_candidate",
                side=side,
                source="wave_derived_lines",
                anchor_parts=anchors,
                price=price,
                reason_ja=(
                    "波形崩れラインです。"
                    "WSLとしてSTOP候補に使います。"
                ),
                confidence=0.8,
                debug=debug_payload,
            ))
        elif role_raw == "target_candidate":
            counters["stp"] += 1
            out.append(_line(
                line_id=f"STP{counters['stp']}",
                kind="structural_target",
                role="target_candidate",
                side=side,
                source="wave_derived_lines",
                anchor_parts=["BR"],
                price=price,
                reason_ja=(
                    "波形から導いた利確候補です。"
                    "WTPとしてTPに使います。"
                ),
                confidence=0.75,
                debug=debug_payload,
            ))
    return out


def _build_from_pattern_levels(
    *,
    pattern_levels: dict[str, Any],
    side: str,
    counters: dict[str, int],
    skip_kinds: set[str],
) -> list[StructuralLine]:
    """Fallback for when wave_derived_lines is empty.

    Reads pattern_levels.{trigger_line_price, stop_price,
    target_price / target_extended_price} and emits structural
    neckline / invalidation / target lines so the panel always shows
    SOMETHING when a pattern was recognised.

    Skipped if a kind is already present in `skip_kinds` (so we don't
    double-emit when wave_derived_lines already covered it).
    """
    out: list[StructuralLine] = []
    pl = pattern_levels or {}
    parts = pl.get("parts") or {}

    if "structural_neckline" not in skip_kinds:
        nl_price = _as_float(pl.get("trigger_line_price"))
        if nl_price is None:
            nl_price = _as_float((parts.get("NL") or {}).get("price"))
        if nl_price is not None:
            counters["snl"] += 1
            out.append(_line(
                line_id=f"SNL{counters['snl']}",
                kind="structural_neckline",
                role="entry_trigger",
                side=side,
                source="pattern_levels",
                anchor_parts=["NL"],
                price=nl_price,
                reason_ja=(
                    "pattern_levels.NL から導いたネックラインです。"
                ),
                confidence=0.8,
            ))

    if "structural_invalidation" not in skip_kinds:
        stop_price = _as_float(pl.get("stop_price"))
        if stop_price is not None:
            anchors = (
                ["P2"] if side == "SELL" else (
                    ["B2"] if side == "BUY" else []
                )
            )
            counters["sil"] += 1
            out.append(_line(
                line_id=f"SIL{counters['sil']}",
                kind="structural_invalidation",
                role="stop_candidate",
                side=side,
                source="pattern_levels",
                anchor_parts=anchors,
                price=stop_price,
                reason_ja=(
                    "pattern_levels.stop_price から導いた波形崩れ"
                    "ラインです。STOP候補として使います。"
                ),
                confidence=0.75,
            ))

    if "structural_target" not in skip_kinds:
        tgt_price = _as_float(
            pl.get("target_extended_price")
            if pl.get("target_extended_price") is not None
            else pl.get("target_price")
        )
        if tgt_price is not None:
            counters["stp"] += 1
            out.append(_line(
                line_id=f"STP{counters['stp']}",
                kind="structural_target",
                role="target_candidate",
                side=side,
                source="pattern_levels",
                anchor_parts=["BR"],
                price=tgt_price,
                reason_ja=(
                    "pattern_levels.target_(extended_)price から導いた"
                    "利確候補です。"
                ),
                confidence=0.7,
            ))

    return out


def _build_pattern_p1_p2_or_b1_b2(
    *,
    pattern_levels: dict[str, Any],
    side: str,
    counters: dict[str, int],
) -> list[StructuralLine]:
    """Generate a structural trendline from DB B1↔B2 or DT P1↔P2.

    For DB → bottom_structure_line / support
    For DT → top_structure_line / resistance
    Slope is computed from (idx1, price1) → (idx2, price2). The line
    is informative even if the slope is near zero (parallel highs /
    parallel lows = horizontal structural line).
    """
    pl = pattern_levels or {}
    parts = pl.get("parts") or {}
    pattern_kind = str(pl.get("pattern_kind") or "").lower()

    if pattern_kind in ("double_top",) and "P1" in parts and "P2" in parts:
        p1 = parts["P1"] or {}
        p2 = parts["P2"] or {}
        i1 = _as_float(p1.get("index"))
        v1 = _as_float(p1.get("price"))
        i2 = _as_float(p2.get("index"))
        v2 = _as_float(p2.get("price"))
        if all(x is not None for x in (i1, v1, i2, v2)) and i1 != i2:
            slope = (v2 - v1) / (i2 - i1)
            counters["stl"] += 1
            return [_line(
                line_id=f"STL{counters['stl']}",
                kind="structural_trendline",
                role="top_structure_line",
                side=side,
                source="pattern_levels",
                anchor_parts=["P1", "P2"],
                slope=slope,
                x1=i1, y1=v1, x2=i2, y2=v2,
                reason_ja=(
                    "DTのP1とP2を結ぶ上値抵抗構造線です。"
                ),
                confidence=0.8,
            )]
    elif pattern_kind in ("double_bottom",) and "B1" in parts and "B2" in parts:
        b1 = parts["B1"] or {}
        b2 = parts["B2"] or {}
        i1 = _as_float(b1.get("index"))
        v1 = _as_float(b1.get("price"))
        i2 = _as_float(b2.get("index"))
        v2 = _as_float(b2.get("price"))
        if all(x is not None for x in (i1, v1, i2, v2)) and i1 != i2:
            slope = (v2 - v1) / (i2 - i1)
            counters["stl"] += 1
            return [_line(
                line_id=f"STL{counters['stl']}",
                kind="structural_trendline",
                role="bottom_structure_line",
                side=side,
                source="pattern_levels",
                anchor_parts=["B1", "B2"],
                slope=slope,
                x1=i1, y1=v1, x2=i2, y2=v2,
                reason_ja=(
                    "DBのB1とB2を結ぶ下値支持構造線です。"
                ),
                confidence=0.8,
            )]
    return []


def _extract_pivots_from_multi_scale_chart(
    multi_scale_chart: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Defensive pivot extractor.

    Tries:
      multi_scale_chart["wave_skeleton"]["pivots"]
      multi_scale_chart["pivots"]
      multi_scale_chart["scales"][k]["pivots"] / wave_skeleton.pivots
    Returns the first non-empty pivot list found.
    """
    msc = multi_scale_chart or {}
    if not isinstance(msc, dict):
        return []
    direct = (msc.get("wave_skeleton") or {}).get("pivots") or []
    if direct:
        return list(direct)
    direct2 = msc.get("pivots") or []
    if direct2:
        return list(direct2)
    scales = msc.get("scales")
    if isinstance(scales, dict):
        scales_iter = list(scales.values())
    elif isinstance(scales, list):
        scales_iter = list(scales)
    else:
        scales_iter = []
    for scale in scales_iter:
        if not isinstance(scale, dict):
            continue
        sk = scale.get("wave_skeleton") or {}
        ps = sk.get("pivots") or scale.get("pivots") or []
        if ps:
            return list(ps)
    return []


def _pivot_kind(p: dict[str, Any]) -> str:
    return str(
        p.get("kind")
        or p.get("pivot_kind")
        or p.get("type")
        or ""
    ).upper()


def _build_from_multi_scale_chart(
    *,
    multi_scale_chart: dict[str, Any],
    dow_trend: str,
    side: str,
    counters: dict[str, int],
) -> tuple[list[StructuralLine], list[str]]:
    """Build a single HL-HL or LH-LH structural trendline if the
    multi-scale skeleton has enough higher-low / lower-high pivots.

    Returns (lines, cautions).
    """
    pivots = _extract_pivots_from_multi_scale_chart(multi_scale_chart)
    if len(pivots) < 4:
        return [], ["multi_scale_pivots_insufficient_for_structural_trendline"]

    # Normalise + sort by index.
    norm: list[tuple[int, float, str]] = []
    for p in pivots:
        if not isinstance(p, dict):
            continue
        idx = p.get("index") if p.get("index") is not None else p.get("idx")
        try:
            idx_i = int(idx)
        except (TypeError, ValueError):
            continue
        price = _as_float(p.get("price"))
        if price is None:
            continue
        norm.append((idx_i, price, _pivot_kind(p)))
    if len(norm) < 4:
        return [], ["multi_scale_pivots_insufficient_for_structural_trendline"]
    norm.sort(key=lambda r: r[0])

    if dow_trend == "UP":
        # Higher-lows: take the two most recent L pivots.
        lows = [r for r in norm if r[2] == "L"]
        if len(lows) >= 2:
            (i1, v1, _), (i2, v2, _) = lows[-2], lows[-1]
            if v2 > v1 and i2 != i1:
                slope = (v2 - v1) / (i2 - i1)
                counters["stl"] += 1
                return [_line(
                    line_id=f"STL{counters['stl']}",
                    kind="structural_trendline",
                    role="uptrend_support",
                    side=side,
                    source="multi_scale_chart",
                    anchor_parts=["HL", "HL"],
                    slope=slope,
                    x1=float(i1), y1=v1, x2=float(i2), y2=v2,
                    reason_ja=(
                        "ダウのHL（直近の押し安値）同士を結ぶ"
                        "上昇支持構造線です。"
                    ),
                    confidence=0.65,
                )], []
        return [], ["multi_scale_higher_lows_not_found"]
    if dow_trend == "DOWN":
        highs = [r for r in norm if r[2] == "H"]
        if len(highs) >= 2:
            (i1, v1, _), (i2, v2, _) = highs[-2], highs[-1]
            if v2 < v1 and i2 != i1:
                slope = (v2 - v1) / (i2 - i1)
                counters["stl"] += 1
                return [_line(
                    line_id=f"STL{counters['stl']}",
                    kind="structural_trendline",
                    role="downtrend_resistance",
                    side=side,
                    source="multi_scale_chart",
                    anchor_parts=["LH", "LH"],
                    slope=slope,
                    x1=float(i1), y1=v1, x2=float(i2), y2=v2,
                    reason_ja=(
                        "ダウのLH（直近の戻り高値）同士を結ぶ"
                        "下降抵抗構造線です。"
                    ),
                    confidence=0.65,
                )], []
        return [], ["multi_scale_lower_highs_not_found"]
    # RANGE / MIXED / UNKNOWN — don't synthesise a directional line.
    return [], []


# ─────────────────────────────────────────────────────────────────
# Numeric alignment
# ─────────────────────────────────────────────────────────────────


def _numeric_slopes(
    trendline_context: dict[str, Any] | None,
) -> list[tuple[str, float]]:
    """Pull (id, slope) tuples from numeric trendline_context."""
    tc = trendline_context or {}
    sel = (
        tc.get("selected_trendlines_top3")
        or tc.get("selected_trendlines")
        or []
    )
    out: list[tuple[str, float]] = []
    for i, t in enumerate(sel):
        if not isinstance(t, dict):
            continue
        slope = (
            t.get("slope")
            or t.get("line_slope")
            or t.get("price_slope")
        )
        s = _as_float(slope)
        if s is None:
            continue
        line_id = (
            str(t.get("id") or t.get("line_id") or t.get("rank") or "")
            or f"T{i + 1}"
        )
        out.append((line_id, s))
    return out


def annotate_numeric_alignment(
    structural_lines: list[StructuralLine],
    trendline_context: dict[str, Any] | None,
) -> list[StructuralLine]:
    """Tag each diagonal structural line with how it compares to the
    numeric trendline_context output. Horizontal structural lines are
    left as UNKNOWN (the numeric layer is for diagonal trendlines)."""
    numeric = _numeric_slopes(trendline_context)

    annotated: list[StructuralLine] = []
    for s in structural_lines:
        if s.kind != "structural_trendline" or s.slope is None:
            annotated.append(s)
            continue
        if not numeric:
            annotated.append(_replace_alignment(
                s, alignment="NONE", matched_id=None,
            ))
            continue
        # Find the numeric line whose slope is closest in absolute
        # difference. CONFLICT if signs differ.
        s_slope = s.slope
        # Best candidate
        best_id = None
        best_diff = None
        for nid, nslope in numeric:
            d = abs(nslope - s_slope)
            if best_diff is None or d < best_diff:
                best_diff = d
                best_id = nid
                best_slope = nslope
        # Sign conflict?
        sign_conflict = (
            (s_slope > 0 and best_slope < 0)
            or (s_slope < 0 and best_slope > 0)
        )
        ref = max(abs(s_slope), abs(best_slope), 1e-12)
        rel = best_diff / ref
        if sign_conflict and abs(s_slope) > 1e-9 and abs(best_slope) > 1e-9:
            alignment = "CONFLICT"
        elif rel <= 0.15:
            alignment = "MATCH"
        elif rel <= 0.35:
            alignment = "NEAR"
        else:
            alignment = "CONFLICT"
        annotated.append(_replace_alignment(
            s, alignment=alignment, matched_id=best_id,
        ))
    return annotated


def _replace_alignment(
    s: StructuralLine, *,
    alignment: NumericAlignment,
    matched_id: str | None,
) -> StructuralLine:
    return StructuralLine(
        schema_version=s.schema_version,
        id=s.id,
        kind=s.kind,
        role=s.role,
        side=s.side,
        price=s.price,
        x1=s.x1, y1=s.y1, x2=s.x2, y2=s.y2,
        slope=s.slope,
        source=s.source,
        anchor_parts=list(s.anchor_parts),
        reason_ja=s.reason_ja,
        used_in_royal_road=s.used_in_royal_road,
        matched_numeric_line_id=matched_id,
        numeric_alignment=alignment,
        confidence=s.confidence,
        cautions=list(s.cautions),
        debug=dict(s.debug),
    )


# ─────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────


def build_structural_lines(
    *,
    pattern_levels: dict[str, Any] | None,
    wave_derived_lines: list[dict[str, Any]] | None,
    multi_scale_chart: dict[str, Any] | None,
    dow_structure_review: dict[str, Any] | None,
    support_resistance_v2: dict[str, Any] | None,
    trendline_context: dict[str, Any] | None,
) -> StructuralLinesSnapshot:
    """Build the structural-lines snapshot from the wave / pattern /
    dow context. The snapshot lives next to the existing numeric
    `trendline_context` / `support_resistance_v2`; it does not
    replace them.
    """
    pl = pattern_levels or {}
    side = str(pl.get("side") or "NEUTRAL")

    counters = {"snl": 0, "sil": 0, "stp": 0, "stl": 0}
    cautions: list[str] = []

    lines: list[StructuralLine] = []

    # A. wave_derived_lines (canonical W lines).
    wd_lines = _build_from_wave_derived_lines(
        wave_derived_lines=wave_derived_lines or [],
        side=side, counters=counters,
    )
    lines.extend(wd_lines)
    skip = {ln.kind for ln in wd_lines}

    # B. pattern_levels fallback for any missing kinds.
    pl_lines = _build_from_pattern_levels(
        pattern_levels=pl, side=side, counters=counters, skip_kinds=skip,
    )
    lines.extend(pl_lines)

    # C. DB B1↔B2 / DT P1↔P2 structural trendline.
    pattern_lines = _build_pattern_p1_p2_or_b1_b2(
        pattern_levels=pl, side=side, counters=counters,
    )
    lines.extend(pattern_lines)

    # D. multi_scale_chart structural trendline (HL-HL / LH-LH).
    dow_trend = str(
        (dow_structure_review or {}).get("trend")
        or (dow_structure_review or {}).get("dow_trend")
        or ""
    ).upper()
    msc_lines, msc_cautions = _build_from_multi_scale_chart(
        multi_scale_chart=multi_scale_chart or {},
        dow_trend=dow_trend, side=side, counters=counters,
    )
    lines.extend(msc_lines)
    cautions.extend(msc_cautions)

    # E. numeric alignment annotation.
    lines = annotate_numeric_alignment(lines, trendline_context)

    # Counts.
    counts: dict[str, int] = {
        "total": len(lines),
        "structural_neckline": sum(
            1 for ln in lines if ln.kind == "structural_neckline"
        ),
        "structural_invalidation": sum(
            1 for ln in lines if ln.kind == "structural_invalidation"
        ),
        "structural_target": sum(
            1 for ln in lines if ln.kind == "structural_target"
        ),
        "structural_trendline": sum(
            1 for ln in lines if ln.kind == "structural_trendline"
        ),
        "structural_channel": sum(
            1 for ln in lines if ln.kind == "structural_channel"
        ),
        "structural_support_resistance": sum(
            1 for ln in lines if ln.kind == "structural_support_resistance"
        ),
        "numeric_match": sum(
            1 for ln in lines if ln.numeric_alignment == "MATCH"
        ),
        "numeric_near": sum(
            1 for ln in lines if ln.numeric_alignment == "NEAR"
        ),
        "numeric_conflict": sum(
            1 for ln in lines if ln.numeric_alignment == "CONFLICT"
        ),
        "numeric_none": sum(
            1 for ln in lines if ln.numeric_alignment == "NONE"
        ),
    }

    if counts["total"] == 0:
        summary_ja = "波形構造から構造ラインを生成できませんでした。"
    else:
        bits = [f"構造ラインを{counts['total']}本生成しました。"]
        nick_parts = []
        if counts["structural_neckline"]:
            nick_parts.append(f"SNL×{counts['structural_neckline']}")
        if counts["structural_invalidation"]:
            nick_parts.append(f"SIL×{counts['structural_invalidation']}")
        if counts["structural_target"]:
            nick_parts.append(f"STP×{counts['structural_target']}")
        if counts["structural_trendline"]:
            nick_parts.append(f"STL×{counts['structural_trendline']}")
        if nick_parts:
            bits.append("(" + " / ".join(nick_parts) + ")")
        if counts["numeric_conflict"]:
            bits.append(
                "数値検出ラインと CONFLICT が"
                f"{counts['numeric_conflict']}本あります。"
            )
        summary_ja = " ".join(bits)

    return StructuralLinesSnapshot(
        schema_version=STRUCTURAL_LINES_SCHEMA_VERSION,
        lines=lines,
        summary_ja=summary_ja,
        counts=counts,
        cautions=cautions,
    )


__all__ = [
    "STRUCTURAL_LINES_SCHEMA_VERSION",
    "StructuralLine",
    "StructuralLinesSnapshot",
    "build_structural_lines",
    "annotate_numeric_alignment",
]
