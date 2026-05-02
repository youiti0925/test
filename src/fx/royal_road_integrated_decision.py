"""royal_road_integrated_decision — opt-in research decision profile.

Audit information IS the decision. The standard royal_road_decision_v2
keeps wave / fib / Masterclass panels as audit-only; this profile
elevates them into first-class evidence the action is built from:

  - wave_shape_review                (波形)
  - wave_derived_lines               (W lines: WNL/WSL/WTP/WBR/WUP/WLOW)
  - fibonacci_context_review         (fib confluence)
  - candlestick_anatomy_review       (entry candle anatomy)
  - dow_structure_review             (HH/HL/LH/LL + BoS)
  - level_psychology_review          (S/R clustering)
  - ma_context_review / granville_entry_review
  - rsi_regime_filter                (regime-aware, NOT alone)
  - bollinger_lifecycle_review       (helper)
  - macd_architecture_review         (helper)
  - divergence_review                (caution-only, NOT alone)
  - invalidation_engine_v2 + RR      (REQUIRED)
  - macro_real_data / daily_roadmap_review / symbol_macro_briefing_review
                                      (strict=BLOCK / balanced=WARN)

Two opt-in modes:

  integrated_strict
      Required data missing (macro / lower_tf / position sizing /
      news) → HOLD.

  integrated_balanced
      Required data missing → WARN, not auto-BLOCK. RR / stop /
      invalidation / WNL still HOLD if missing.

Strict invariants:

  - Default profile remains current_runtime (byte-identical to PR #21).
  - This module is NEVER imported by live / OANDA / paper paths.
  - The existing royal_road_decision_v2 / v1 / current_runtime profiles
    are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from .chart_patterns import detect_patterns
from .decision_engine import Decision, MIN_RISK_REWARD, _hold
from .macro_alignment import compute_macro_alignment
from .masterclass_candlestick import build_candlestick_anatomy_review
from .masterclass_dow import build_dow_structure_review
from .patterns import PatternResult
from .risk_gate import RiskState, evaluate as evaluate_gate
from .stop_modes import DEFAULT_STOP_MODE, plan_stop
from .support_resistance import detect_levels


PROFILE_NAME_V2_INTEGRATED: Final[str] = "royal_road_decision_v2_integrated"


# ── Mode constants ────────────────────────────────────────────────
INTEGRATED_MODE_STRICT: Final[str] = "integrated_strict"
INTEGRATED_MODE_BALANCED: Final[str] = "integrated_balanced"
DEFAULT_INTEGRATED_MODE: Final[str] = INTEGRATED_MODE_BALANCED

SUPPORTED_INTEGRATED_MODES: Final[tuple[str, ...]] = (
    INTEGRATED_MODE_BALANCED,
    INTEGRATED_MODE_STRICT,
)


# ── Evidence axis status enum ─────────────────────────────────────
PASS: Final[str] = "PASS"
WARN: Final[str] = "WARN"
BLOCK: Final[str] = "BLOCK"
UNKNOWN: Final[str] = "UNKNOWN"

SUPPORTED_AXIS_STATUS: Final[tuple[str, ...]] = (PASS, WARN, BLOCK, UNKNOWN)

SIDE_BUY: Final[str] = "BUY"
SIDE_SELL: Final[str] = "SELL"
SIDE_NEUTRAL: Final[str] = "NEUTRAL"


# Pattern → side classification. Accept both wave_shape_review naming
# ("double_bottom") and chart_patterns naming ("triangle_ascending").
_BUY_PATTERN_KINDS: Final[frozenset] = frozenset({
    # wave_shape_review naming
    "double_bottom", "inverse_head_and_shoulders",
    "falling_wedge", "bullish_flag", "ascending_triangle",
    # chart_patterns naming
    "wedge_falling", "flag_bullish", "triangle_ascending",
})
_SELL_PATTERN_KINDS: Final[frozenset] = frozenset({
    # wave_shape_review naming
    "double_top", "head_and_shoulders",
    "rising_wedge", "bearish_flag", "descending_triangle",
    # chart_patterns naming
    "wedge_rising", "flag_bearish", "triangle_descending",
})
_NEUTRAL_PATTERN_KINDS: Final[frozenset] = frozenset({
    "symmetric_triangle", "triangle_symmetric",
})


def validate_integrated_mode(name: str) -> str:
    if name not in SUPPORTED_INTEGRATED_MODES:
        raise ValueError(
            f"unknown integrated mode: {name!r}. "
            f"Supported: {', '.join(SUPPORTED_INTEGRATED_MODES)}"
        )
    return name


# ── Data structures ───────────────────────────────────────────────
@dataclass(frozen=True)
class IntegratedEvidenceAxis:
    """Single axis of evidence used by the integrated decision.

    `used_in_decision=True` means this axis CAN influence action.
    `required=True` means absence/UNKNOWN forces HOLD. Other axes
    contribute confluence / confidence only.
    """
    axis: str
    side: str
    status: str
    strength: float
    confidence: float
    used_in_decision: bool
    required: bool
    reason_ja: str
    source: str

    def __post_init__(self) -> None:
        if self.side not in (SIDE_BUY, SIDE_SELL, SIDE_NEUTRAL):
            raise ValueError(f"invalid side: {self.side!r}")
        if self.status not in SUPPORTED_AXIS_STATUS:
            raise ValueError(f"invalid status: {self.status!r}")
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"strength out of range: {self.strength}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence out of range: {self.confidence}")

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "side": self.side,
            "status": self.status,
            "strength": float(self.strength),
            "confidence": float(self.confidence),
            "used_in_decision": bool(self.used_in_decision),
            "required": bool(self.required),
            "reason_ja": self.reason_ja,
            "source": self.source,
        }


@dataclass(frozen=True)
class RoyalRoadIntegratedDecision:
    action: str
    label: str
    side_bias: str
    confidence: float
    mode: str
    block_reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    axes: list[IntegratedEvidenceAxis] = field(default_factory=list)
    used_modules: list[str] = field(default_factory=list)
    audit_only_modules: list[str] = field(default_factory=list)
    not_connected_modules: list[str] = field(default_factory=list)
    explanation_ja: str = ""

    def to_dict(self) -> dict:
        return {
            "schema_version": "royal_road_integrated_decision_v1",
            "action": self.action,
            "label": self.label,
            "side_bias": self.side_bias,
            "confidence": float(self.confidence),
            "mode": self.mode,
            "block_reasons": list(self.block_reasons),
            "cautions": list(self.cautions),
            "axes": [a.to_dict() for a in self.axes],
            "used_modules": list(self.used_modules),
            "audit_only_modules": list(self.audit_only_modules),
            "not_connected_modules": list(self.not_connected_modules),
            "explanation_ja": self.explanation_ja,
        }


def empty_integrated_decision(
    *, mode: str, reason_ja: str, label: str = "HOLD_INTEGRATED",
) -> RoyalRoadIntegratedDecision:
    return RoyalRoadIntegratedDecision(
        action="HOLD",
        label=label,
        side_bias=SIDE_NEUTRAL,
        confidence=0.0,
        mode=mode,
        block_reasons=[],
        cautions=[],
        axes=[],
        used_modules=[],
        audit_only_modules=[],
        not_connected_modules=[],
        explanation_ja=reason_ja,
    )


# ── Helpers ───────────────────────────────────────────────────────
def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _classify_pattern_side(kind: str | None, *, fallback: str = SIDE_NEUTRAL) -> str:
    if not kind:
        return fallback
    k = str(kind).lower()
    if k in _BUY_PATTERN_KINDS:
        return SIDE_BUY
    if k in _SELL_PATTERN_KINDS:
        return SIDE_SELL
    if k in _NEUTRAL_PATTERN_KINDS:
        return SIDE_NEUTRAL
    return fallback


# ── Axis builders ─────────────────────────────────────────────────
def _axis_wave_pattern(chart_pattern: dict | None) -> IntegratedEvidenceAxis:
    """Build wave_pattern axis from chart_pattern_review (or detect_patterns).

    PASS  → breakout / formation_complete
    WARN  → forming
    BLOCK → invalidated
    """
    cp = _as_dict(chart_pattern)
    if not cp or not cp.get("available", True):
        return IntegratedEvidenceAxis(
            axis="wave_pattern", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=False,
            reason_ja="波形パターン候補が見つかりません。",
            source="chart_patterns",
        )
    kind = cp.get("detected_pattern") or cp.get("kind")
    side = _classify_pattern_side(kind)
    state = (cp.get("state") or cp.get("formation_state") or "").lower()
    invalidated = bool(cp.get("invalidated"))
    breakout_confirmed = bool(
        cp.get("breakout_confirmed")
        or cp.get("bullish_breakout_confirmed")
        or cp.get("bearish_breakout_confirmed")
    )
    neckline_broken = bool(cp.get("neckline_broken"))
    formation_complete = bool(cp.get("formation_complete"))

    if invalidated:
        status, strength = BLOCK, 0.0
        reason_ja = f"波形 {kind} が無効化されました。"
    elif breakout_confirmed or neckline_broken or formation_complete:
        status = PASS
        strength = 0.85
        reason_ja = f"波形 {kind} がブレイク / ネックライン突破を確認。"
    elif state in ("forming", "incomplete") or kind in _NEUTRAL_PATTERN_KINDS:
        status = WARN
        strength = 0.4
        reason_ja = (
            f"波形 {kind} は形成中。ネックライン突破を待つ局面です。"
        )
    elif side != SIDE_NEUTRAL:
        status = WARN
        strength = 0.5
        reason_ja = f"波形 {kind} を検出しましたが、確証は弱めです。"
    else:
        status = UNKNOWN
        strength = 0.0
        reason_ja = "波形が方向性を示していません。"

    confidence = 0.85 if status == PASS else 0.5 if status == WARN else 0.2
    return IntegratedEvidenceAxis(
        axis="wave_pattern", side=side, status=status,
        strength=strength, confidence=confidence,
        used_in_decision=True, required=False,
        reason_ja=reason_ja, source="chart_patterns",
    )


def _axis_wave_lines(
    *,
    wave_lines: list[dict] | None,
    chart_pattern: dict | None,
    last_close: float | None,
    entry_summary: dict | None,
    min_rr: float,
) -> IntegratedEvidenceAxis:
    """Build wave_lines axis (WNL/WSL/WTP/WBR breakout + RR).

    REQUIRED axis. PASS only if WNL broken + WSL present + WTP present
    + RR >= min_rr. Missing any → BLOCK (forces HOLD).
    """
    side_hint = _classify_pattern_side(
        (chart_pattern or {}).get("detected_pattern") or (chart_pattern or {}).get("kind")
    )
    wlines = list(wave_lines or [])
    if not wlines:
        return IntegratedEvidenceAxis(
            axis="wave_lines", side=side_hint, status=BLOCK,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=True,
            reason_ja=(
                "W ライン (WNL/WSL/WTP) が引けていません。"
                "波形が確定していないため HOLD。"
            ),
            source="wave_derived_lines",
        )

    wnl = next((l for l in wlines if l.get("role") == "entry_confirmation_line"), None)
    wsl = next((l for l in wlines if l.get("role") == "stop_candidate"), None)
    wtp = next((l for l in wlines if l.get("role") == "target_candidate"), None)

    es = _as_dict(entry_summary)
    rr = es.get("rr") if es else None
    rr_ok = isinstance(rr, (int, float)) and float(rr) >= min_rr

    missing: list[str] = []
    if wnl is None:
        missing.append("WNL")
    if wsl is None:
        missing.append("WSL")
    if wtp is None:
        missing.append("WTP")

    # WNL broken? Compare last_close vs WNL price + side_hint.
    wnl_broken = False
    if wnl is not None and last_close is not None:
        try:
            wnl_price = float(wnl["price"])
            if side_hint == SIDE_BUY:
                wnl_broken = float(last_close) > wnl_price
            elif side_hint == SIDE_SELL:
                wnl_broken = float(last_close) < wnl_price
        except (KeyError, TypeError, ValueError):
            wnl_broken = False

    if missing:
        return IntegratedEvidenceAxis(
            axis="wave_lines", side=side_hint, status=BLOCK,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=True,
            reason_ja=(
                f"W ライン不足 ({'/'.join(missing)})。"
                "損切り / 利確 / エントリー基準のどれかが欠けています。"
            ),
            source="wave_derived_lines",
        )

    if not wnl_broken:
        return IntegratedEvidenceAxis(
            axis="wave_lines", side=side_hint, status=BLOCK,
            strength=0.3, confidence=0.4,
            used_in_decision=True, required=True,
            reason_ja=(
                f"WNL ({float(wnl['price']):.5f}) が未突破。"
                "ブレイク確認まで HOLD。"
            ),
            source="wave_derived_lines",
        )

    if not rr_ok:
        return IntegratedEvidenceAxis(
            axis="wave_lines", side=side_hint, status=BLOCK,
            strength=0.4, confidence=0.5,
            used_in_decision=True, required=True,
            reason_ja=(
                f"RR={rr if rr is not None else 'なし'} が必要値 {min_rr} 未満。"
            ),
            source="wave_derived_lines",
        )

    return IntegratedEvidenceAxis(
        axis="wave_lines", side=side_hint, status=PASS,
        strength=0.9, confidence=0.85,
        used_in_decision=True, required=True,
        reason_ja=(
            f"WNL ({float(wnl['price']):.5f}) 突破 + WSL/WTP あり + RR={rr:.2f}。"
        ),
        source="wave_derived_lines",
    )


def _axis_fibonacci(fib_panel: dict | None) -> IntegratedEvidenceAxis:
    p = _as_dict(fib_panel)
    if not p or not p.get("available"):
        return IntegratedEvidenceAxis(
            axis="fibonacci", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=False,
            reason_ja="フィボナッチ参照が利用できません。",
            source="source_pack_fibonacci",
        )
    side_field = (p.get("side") or "").upper()
    side = SIDE_BUY if side_field == "UP" else SIDE_SELL if side_field == "DOWN" else SIDE_NEUTRAL
    panel_status = p.get("status") or "UNKNOWN"
    if panel_status not in SUPPORTED_AXIS_STATUS:
        panel_status = UNKNOWN
    strength = {PASS: 0.7, WARN: 0.4, BLOCK: 0.0, UNKNOWN: 0.2}[panel_status]
    confidence = 0.5 if panel_status == PASS else 0.3
    reason_ja = p.get("meaning_ja") or "フィボ位置の意味付けは未定です。"
    return IntegratedEvidenceAxis(
        axis="fibonacci", side=side, status=panel_status,
        strength=strength, confidence=confidence,
        used_in_decision=True, required=False,
        reason_ja=reason_ja, source="source_pack_fibonacci",
    )


def _axis_candlestick(candle_panel: dict | None) -> IntegratedEvidenceAxis:
    p = _as_dict(candle_panel)
    if not p or not p.get("available"):
        return IntegratedEvidenceAxis(
            axis="candlestick", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=False,
            reason_ja="ローソク足解剖が利用できません。",
            source="masterclass_candlestick",
        )
    direction = (p.get("direction") or "").upper()
    side = (
        SIDE_BUY if direction == "BUY"
        else SIDE_SELL if direction == "SELL"
        else SIDE_NEUTRAL
    )
    loc = p.get("location_quality") or "neutral"
    bar_type = p.get("bar_type") or ""
    power_chase_risk = bool(p.get("power_chase_risk"))

    if loc in ("against_support", "against_resistance"):
        status, strength = BLOCK, 0.0
        reason_ja = (
            f"ローソク足 {bar_type} が逆位置 ({loc})。"
        )
    elif power_chase_risk:
        status, strength = WARN, 0.3
        reason_ja = (
            f"ローソク足 {bar_type} はサポレジ無しの power move。追いかけ注意。"
        )
    elif loc in ("at_support", "into_resistance") and side != SIDE_NEUTRAL:
        status, strength = PASS, 0.8
        reason_ja = (
            f"ローソク足 {bar_type} が {loc} で {direction} シグナル。"
        )
    elif loc == "midrange":
        status, strength = WARN, 0.3
        reason_ja = (
            f"ローソク足 {bar_type} は midrange (サポレジから離れている)。"
        )
    else:
        status, strength = WARN, 0.4
        reason_ja = f"ローソク足 {bar_type}, location={loc}."

    confidence = 0.7 if status == PASS else 0.4
    return IntegratedEvidenceAxis(
        axis="candlestick", side=side, status=status,
        strength=strength, confidence=confidence,
        used_in_decision=True, required=False,
        reason_ja=reason_ja, source="masterclass_candlestick",
    )


def _axis_dow(dow_panel: dict | None) -> IntegratedEvidenceAxis:
    p = _as_dict(dow_panel)
    if not p or not p.get("available"):
        return IntegratedEvidenceAxis(
            axis="dow_structure", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=True,
            reason_ja="ダウ構造が判定できません。",
            source="masterclass_dow",
        )
    trend = (p.get("trend") or "").upper()
    side = (
        SIDE_BUY if trend == "UP"
        else SIDE_SELL if trend == "DOWN"
        else SIDE_NEUTRAL
    )
    if trend in ("UP", "DOWN"):
        status = PASS
        strength = 0.8
        reason_ja = (
            f"ダウ構造 = {trend}。"
            f"反転確認価格 {p.get('reversal_confirmation_price')}。"
        )
    elif trend in ("RANGE", "MIXED"):
        status = WARN
        strength = 0.2
        reason_ja = f"ダウ構造 = {trend}。トレンドが明確ではありません。"
    else:
        status = UNKNOWN
        strength = 0.0
        reason_ja = "ダウ構造が判定できません。"
    confidence = 0.8 if status == PASS else 0.4
    return IntegratedEvidenceAxis(
        axis="dow_structure", side=side, status=status,
        strength=strength, confidence=confidence,
        used_in_decision=True, required=True,
        reason_ja=reason_ja, source="masterclass_dow",
    )


def _axis_levels(levels_panel: dict | None) -> IntegratedEvidenceAxis:
    p = _as_dict(levels_panel)
    if not p or not p.get("available"):
        return IntegratedEvidenceAxis(
            axis="levels", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=False,
            reason_ja="サポレジ評価が利用できません。",
            source="masterclass_levels",
        )
    near_support = bool(p.get("near_strong_support") or p.get("near_support"))
    near_resistance = bool(p.get("near_strong_resistance") or p.get("near_resistance"))
    if near_support and not near_resistance:
        side, status, strength = SIDE_BUY, PASS, 0.7
        reason_ja = "強いサポート帯付近です。"
    elif near_resistance and not near_support:
        side, status, strength = SIDE_SELL, PASS, 0.7
        reason_ja = "強いレジスタンス帯付近です。"
    else:
        side, status, strength = SIDE_NEUTRAL, WARN, 0.3
        reason_ja = "サポート・レジスタンスから等距離 / 不明確です。"
    return IntegratedEvidenceAxis(
        axis="levels", side=side, status=status,
        strength=strength, confidence=0.6,
        used_in_decision=True, required=False,
        reason_ja=reason_ja, source="masterclass_levels",
    )


def _axis_ma(
    ma_panel: dict | None, granville_panel: dict | None,
) -> IntegratedEvidenceAxis:
    ma = _as_dict(ma_panel)
    gv = _as_dict(granville_panel)
    if not ma or not ma.get("available"):
        return IntegratedEvidenceAxis(
            axis="ma_context", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=False,
            reason_ja="MA コンテキストが利用できません。",
            source="masterclass_indicators_ma",
        )
    sma_slope = (ma.get("sma_20_slope_label") or "").lower()
    granville_label = (gv.get("granville_label") or "").lower()
    trap = "trap" in granville_label
    if "buy" in granville_label and "trend_pullback" in granville_label:
        side, status, strength = SIDE_BUY, PASS, 0.7
        reason_ja = "MA 押し目買い (グランビル trend pullback BUY)。"
    elif "sell" in granville_label and "trend_pullback" in granville_label:
        side, status, strength = SIDE_SELL, PASS, 0.7
        reason_ja = "MA 戻り売り (グランビル trend pullback SELL)。"
    elif trap:
        side = SIDE_NEUTRAL
        status = WARN
        strength = 0.2
        reason_ja = f"MA トラップシグナル ({granville_label})。"
    elif "rising" in sma_slope:
        side, status, strength = SIDE_BUY, WARN, 0.4
        reason_ja = "SMA 上昇傾斜。BUY バイアスを支持。"
    elif "falling" in sma_slope:
        side, status, strength = SIDE_SELL, WARN, 0.4
        reason_ja = "SMA 下降傾斜。SELL バイアスを支持。"
    else:
        side, status, strength = SIDE_NEUTRAL, WARN, 0.3
        reason_ja = "MA / グランビル評価は中立です。"
    return IntegratedEvidenceAxis(
        axis="ma_context", side=side, status=status,
        strength=strength, confidence=0.5,
        used_in_decision=True, required=False,
        reason_ja=reason_ja, source="masterclass_indicators_ma",
    )


def _axis_rsi(rsi_panel: dict | None) -> IntegratedEvidenceAxis:
    """RSI is regime-aware. Never produces BUY/SELL alone — always WARN
    or UNKNOWN. Caution-only.
    """
    p = _as_dict(rsi_panel)
    if not p or not p.get("available"):
        return IntegratedEvidenceAxis(
            axis="rsi_regime", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=False,
            reason_ja="RSI レジーム評価が利用できません。",
            source="masterclass_indicators_rsi",
        )
    regime = (p.get("regime") or "").upper()
    rsi_state = (p.get("rsi_state") or "").lower()
    # Regime-aware: in TREND, overbought/oversold are NOT reversal signals
    if regime == "TREND" and rsi_state in ("overbought", "oversold"):
        return IntegratedEvidenceAxis(
            axis="rsi_regime", side=SIDE_NEUTRAL, status=WARN,
            strength=0.0, confidence=0.4,
            used_in_decision=True, required=False,
            reason_ja=(
                f"TREND 環境で RSI {rsi_state}。それだけでは逆張り根拠に"
                "なりません (caution)。"
            ),
            source="masterclass_indicators_rsi",
        )
    if regime == "RANGE" and rsi_state == "oversold":
        return IntegratedEvidenceAxis(
            axis="rsi_regime", side=SIDE_BUY, status=WARN,
            strength=0.3, confidence=0.4,
            used_in_decision=True, required=False,
            reason_ja="RANGE で RSI oversold。BUY 側補助 (alone では不十分)。",
            source="masterclass_indicators_rsi",
        )
    if regime == "RANGE" and rsi_state == "overbought":
        return IntegratedEvidenceAxis(
            axis="rsi_regime", side=SIDE_SELL, status=WARN,
            strength=0.3, confidence=0.4,
            used_in_decision=True, required=False,
            reason_ja="RANGE で RSI overbought。SELL 側補助 (alone では不十分)。",
            source="masterclass_indicators_rsi",
        )
    return IntegratedEvidenceAxis(
        axis="rsi_regime", side=SIDE_NEUTRAL, status=WARN,
        strength=0.0, confidence=0.3,
        used_in_decision=True, required=False,
        reason_ja=f"RSI 中立 (regime={regime}, state={rsi_state})。",
        source="masterclass_indicators_rsi",
    )


def _axis_bb(bb_panel: dict | None) -> IntegratedEvidenceAxis:
    p = _as_dict(bb_panel)
    if not p or not p.get("available"):
        return IntegratedEvidenceAxis(
            axis="bollinger", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=False,
            reason_ja="BB ライフサイクル評価が利用できません。",
            source="masterclass_indicators_bb",
        )
    phase = (p.get("phase") or "").lower()
    band_walk = (p.get("band_walk_direction") or "").upper()
    if phase == "expansion" and band_walk == "UP":
        side, status, strength = SIDE_BUY, PASS, 0.5
        reason_ja = "BB エクスパンション + アッパーバンドウォーク。BUY 補助。"
    elif phase == "expansion" and band_walk == "DOWN":
        side, status, strength = SIDE_SELL, PASS, 0.5
        reason_ja = "BB エクスパンション + ロワーバンドウォーク。SELL 補助。"
    elif phase == "reversal":
        side, status, strength = SIDE_NEUTRAL, WARN, 0.2
        reason_ja = "BB リバーサル相。reversal リスクあり。"
    else:
        side, status, strength = SIDE_NEUTRAL, WARN, 0.2
        reason_ja = f"BB phase={phase}, walk={band_walk} (補助のみ)."
    return IntegratedEvidenceAxis(
        axis="bollinger", side=side, status=status,
        strength=strength, confidence=0.4,
        used_in_decision=True, required=False,
        reason_ja=reason_ja, source="masterclass_indicators_bb",
    )


def _axis_macd(macd_panel: dict | None) -> IntegratedEvidenceAxis:
    p = _as_dict(macd_panel)
    if not p or not p.get("available"):
        return IntegratedEvidenceAxis(
            axis="macd", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=False,
            reason_ja="MACD アーキテクチャ評価が利用できません。",
            source="masterclass_indicators_macd",
        )
    zero_line = (p.get("zero_line_position") or "").lower()
    cross = (p.get("recent_cross") or "").lower()
    if zero_line == "above" and cross == "bullish":
        side, status, strength = SIDE_BUY, PASS, 0.5
        reason_ja = "MACD ゼロ線上 + 直近 bullish cross。BUY 補助。"
    elif zero_line == "below" and cross == "bearish":
        side, status, strength = SIDE_SELL, PASS, 0.5
        reason_ja = "MACD ゼロ線下 + 直近 bearish cross。SELL 補助。"
    else:
        side, status, strength = SIDE_NEUTRAL, WARN, 0.2
        reason_ja = f"MACD zero={zero_line}, cross={cross} (補助のみ)."
    return IntegratedEvidenceAxis(
        axis="macd", side=side, status=status,
        strength=strength, confidence=0.4,
        used_in_decision=True, required=False,
        reason_ja=reason_ja, source="masterclass_indicators_macd",
    )


def _axis_divergence(div_panel: dict | None) -> IntegratedEvidenceAxis:
    """Divergence is caution-only — NEVER produces BUY/SELL alone."""
    p = _as_dict(div_panel)
    if not p or not p.get("available"):
        return IntegratedEvidenceAxis(
            axis="divergence", side=SIDE_NEUTRAL, status=UNKNOWN,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=False,
            reason_ja="ダイバージェンス評価が利用できません。",
            source="masterclass_indicators_divergence",
        )
    has_div = bool(p.get("has_divergence") or p.get("divergence_present"))
    div_kind = (p.get("divergence_kind") or "").lower()
    if has_div:
        # Regardless of bullish/bearish — divergence is caution
        return IntegratedEvidenceAxis(
            axis="divergence", side=SIDE_NEUTRAL, status=WARN,
            strength=0.2, confidence=0.3,
            used_in_decision=True, required=False,
            reason_ja=(
                f"ダイバージェンス検出 ({div_kind})。"
                "他根拠と組み合わせて caution として扱います。"
            ),
            source="masterclass_indicators_divergence",
        )
    return IntegratedEvidenceAxis(
        axis="divergence", side=SIDE_NEUTRAL, status=WARN,
        strength=0.0, confidence=0.2,
        used_in_decision=True, required=False,
        reason_ja="顕著なダイバージェンスは検出されません。",
        source="masterclass_indicators_divergence",
    )


def _axis_invalidation_rr(
    inv_panel: dict | None, entry_summary: dict | None, *, min_rr: float,
) -> IntegratedEvidenceAxis:
    """Invalidation + RR. REQUIRED axis. BLOCK if missing."""
    p = _as_dict(inv_panel)
    es = _as_dict(entry_summary)
    if not p.get("available") and not es.get("entry_price"):
        return IntegratedEvidenceAxis(
            axis="invalidation_rr", side=SIDE_NEUTRAL, status=BLOCK,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=True,
            reason_ja=(
                "インバリデーション計画 / エントリーサマリが未設定。HOLD。"
            ),
            source="masterclass_invalidation",
        )
    stop = es.get("stop_price") if es else p.get("stop_price")
    tp = es.get("take_profit") if es else None
    rr = es.get("rr") if es else None
    if stop is None or tp is None:
        return IntegratedEvidenceAxis(
            axis="invalidation_rr", side=SIDE_NEUTRAL, status=BLOCK,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=True,
            reason_ja="損切り / 利確のいずれかが欠けています。HOLD。",
            source="masterclass_invalidation",
        )
    if not isinstance(rr, (int, float)) or float(rr) < min_rr:
        return IntegratedEvidenceAxis(
            axis="invalidation_rr", side=SIDE_NEUTRAL, status=BLOCK,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=True,
            reason_ja=(
                f"RR={rr if rr is not None else 'なし'} が最低基準 {min_rr} 未満。"
            ),
            source="masterclass_invalidation",
        )
    is_structure = (
        p.get("structure_stop_price") is not None
        and p.get("structure_stop_price") == stop
    )
    if is_structure:
        status, strength = PASS, 0.9
        reason_ja = (
            f"構造ベースの損切り ({stop:.5f}) + RR={rr:.2f}。"
        )
    else:
        status, strength = WARN, 0.5
        reason_ja = (
            f"ATR ベースの損切り ({stop:.5f}) + RR={rr:.2f}。"
            "構造由来でないため caution。"
        )
    return IntegratedEvidenceAxis(
        axis="invalidation_rr", side=SIDE_NEUTRAL, status=status,
        strength=strength, confidence=0.85,
        used_in_decision=True, required=True,
        reason_ja=reason_ja, source="masterclass_invalidation",
    )


def _axis_macro(
    macro_score: float | None, macro_strong_against: str | None,
    *, mode: str,
) -> IntegratedEvidenceAxis:
    """Macro alignment. macro_strong_against → BLOCK in both modes
    (regardless of side bias — the integrator decides whether the side
    actually matches). Pure missing → BLOCK in strict, WARN in balanced."""
    if macro_score is None:
        if mode == INTEGRATED_MODE_STRICT:
            status = BLOCK
            reason_ja = "マクロ実データが未接続。strict モードでは HOLD。"
        else:
            status = WARN
            reason_ja = "マクロ実データが未接続。balanced モードでは WARN。"
        return IntegratedEvidenceAxis(
            axis="macro", side=SIDE_NEUTRAL, status=status,
            strength=0.0, confidence=0.0,
            used_in_decision=True, required=(mode == INTEGRATED_MODE_STRICT),
            reason_ja=reason_ja, source="macro_alignment",
        )
    score = float(macro_score)
    against = (macro_strong_against or "UNKNOWN").upper()
    if against in ("BUY", "SELL"):
        return IntegratedEvidenceAxis(
            axis="macro", side=against, status=BLOCK,
            strength=1.0, confidence=0.8,
            used_in_decision=True, required=True,
            reason_ja=(
                f"マクロが {against} 方向に強く逆行 (score={score:.2f})。HOLD。"
            ),
            source="macro_alignment",
        )
    if score >= 0.5:
        side, status, strength = SIDE_BUY, PASS, min(1.0, score)
        reason_ja = f"マクロが BUY 側に整合 (score={score:.2f})."
    elif score <= -0.5:
        side, status, strength = SIDE_SELL, PASS, min(1.0, abs(score))
        reason_ja = f"マクロが SELL 側に整合 (score={score:.2f})."
    else:
        side, status, strength = SIDE_NEUTRAL, WARN, 0.3
        reason_ja = f"マクロは中立 (score={score:.2f})."
    return IntegratedEvidenceAxis(
        axis="macro", side=side, status=status,
        strength=strength, confidence=0.6,
        used_in_decision=True, required=False,
        reason_ja=reason_ja, source="macro_alignment",
    )


def _axis_roadmap(
    roadmap_panel: dict | None, *, mode: str,
) -> IntegratedEvidenceAxis:
    p = _as_dict(roadmap_panel)
    available = bool(p.get("available"))
    if not available:
        if mode == INTEGRATED_MODE_STRICT:
            status, reason = BLOCK, "Daily Roadmap データ未接続。strict で HOLD。"
        else:
            status, reason = WARN, "Daily Roadmap データ未接続 (balanced WARN)."
        return IntegratedEvidenceAxis(
            axis="daily_roadmap", side=SIDE_NEUTRAL, status=status,
            strength=0.0, confidence=0.0,
            used_in_decision=True,
            required=(mode == INTEGRATED_MODE_STRICT),
            reason_ja=reason, source="source_pack_daily_roadmap",
        )
    return IntegratedEvidenceAxis(
        axis="daily_roadmap", side=SIDE_NEUTRAL, status=PASS,
        strength=0.5, confidence=0.5,
        used_in_decision=True, required=False,
        reason_ja="Daily Roadmap チェックリスト到達。",
        source="source_pack_daily_roadmap",
    )


def _axis_symbol_briefing(
    briefing_panel: dict | None, *, mode: str,
) -> IntegratedEvidenceAxis:
    p = _as_dict(briefing_panel)
    available = bool(p.get("available"))
    if not available:
        if mode == INTEGRATED_MODE_STRICT:
            status, reason = BLOCK, "Symbol macro briefing 未接続。strict で HOLD。"
        else:
            status, reason = WARN, "Symbol macro briefing 未接続 (balanced WARN)."
        return IntegratedEvidenceAxis(
            axis="symbol_macro_briefing", side=SIDE_NEUTRAL, status=status,
            strength=0.0, confidence=0.0,
            used_in_decision=True,
            required=(mode == INTEGRATED_MODE_STRICT),
            reason_ja=reason, source="source_pack_symbol_briefing",
        )
    return IntegratedEvidenceAxis(
        axis="symbol_macro_briefing", side=SIDE_NEUTRAL, status=PASS,
        strength=0.5, confidence=0.5,
        used_in_decision=True, required=False,
        reason_ja="Symbol macro briefing 到達。",
        source="source_pack_symbol_briefing",
    )


# ── Side-bias resolution ─────────────────────────────────────────
_DIRECTIONAL_AXES: Final[frozenset] = frozenset({
    "wave_pattern", "wave_lines", "dow_structure", "candlestick",
    "fibonacci", "levels", "ma_context",
})

_SIDE_VOTING_WEIGHTS: Final[dict[str, float]] = {
    "wave_pattern": 1.4,
    "wave_lines": 1.6,
    "dow_structure": 1.5,
    "candlestick": 1.0,
    "levels": 0.8,
    "fibonacci": 0.8,
    "ma_context": 0.6,
}


def _resolve_side_bias(axes: list[IntegratedEvidenceAxis]) -> str:
    """Vote-based side bias from directional, used-in-decision axes
    that are PASS or WARN. BLOCK and UNKNOWN don't vote."""
    buy_score = 0.0
    sell_score = 0.0
    for ax in axes:
        if ax.axis not in _DIRECTIONAL_AXES:
            continue
        if not ax.used_in_decision:
            continue
        if ax.status not in (PASS, WARN):
            continue
        weight = _SIDE_VOTING_WEIGHTS.get(ax.axis, 0.5)
        contribution = weight * (1.0 if ax.status == PASS else 0.4) * ax.strength
        if ax.side == SIDE_BUY:
            buy_score += contribution
        elif ax.side == SIDE_SELL:
            sell_score += contribution
    if buy_score > sell_score and buy_score >= 1.0:
        return SIDE_BUY
    if sell_score > buy_score and sell_score >= 1.0:
        return SIDE_SELL
    return SIDE_NEUTRAL


def _compute_confidence(
    axes: list[IntegratedEvidenceAxis], side_bias: str,
) -> float:
    if side_bias == SIDE_NEUTRAL:
        return 0.0
    pass_count = sum(
        1 for a in axes
        if a.status == PASS and a.used_in_decision and (
            a.side == side_bias or not a.side or a.side == SIDE_NEUTRAL
        )
    )
    block_count = sum(
        1 for a in axes
        if a.status == BLOCK and a.used_in_decision
    )
    raw = 0.5 + 0.08 * pass_count - 0.15 * block_count
    return float(max(0.0, min(0.95, raw)))


# ── Audit panel auto-build (when caller didn't provide them) ─────
def _build_panels_from_df(
    *,
    df_window: pd.DataFrame,
    atr_value: float,
    last_close: float,
    higher_timeframe_trend: str | None,
    symbol: str,
    macro_context: dict | None,
    stop_mode: str,
    stop_atr_mult: float,
    tp_atr_mult: float,
    pre_supplied: dict | None = None,
) -> dict:
    """Build the minimum panel set used by the integrated decision."""
    pre = pre_supplied or {}

    chart_pattern = pre.get("chart_pattern_review")
    if chart_pattern is None:
        cp = detect_patterns(
            df_window, atr_value=atr_value, last_close=last_close,
        )
        chart_pattern = (
            cp.selected_patterns_top5[0].to_dict()
            if cp.selected_patterns_top5 else
            {"available": False}
        )
    sr_snapshot = detect_levels(
        df_window, atr_value=atr_value, last_close=last_close,
    )
    near_support = bool(sr_snapshot.near_strong_support)
    near_resistance = bool(sr_snapshot.near_strong_resistance)

    # Auto-built panels (df-only). Pre-supplied wins.
    candle = pre.get("candlestick_anatomy_review") or build_candlestick_anatomy_review(
        visible_df=df_window, atr_value=atr_value,
        near_support=near_support, near_resistance=near_resistance,
        higher_tf_trend=higher_timeframe_trend,
    )
    dow = pre.get("dow_structure_review") or build_dow_structure_review(
        visible_df=df_window,
    )
    # Indicator-derived panels need pre-computed sma/rsi/macd/bb. The
    # caller (visual_audit / tests) is expected to compute and pass
    # them in via `audit_panels`. When absent, they remain None and the
    # corresponding axis becomes UNKNOWN.
    fib = pre.get("fibonacci_context_review")
    levels = pre.get("level_psychology_review")
    ma_panel = pre.get("ma_context_review")
    granville = pre.get("granville_entry_review")
    rsi = pre.get("rsi_regime_filter")
    bb = pre.get("bollinger_lifecycle_review")
    macd = pre.get("macd_architecture_review")
    divergence = pre.get("divergence_review")

    # Build entry_summary if not provided so invalidation can run
    entry_summary = pre.get("entry_summary")
    if entry_summary is None:
        side_hint = _classify_pattern_side(
            (chart_pattern or {}).get("detected_pattern")
            or (chart_pattern or {}).get("kind")
        )
        plan_side = SIDE_BUY if side_hint == SIDE_BUY else SIDE_SELL
        plan = plan_stop(
            mode=stop_mode, side=plan_side, entry=last_close,
            atr=atr_value, stop_atr_mult=stop_atr_mult,
            tp_atr_mult=tp_atr_mult,
            structure_stop_price=None,
        )
        stop_price = plan.stop_price
        tp_price = plan.take_profit_price
        if (
            stop_price is None or tp_price is None
            or stop_price == last_close
        ):
            rr_v = None
        else:
            rr_v = abs(tp_price - last_close) / abs(last_close - stop_price)
        entry_summary = {
            "entry_price": float(last_close),
            "stop_price": float(stop_price) if stop_price is not None else None,
            "take_profit": float(tp_price) if tp_price is not None else None,
            "rr": float(rr_v) if rr_v is not None else None,
            "structure_stop_price": None,
            "atr_stop_price": float(plan.atr_stop_price),
        }

    invalidation = pre.get("invalidation_engine_v2")  # caller-supplied or None

    macro_snap = compute_macro_alignment(
        symbol=symbol, macro_context=macro_context,
    )

    return {
        "chart_pattern_review": chart_pattern,
        "wave_derived_lines": pre.get("wave_derived_lines") or [],
        "fibonacci_context_review": fib,
        "candlestick_anatomy_review": candle,
        "dow_structure_review": dow,
        "level_psychology_review": levels,
        "ma_context_review": ma_panel,
        "granville_entry_review": granville,
        "rsi_regime_filter": rsi,
        "bollinger_lifecycle_review": bb,
        "macd_architecture_review": macd,
        "divergence_review": divergence,
        "invalidation_engine_v2": invalidation,
        "entry_summary": entry_summary,
        "daily_roadmap_review": pre.get("daily_roadmap_review"),
        "symbol_macro_briefing_review": pre.get("symbol_macro_briefing_review"),
        "_macro_score": float(macro_snap.macro_score),
        "_macro_strong_against": macro_snap.macro_strong_against,
    }


# ── Public API ────────────────────────────────────────────────────
def decide_royal_road_v2_integrated(
    *,
    df_window: pd.DataFrame,
    technical_confluence: dict,
    pattern: PatternResult | None,
    higher_timeframe_trend: str | None,
    risk_reward: float | None,
    risk_state: RiskState,
    atr_value: float,
    last_close: float,
    symbol: str,
    macro_context: dict | None,
    df_lower_tf: pd.DataFrame | None,
    lower_tf_interval: str | None,
    stop_mode: str = DEFAULT_STOP_MODE,
    stop_atr_mult: float = 2.0,
    tp_atr_mult: float = 3.0,
    base_bar_close_ts: pd.Timestamp,
    mode: str = DEFAULT_INTEGRATED_MODE,
    min_risk_reward: float = MIN_RISK_REWARD,
    audit_panels: dict | None = None,
) -> Decision:
    """Build evidence axes from audit panels and decide BUY/SELL/HOLD.

    The function computes every evidence axis (or accepts pre-built
    panels via `audit_panels`) and applies the integrated decision
    rule chain:

      1. Risk gate (HOLD on any block)
      2. Required axes — wave_lines / dow_structure / invalidation_rr
         must be PASS or WARN (BLOCK → HOLD)
      3. Macro_strong_against → HOLD when it agrees with the would-be
         side bias
      4. Side-bias resolution from directional PASS/WARN axes
      5. Action = side_bias when side_bias != NEUTRAL and no block
         reasons; else HOLD
    """
    canonical_mode = validate_integrated_mode(mode)
    chain: list[str] = ["risk_gate", f"integrated_mode:{canonical_mode}"]
    block_reasons: list[str] = []
    cautions: list[str] = []

    # 1. Risk gate
    gate = evaluate_gate(risk_state)
    if not gate.allow_trade:
        codes = tuple(gate.blocked_codes)
        integrated = empty_integrated_decision(
            mode=canonical_mode,
            label="HOLD_RISK_GATE",
            reason_ja=f"リスクゲートで HOLD: {','.join(codes)}",
        )
        return _hold(
            reason="risk gate blocked",
            blocked_by=codes,
            chain=tuple(chain),
            confidence=0.0,
            advisory={
                "profile": PROFILE_NAME_V2_INTEGRATED,
                "mode": canonical_mode,
                "integrated_decision": integrated.to_dict(),
            },
        )

    # 2. Build panel set (or use pre-supplied)
    panels = _build_panels_from_df(
        df_window=df_window,
        atr_value=atr_value,
        last_close=last_close,
        higher_timeframe_trend=higher_timeframe_trend,
        symbol=symbol,
        macro_context=macro_context,
        stop_mode=stop_mode,
        stop_atr_mult=stop_atr_mult,
        tp_atr_mult=tp_atr_mult,
        pre_supplied=audit_panels,
    )

    # 3. Build axes
    axes: list[IntegratedEvidenceAxis] = []
    axes.append(_axis_wave_pattern(panels["chart_pattern_review"]))
    axes.append(_axis_wave_lines(
        wave_lines=panels["wave_derived_lines"],
        chart_pattern=panels["chart_pattern_review"],
        last_close=last_close,
        entry_summary=panels["entry_summary"],
        min_rr=min_risk_reward,
    ))
    axes.append(_axis_fibonacci(panels["fibonacci_context_review"]))
    axes.append(_axis_candlestick(panels["candlestick_anatomy_review"]))
    axes.append(_axis_dow(panels["dow_structure_review"]))
    axes.append(_axis_levels(panels["level_psychology_review"]))
    axes.append(_axis_ma(
        panels["ma_context_review"], panels["granville_entry_review"],
    ))
    axes.append(_axis_rsi(panels["rsi_regime_filter"]))
    axes.append(_axis_bb(panels["bollinger_lifecycle_review"]))
    axes.append(_axis_macd(panels["macd_architecture_review"]))
    axes.append(_axis_divergence(panels["divergence_review"]))
    axes.append(_axis_invalidation_rr(
        panels["invalidation_engine_v2"], panels["entry_summary"],
        min_rr=min_risk_reward,
    ))
    axes.append(_axis_macro(
        panels.get("_macro_score"),
        panels.get("_macro_strong_against"),
        mode=canonical_mode,
    ))
    axes.append(_axis_roadmap(
        panels["daily_roadmap_review"], mode=canonical_mode,
    ))
    axes.append(_axis_symbol_briefing(
        panels["symbol_macro_briefing_review"], mode=canonical_mode,
    ))

    # 4. Required-axis check
    for ax in axes:
        if ax.required and ax.status == BLOCK:
            block_reasons.append(f"{ax.axis}_blocked")
        elif ax.required and ax.status == UNKNOWN:
            block_reasons.append(f"{ax.axis}_unknown")
        elif ax.status == BLOCK and ax.used_in_decision and ax.axis == "macro":
            # macro BLOCK is always blocking
            block_reasons.append("macro_strong_against")

    # 5. Side-bias resolution
    side_bias = _resolve_side_bias(axes)

    # 6. Counter-trend block: dow says one side, would-be action says other
    dow_axis = next((a for a in axes if a.axis == "dow_structure"), None)
    if (
        dow_axis is not None
        and dow_axis.side != SIDE_NEUTRAL
        and side_bias != SIDE_NEUTRAL
        and dow_axis.side != side_bias
        and dow_axis.status == PASS
    ):
        block_reasons.append("dow_counter_trend")

    # 7. Pattern-side counter to side_bias
    wp_axis = next((a for a in axes if a.axis == "wave_pattern"), None)
    if (
        wp_axis is not None
        and wp_axis.side != SIDE_NEUTRAL
        and side_bias != SIDE_NEUTRAL
        and wp_axis.side != side_bias
        and wp_axis.status in (PASS, WARN)
    ):
        block_reasons.append("pattern_counter_to_side_bias")

    # 8. Cautions = WARN axes that would otherwise reduce confidence
    for ax in axes:
        if ax.status == WARN and ax.used_in_decision:
            cautions.append(f"{ax.axis}_warn")

    # 9. Decide action
    if block_reasons or side_bias == SIDE_NEUTRAL:
        action = "HOLD"
    else:
        action = side_bias

    # 10. Confidence + label + explanation
    confidence = _compute_confidence(axes, side_bias) if action != "HOLD" else 0.0
    if action == "HOLD":
        if block_reasons:
            label = "HOLD_BLOCKED"
        else:
            label = "HOLD_NEUTRAL"
    else:
        label = f"{action}_INTEGRATED"

    used_modules = sorted({a.source for a in axes if a.used_in_decision and a.status == PASS})
    audit_only_modules = sorted({a.source for a in axes if a.used_in_decision and a.status == WARN})
    not_connected_modules = sorted({a.source for a in axes if a.status in (BLOCK, UNKNOWN) and a.used_in_decision})

    explanation_ja = _compose_explanation_ja(
        action=action, side_bias=side_bias,
        block_reasons=block_reasons, cautions=cautions,
        axes=axes, mode=canonical_mode,
    )

    integrated = RoyalRoadIntegratedDecision(
        action=action, label=label, side_bias=side_bias,
        confidence=confidence, mode=canonical_mode,
        block_reasons=block_reasons, cautions=cautions,
        axes=axes,
        used_modules=used_modules,
        audit_only_modules=audit_only_modules,
        not_connected_modules=not_connected_modules,
        explanation_ja=explanation_ja,
    )

    # Structure_stop_plan compatible with the v2 trace slice, so
    # visual_audit can render the entry/stop/TP/RR card without the
    # integrated profile needing to populate every v2 advisory field.
    es = panels.get("entry_summary") or {}
    structure_stop_plan = (
        {
            "chosen_mode": "structure" if es.get("structure_stop_price") else "atr",
            "outcome": "ok" if es.get("rr") and es["rr"] >= min_risk_reward else "needs_review",
            "stop_price": es.get("stop_price"),
            "take_profit_price": es.get("take_profit"),
            "rr_realized": es.get("rr"),
            "structure_stop_price": es.get("structure_stop_price"),
            "atr_stop_price": es.get("atr_stop_price"),
        }
        if es.get("entry_price") is not None
        else None
    )

    advisory = {
        "profile": PROFILE_NAME_V2_INTEGRATED,
        "mode": canonical_mode,
        "integrated_decision": integrated.to_dict(),
        "side_bias": side_bias,
        "block_reasons": list(block_reasons),
        "cautions": list(cautions),
        # v2 trace compatibility — visual_audit reads these.
        "structure_stop_plan": structure_stop_plan,
        "evidence_axes": {
            ax.axis: ax.to_dict() for ax in axes
        },
        "support_resistance_v2": {},
        "trendline_context": {},
        "chart_pattern_v2": panels.get("chart_pattern_review") or {},
        "lower_tf_trigger": {},
        "macro_alignment": {
            "macro_score": panels.get("_macro_score") or 0.0,
            "macro_strong_against": panels.get("_macro_strong_against") or "UNKNOWN",
        },
        "setup_candidates": [],
        "best_setup": None,
        "reconstruction_quality": {"total_reconstruction_score": float(integrated.confidence)},
        "multi_scale_chart": {},
    }

    chain.extend([f"side_bias:{side_bias}", f"action:{action}"])
    if action == "HOLD":
        return _hold(
            reason="; ".join(block_reasons) if block_reasons else f"side_bias={side_bias}",
            blocked_by=tuple(block_reasons) if block_reasons else (f"side_bias_{side_bias}",),
            chain=tuple(chain),
            confidence=confidence,
            advisory=advisory,
        )
    return Decision(
        action=action,
        confidence=confidence,
        reason=f"integrated {action} on side_bias={side_bias}",
        blocked_by=(),
        rule_chain=tuple(chain),
        advisory=advisory,
    )


def _compose_explanation_ja(
    *, action: str, side_bias: str,
    block_reasons: list[str], cautions: list[str],
    axes: list[IntegratedEvidenceAxis], mode: str,
) -> str:
    if action == "HOLD":
        if block_reasons:
            joined = ", ".join(block_reasons)
            return (
                f"HOLD: {joined}. mode={mode}."
                "\n各 evidence axis を確認して、不足条件が満たされるまで待機。"
            )
        return (
            f"HOLD: 方向性が不明 (side_bias={side_bias}). mode={mode}."
            "\nダウ構造 / 波形 / W ライン / candle のいずれかが PASS で揃って"
            "いません。"
        )
    pass_axes = [a for a in axes if a.status == PASS and a.used_in_decision]
    pass_summary = ", ".join(a.axis for a in pass_axes) or "(none)"
    caution_summary = ", ".join(cautions) if cautions else "(none)"
    return (
        f"{action} integrated decision (mode={mode}). "
        f"PASS axes: {pass_summary}. cautions: {caution_summary}."
    )


__all__ = [
    "PROFILE_NAME_V2_INTEGRATED",
    "INTEGRATED_MODE_STRICT",
    "INTEGRATED_MODE_BALANCED",
    "DEFAULT_INTEGRATED_MODE",
    "SUPPORTED_INTEGRATED_MODES",
    "PASS",
    "WARN",
    "BLOCK",
    "UNKNOWN",
    "SUPPORTED_AXIS_STATUS",
    "SIDE_BUY",
    "SIDE_SELL",
    "SIDE_NEUTRAL",
    "IntegratedEvidenceAxis",
    "RoyalRoadIntegratedDecision",
    "empty_integrated_decision",
    "validate_integrated_mode",
    "decide_royal_road_v2_integrated",
]
