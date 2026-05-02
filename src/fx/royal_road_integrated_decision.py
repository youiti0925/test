"""royal_road_integrated_decision — opt-in research decision profile.

This module exists to evolve the audit pipeline from "observation only"
into "the audit information IS the decision". The standard
`royal_road_decision_v2` profile keeps the v2 evidence axes only as a
non-blocking scoring input; the integrated profile elevates the
following audit modules into first-class evidence the action is built
from:

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
    cmd_trade --broker oanda does not call run_engine_backtest, so
    the integrated profile cannot leak into live trading.
  - The existing royal_road_decision_v2 / v1 / current_runtime profiles
    are unaffected — this is an additional opt-in profile.
  - Phase A: scaffolding only. The skeleton entrypoint returns
    Decision(HOLD) with reason "integrated_phase_a_scaffold". The real
    evidence builders land in Phase B.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from .decision_engine import Decision, MIN_RISK_REWARD, _hold
from .patterns import PatternResult
from .risk_gate import RiskState
from .stop_modes import DEFAULT_STOP_MODE


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

# Side
SIDE_BUY: Final[str] = "BUY"
SIDE_SELL: Final[str] = "SELL"
SIDE_NEUTRAL: Final[str] = "NEUTRAL"


def validate_integrated_mode(name: str) -> str:
    """Return canonical integrated mode name or raise ValueError.

    Mirrors `validate_royal_road_mode` in style. Used by the CLI in
    Phase C; safe to call from tests in Phase A.
    """
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
    side: str            # BUY / SELL / NEUTRAL
    status: str          # PASS / WARN / BLOCK / UNKNOWN
    strength: float      # 0.0 .. 1.0
    confidence: float    # 0.0 .. 1.0
    used_in_decision: bool
    required: bool
    reason_ja: str
    source: str          # source module / panel name

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
    """Container for the integrated decision result.

    The action / confidence are also returned via the legacy `Decision`
    object so backtest_engine can consume it without changes; this
    container is what visual_audit / decision_bridge read.
    """
    action: str          # BUY / SELL / HOLD
    label: str           # e.g. "STRONG_BUY_INTEGRATED" / "HOLD_PHASE_A_SCAFFOLD"
    side_bias: str       # BUY / SELL / NEUTRAL
    confidence: float
    mode: str            # integrated_balanced / integrated_strict
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
    """Build an empty HOLD integrated decision (used by scaffolds /
    early-exit branches)."""
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


# ── Public API (Phase A: scaffold) ────────────────────────────────
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
) -> Decision:
    """Phase A scaffold: returns Decision(HOLD) with a clear reason.

    The signature mirrors `decide_royal_road_v2` so Phase C can wire
    it into backtest_engine.dispatch with a single elif branch.
    Phase B will replace the body with real evidence builders +
    BUY/SELL/HOLD logic.
    """
    canonical_mode = validate_integrated_mode(mode)
    integrated = empty_integrated_decision(
        mode=canonical_mode,
        label="HOLD_PHASE_A_SCAFFOLD",
        reason_ja=(
            "Phase A scaffold です。波形 / W ライン / フィボ / ローソク足 / "
            "ダウ / MA / RSI / BB / MACD / 損切り / RR / マクロを統合する"
            "実装は Phase B で入ります。"
        ),
    )
    return _hold(
        reason="integrated_phase_a_scaffold_not_yet_wired",
        blocked_by=("integrated_phase_a_scaffold",),
        chain=("integrated", f"mode:{canonical_mode}", "scaffold_only"),
        confidence=0.0,
        advisory={
            "profile": PROFILE_NAME_V2_INTEGRATED,
            "mode": canonical_mode,
            "scaffold": True,
            "integrated_decision": integrated.to_dict(),
        },
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
