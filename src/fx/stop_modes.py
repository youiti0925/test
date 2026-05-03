"""Stop-placement modes for backtest engine — opt-in, default unchanged.

Modes:
  - `atr` (default)         : stop = entry +/- (stop_atr_mult * ATR).
                              Byte-identical to PR #21 main when
                              chosen (which is the default).
  - `structure`             : stop = structure_stop_price (the most
                              recent confirmed swing low for BUY, swing
                              high for SELL). Returns None if not
                              available — caller decides whether to
                              HOLD or fall back.
  - `hybrid`                : structure if 0.5..3.0 ATR away, else fall
                              back to ATR. If structure is too close
                              (<0.5 ATR) the trade is invalidated
                              (caller produces HOLD with the
                              `structure_stop_too_close` reason).

These modes apply ONLY in the backtest engine (`run_engine_backtest`),
NEVER in the live cmd_trade / OANDA / paper paths. Pinned by tests.

This module intentionally does not change `risk_gate` or `decide_action`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal


SUPPORTED_STOP_MODES: Final[tuple[str, ...]] = ("atr", "structure", "hybrid")
DEFAULT_STOP_MODE: Final[str] = "atr"

_STRUCTURE_TOO_CLOSE_ATR: Final[float] = 0.5
_STRUCTURE_TOO_FAR_ATR: Final[float] = 3.0


StopMode = Literal["atr", "structure", "hybrid"]
StopOutcome = Literal[
    "atr",
    "structure",
    "hybrid_structure",
    "hybrid_atr_fallback",
    "invalid_too_close",
    "invalid_no_structure",
]


@dataclass(frozen=True)
class StopPlan:
    chosen_mode: StopMode
    outcome: StopOutcome
    stop_price: float | None        # None = invalid; caller HOLDs
    take_profit_price: float | None
    rr_realized: float | None
    structure_stop_price: float | None
    structure_stop_distance_atr: float | None
    atr_stop_price: float
    invalidation_reason: str | None

    def to_dict(self) -> dict:
        return {
            "chosen_mode": self.chosen_mode,
            "outcome": self.outcome,
            "stop_price": (
                float(self.stop_price) if self.stop_price is not None
                else None
            ),
            "take_profit_price": (
                float(self.take_profit_price)
                if self.take_profit_price is not None else None
            ),
            "rr_realized": (
                float(self.rr_realized) if self.rr_realized is not None
                else None
            ),
            "structure_stop_price": (
                float(self.structure_stop_price)
                if self.structure_stop_price is not None else None
            ),
            "structure_stop_distance_atr": (
                float(self.structure_stop_distance_atr)
                if self.structure_stop_distance_atr is not None
                else None
            ),
            "atr_stop_price": float(self.atr_stop_price),
            "invalidation_reason": self.invalidation_reason,
        }


def validate_stop_mode(name: str) -> str:
    if name not in SUPPORTED_STOP_MODES:
        raise ValueError(
            f"unknown --stop-mode: {name!r}. "
            f"Supported: {', '.join(SUPPORTED_STOP_MODES)}"
        )
    return name


def _atr_stop_price(side: str, entry: float, atr: float, mult: float) -> float:
    return entry - mult * atr if side == "BUY" else entry + mult * atr


def _atr_tp_price(side: str, entry: float, atr: float, mult: float) -> float:
    return entry + mult * atr if side == "BUY" else entry - mult * atr


def plan_stop(
    *,
    mode: StopMode,
    side: str,
    entry: float,
    atr: float,
    stop_atr_mult: float,
    tp_atr_mult: float,
    structure_stop_price: float | None,
) -> StopPlan:
    """Resolve the stop / TP plan for one entry.

    `structure_stop_price` is the price suggested by the v2 structure
    layer (most recent confirmed swing low for BUY, swing high for
    SELL). May be None.

    Always populates `atr_stop_price` so trace consumers can compare
    structure vs ATR even when a non-ATR mode is chosen.
    """
    atr_stop = _atr_stop_price(side, entry, atr, stop_atr_mult)
    atr_tp = _atr_tp_price(side, entry, atr, tp_atr_mult)
    atr_rr = float(tp_atr_mult / stop_atr_mult) if stop_atr_mult > 0 else None
    structure_distance_atr = (
        abs(entry - structure_stop_price) / atr
        if (structure_stop_price is not None and atr > 0)
        else None
    )

    if mode == "atr":
        return StopPlan(
            chosen_mode="atr",
            outcome="atr",
            stop_price=float(atr_stop),
            take_profit_price=float(atr_tp),
            rr_realized=atr_rr,
            structure_stop_price=structure_stop_price,
            structure_stop_distance_atr=structure_distance_atr,
            atr_stop_price=float(atr_stop),
            invalidation_reason=None,
        )

    if mode == "structure":
        if structure_stop_price is None:
            return StopPlan(
                chosen_mode="structure",
                outcome="invalid_no_structure",
                stop_price=None,
                take_profit_price=None,
                rr_realized=None,
                structure_stop_price=None,
                structure_stop_distance_atr=None,
                atr_stop_price=float(atr_stop),
                invalidation_reason="structure_stop_missing",
            )
        # TP based on the structure-distance times existing tp/stop
        # ratio. RR is consistent with original ATR-mode RR design.
        ratio = (
            (tp_atr_mult / stop_atr_mult) if stop_atr_mult > 0 else 1.5
        )
        tp_distance = abs(structure_stop_price - entry) * ratio
        tp_price = entry + tp_distance if side == "BUY" else entry - tp_distance
        rr = (
            tp_distance / abs(entry - structure_stop_price)
            if structure_stop_price != entry else None
        )
        return StopPlan(
            chosen_mode="structure",
            outcome="structure",
            stop_price=float(structure_stop_price),
            take_profit_price=float(tp_price),
            rr_realized=rr,
            structure_stop_price=float(structure_stop_price),
            structure_stop_distance_atr=structure_distance_atr,
            atr_stop_price=float(atr_stop),
            invalidation_reason=None,
        )

    if mode == "hybrid":
        if structure_stop_price is None:
            return StopPlan(
                chosen_mode="hybrid",
                outcome="hybrid_atr_fallback",
                stop_price=float(atr_stop),
                take_profit_price=float(atr_tp),
                rr_realized=atr_rr,
                structure_stop_price=None,
                structure_stop_distance_atr=None,
                atr_stop_price=float(atr_stop),
                invalidation_reason="structure_stop_missing_fallback_atr",
            )
        if structure_distance_atr is None:
            # Defensive — should not happen since structure_stop_price
            # is not None.
            return StopPlan(
                chosen_mode="hybrid",
                outcome="hybrid_atr_fallback",
                stop_price=float(atr_stop),
                take_profit_price=float(atr_tp),
                rr_realized=atr_rr,
                structure_stop_price=structure_stop_price,
                structure_stop_distance_atr=None,
                atr_stop_price=float(atr_stop),
                invalidation_reason="atr_zero_fallback",
            )
        if structure_distance_atr < _STRUCTURE_TOO_CLOSE_ATR:
            return StopPlan(
                chosen_mode="hybrid",
                outcome="invalid_too_close",
                stop_price=None,
                take_profit_price=None,
                rr_realized=None,
                structure_stop_price=structure_stop_price,
                structure_stop_distance_atr=structure_distance_atr,
                atr_stop_price=float(atr_stop),
                invalidation_reason="structure_stop_too_close",
            )
        if structure_distance_atr > _STRUCTURE_TOO_FAR_ATR:
            return StopPlan(
                chosen_mode="hybrid",
                outcome="hybrid_atr_fallback",
                stop_price=float(atr_stop),
                take_profit_price=float(atr_tp),
                rr_realized=atr_rr,
                structure_stop_price=structure_stop_price,
                structure_stop_distance_atr=structure_distance_atr,
                atr_stop_price=float(atr_stop),
                invalidation_reason="structure_stop_too_far_fallback_atr",
            )
        # In-band → use structure
        ratio = (
            (tp_atr_mult / stop_atr_mult) if stop_atr_mult > 0 else 1.5
        )
        tp_distance = abs(structure_stop_price - entry) * ratio
        tp_price = entry + tp_distance if side == "BUY" else entry - tp_distance
        rr = (
            tp_distance / abs(entry - structure_stop_price)
            if structure_stop_price != entry else None
        )
        return StopPlan(
            chosen_mode="hybrid",
            outcome="hybrid_structure",
            stop_price=float(structure_stop_price),
            take_profit_price=float(tp_price),
            rr_realized=rr,
            structure_stop_price=float(structure_stop_price),
            structure_stop_distance_atr=structure_distance_atr,
            atr_stop_price=float(atr_stop),
            invalidation_reason=None,
        )

    raise ValueError(f"plan_stop: unknown mode {mode!r}")


__all__ = [
    "DEFAULT_STOP_MODE",
    "SUPPORTED_STOP_MODES",
    "StopPlan",
    "validate_stop_mode",
    "plan_stop",
]
