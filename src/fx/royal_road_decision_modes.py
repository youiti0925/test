"""Mode configuration for royal_road_decision_v1.

The royal-road process and the numeric/code thresholds used to encode it
must be separate. These modes are research profiles, not final trading
rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


DEFAULT_ROYAL_ROAD_MODE: Final[str] = "balanced"


@dataclass(frozen=True)
class RoyalRoadDecisionModeConfig:
    name: str
    description: str
    status: str
    needs_validation: bool
    allow_weak_entries: bool
    require_invalidation_clear: bool
    require_structure_stop: bool
    allow_structure_stop_soft_fail: bool
    min_evidence_axes_strong: int
    min_evidence_axes_weak: int
    block_opposite_level: bool
    htf_counter_trend_blocks: bool
    serious_avoid_reasons: tuple[str, ...]


ROYAL_ROAD_MODE_CONFIGS: Final[dict[str, RoyalRoadDecisionModeConfig]] = {
    "strict": RoyalRoadDecisionModeConfig(
        name="strict",
        description="Diagnostic strict mode. STRONG setups only; hard invalidation and structure-stop requirements.",
        status="heuristic_not_validated",
        needs_validation=True,
        allow_weak_entries=False,
        require_invalidation_clear=True,
        require_structure_stop=True,
        allow_structure_stop_soft_fail=False,
        min_evidence_axes_strong=1,
        min_evidence_axes_weak=99,
        block_opposite_level=True,
        htf_counter_trend_blocks=True,
        serious_avoid_reasons=("fake_breakout", "regime_volatile"),
    ),
    "balanced": RoyalRoadDecisionModeConfig(
        name="balanced",
        description="Default research mode. Allows WEAK setups only when multiple royal-road evidence axes agree.",
        status="heuristic_not_validated_default",
        needs_validation=True,
        allow_weak_entries=True,
        require_invalidation_clear=False,
        require_structure_stop=False,
        allow_structure_stop_soft_fail=True,
        min_evidence_axes_strong=1,
        min_evidence_axes_weak=2,
        block_opposite_level=True,
        htf_counter_trend_blocks=True,
        serious_avoid_reasons=("fake_breakout", "regime_volatile"),
    ),
    "exploratory": RoyalRoadDecisionModeConfig(
        name="exploratory",
        description="Discovery mode. Broadens WEAK setup inclusion for candidate discovery only.",
        status="exploratory_not_validated",
        needs_validation=True,
        allow_weak_entries=True,
        require_invalidation_clear=False,
        require_structure_stop=False,
        allow_structure_stop_soft_fail=True,
        min_evidence_axes_strong=1,
        min_evidence_axes_weak=1,
        block_opposite_level=True,
        htf_counter_trend_blocks=True,
        serious_avoid_reasons=("fake_breakout",),
    ),
}


def supported_royal_road_modes() -> tuple[str, ...]:
    return tuple(ROYAL_ROAD_MODE_CONFIGS)


def get_royal_road_mode_config(mode: str = DEFAULT_ROYAL_ROAD_MODE) -> RoyalRoadDecisionModeConfig:
    if mode not in ROYAL_ROAD_MODE_CONFIGS:
        raise ValueError(
            f"unknown royal-road decision mode: {mode!r}. Supported: {', '.join(supported_royal_road_modes())}"
        )
    return ROYAL_ROAD_MODE_CONFIGS[mode]


__all__ = [
    "DEFAULT_ROYAL_ROAD_MODE",
    "RoyalRoadDecisionModeConfig",
    "ROYAL_ROAD_MODE_CONFIGS",
    "supported_royal_road_modes",
    "get_royal_road_mode_config",
]
