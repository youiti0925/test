# royal_road_decision_v1

Created: 2026-05-01.
Branch: `feat/royal-road-decision-v1`.

## Purpose

`royal_road_decision_v1` is an opt-in decision profile that uses
`technical_confluence_v1` to produce BUY / SELL / HOLD decisions.

It is not the completed royal-road system. It is a research profile that
separates:

- the royal-road procedure,
- the current heuristic thresholds,
- the strictness mode used to translate evidence into action.

The default engine profile remains `current_runtime`.

## Important correction

The first version hard-coded a strict interpretation:

- STRONG setups only,
- WEAK setups always HOLD,
- invalidation_clear required,
- structure_stop required.

That created an all-HOLD risk and could be mistaken for a test of the
entire royal-road method. The implementation now has explicit modes in
`src/fx/royal_road_decision_modes.py`.

## Modes

### strict

Diagnostic mode. Close to the first implementation.

- STRONG_BUY / STRONG_SELL only.
- WEAK_BUY / WEAK_SELL are HOLD.
- invalidation_clear is required.
- structure_stop is required.
- near_resistance blocks BUY.
- near_support blocks SELL.

Use this only to study conservative filtering.

### balanced

Default research mode.

- STRONG setups can enter with at least one royal-road evidence axis.
- WEAK setups can enter only when at least two evidence axes agree.
- Missing invalidation or structure_stop becomes a caution, not an automatic block.
- near_resistance still blocks BUY.
- near_support still blocks SELL.
- fake_breakout / regime_volatile still block.

This is the standard candidate for further analysis.

### exploratory

Discovery mode.

- WEAK setups can enter with broader evidence.
- Intended only to discover promising condition combinations.
- Not for adoption.

## Evidence axes

BUY evidence examples:

- bullish_structure
- near_support
- role_reversal_support_candidate
- bullish_candlestick
- bullish_neckline_break

SELL evidence examples:

- bearish_structure
- near_resistance
- role_reversal_resistance_candidate
- bearish_candlestick
- bearish_neckline_break

## Advisory output

Each royal-road Decision advisory includes:

```json
{
  "profile": "royal_road_decision_v1",
  "mode": "balanced",
  "mode_status": "heuristic_not_validated_default",
  "mode_needs_validation": true,
  "score": 0.0,
  "reasons": [],
  "block_reasons": [],
  "cautions": [],
  "source": "technical_confluence_v1"
}
```

## What this still does not cover

- Full horizontal support/resistance clustering.
- Trendlines.
- Full head-and-shoulders / inverse head-and-shoulders detection.
- Flag / wedge / triangle detection.
- Lower-timeframe 15m / 5m trigger.
- DXY / VIX / yield direction alignment as a strong decision input.
- Actual structure-stop order placement.

## Safety guardrails

- `current_runtime` remains default.
- `decision_engine.py`, `risk_gate.py`, and `indicators.py` are unchanged.
- live / OANDA / paper paths are not connected.
- thresholds are documented in `docs/royal_road_thresholds_v1.md`.
- current mode thresholds are heuristic and need validation.
- 90d results from strict mode must not be described as a completed royal-road evaluation.

## Next implementation direction

Before treating royal-road results as meaningful, implement the missing
royal-road components, especially:

1. support/resistance clustering,
2. structure stop execution comparison,
3. chart patterns,
4. macro alignment,
5. lower-timeframe triggers.
