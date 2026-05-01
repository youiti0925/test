# royal_road_decision_v1 Thresholds

Created: 2026-05-01

This document exists because the first `royal_road_decision_v1` implementation mixed the royal-road idea with hard-coded heuristic thresholds. Those thresholds are not final trading rules.

## Status

- `technical_confluence_v1` is the observation layer.
- `royal_road_decision_v1` is an opt-in research decision profile.
- The current thresholds are heuristic and require validation.
- The profile must not be treated as the completed royal-road system.

## Threshold classification

| Setting | Current value | Source | Status |
|---|---:|---|---|
| RSI 70/30 | 70 / 30 | common default / existing baseline | established default |
| MACD | 12 / 26 / 9 | common default / existing baseline | established default |
| Bollinger | 20 / 2.0 | common default / existing baseline | established default |
| min risk reward | 1.5 | existing decision engine | existing runtime value |
| near level distance | 0.5 ATR | heuristic | not validated |
| structure stop bucket | 0.5 / 1.5 / 3.0 ATR | heuristic | not validated |
| STRONG-only entry | strict heuristic | assistant/implementation choice | not validated |
| WEAK setup forced HOLD | strict heuristic | assistant/implementation choice | not validated |
| serious avoid reasons | fake_breakout / regime_volatile | heuristic | not validated |

## Required correction

The decision profile should separate the royal-road procedure from the chosen strictness level.

Recommended modes:

- `strict`: diagnostic, STRONG setups only.
- `balanced`: default research candidate; allows WEAK setups when support/resistance, candle, structure, and risk plan align.
- `exploratory`: discovery mode for finding better-than-royal-road combinations; not for adoption.

## Guardrail

Any result from the current strict encoding must be described as a strict heuristic profile result, not as a completed royal-road result.
