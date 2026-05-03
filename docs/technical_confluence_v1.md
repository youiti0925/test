# technical_confluence_v1

Created: 2026-05-01.
Branch (implementation): `feat/technical-confluence-v1-observation`.
Baseline: [`docs/technical_baseline_royal_road.md`](technical_baseline_royal_road.md).
Audit context: [`docs/royal_road_gap_audit_v1.md`](royal_road_gap_audit_v1.md).

## Purpose

`technical_confluence_v1` adds a **trace-only** observation layer that
records royal-road technical features (Dow structure, support /
resistance, candlestick triggers, indicator regime context, structure
stop, vote breakdown, final confluence label) for every backtest bar.

The point of this PR is **not** to change BUY/SELL/HOLD. The point is
to make royal-road features visible per-bar so post-hoc WIN/LOSS
analysis (and, eventually, label-conditional A/B tests) can decide
which features actually separate winners from losers.

## Scope

In:

- New module `src/fx/technical_confluence.py`
- New trace slice `TechnicalConfluenceSlice` (optional field on
  `BarDecisionTrace`)
- Backtest engine emits the slice for every traced bar
- Tests for candlestick / support-resistance / observation-only invariants

Out (deferred to future PRs):

- Connecting the slice to `decide_action`, `risk_gate`, or any
  RuleCheck (PR-D, behind opt-in flag, only after evidence)
- True horizontal-level S/R clustering (current v1 reuses confirmed
  swings as level proxies)
- Head-and-shoulders / flag / wedge / triangle detectors
- 15m / 5m sub-interval triggers (data-path change required)
- live cmd_trade / OANDA / paper export wiring

## Observation-only guarantee

The slice is observation-only **by code**:

| Check | Where |
|---|---|
| Decision engine never imports / references confluence | `tests/test_technical_confluence_trace_observation_only.py::test_decision_engine_does_not_import_technical_confluence` |
| Risk gate never imports / references confluence | same file, `test_risk_gate_does_not_import_technical_confluence` |
| Indicators (technical_signal) never references confluence | same file, `test_indicators_does_not_import_technical_confluence` |
| Engine output byte-identical across runs after wiring | same file, `test_engine_decisions_byte_identical_with_confluence_present` |

Adding or removing fields in this slice **must not** change
`result.trades`, `result.metrics`, or `BarDecisionTrace.decision.final_action`.

## Trace schema overview

`BarDecisionTrace.technical_confluence: TechnicalConfluenceSlice | None`.

When non-None, `to_dict()` produces:

```json
{
  "policy_version": "technical_confluence_v1",
  "market_regime": "TREND_UP|TREND_DOWN|RANGE|VOLATILE|UNKNOWN",
  "dow_structure": {
    "structure_code": "HH|HL|LH|LL|MIXED|UNKNOWN",
    "last_swing_high": float | null,
    "last_swing_low": float | null,
    "bos_up": bool,
    "bos_down": bool
  },
  "support_resistance": {
    "nearest_support": float | null,
    "nearest_resistance": float | null,
    "distance_to_support_atr": float | null,
    "distance_to_resistance_atr": float | null,
    "near_support": bool,
    "near_resistance": bool,
    "breakout": bool,
    "pullback": bool,
    "role_reversal": bool,
    "fake_breakout": bool,
    "reason": str
  },
  "candlestick_signal": {
    "bullish_pinbar": bool,
    "bearish_pinbar": bool,
    "bullish_engulfing": bool,
    "bearish_engulfing": bool,
    "harami": bool,
    "strong_bull_body": bool,
    "strong_bear_body": bool,
    "rejection_wick": bool
  },
  "chart_pattern": {
    "double_top": bool, "double_bottom": bool,
    "triple_top": bool, "triple_bottom": bool,
    "head_and_shoulders": bool, "inverse_head_and_shoulders": bool,
    "flag": bool, "wedge": bool, "triangle": bool,
    "neckline_broken": bool, "retested": bool
  },
  "indicator_context": {
    "rsi_value": float | null,
    "rsi_range_valid": bool | null,
    "rsi_trend_danger": bool | null,
    "macd_momentum_up": bool | null,
    "macd_momentum_down": bool | null,
    "bb_squeeze": bool | null,
    "bb_expansion": bool | null,
    "bb_band_walk": bool | null,
    "ma_trend_support": "BUY|SELL|NEUTRAL" | null
  },
  "risk_plan_obs": {
    "atr_stop_distance_atr": float,
    "structure_stop_price": float | null,
    "structure_stop_distance_atr": float | null,
    "rr_atr_based": float | null,
    "rr_structure_based": float | null,
    "invalidation_clear": bool
  },
  "vote_breakdown": {
    "indicator_buy_votes": int,
    "indicator_sell_votes": int,
    "voted_action": "BUY|SELL|HOLD",
    "macro_alignment": "BUY|SELL|NEUTRAL|UNKNOWN",
    "htf_alignment":   "BUY|SELL|NEUTRAL|UNKNOWN",
    "structure_alignment": "BUY|SELL|NEUTRAL|UNKNOWN",
    "sr_alignment": "BUY|SELL|NEUTRAL|UNKNOWN",
    "candle_alignment": "BUY|SELL|NEUTRAL|UNKNOWN"
  },
  "final_confluence": {
    "label": "STRONG_BUY_SETUP | WEAK_BUY_SETUP | STRONG_SELL_SETUP | WEAK_SELL_SETUP | NO_TRADE | AVOID_TRADE | UNKNOWN",
    "score": float,
    "bullish_reasons": [str, ...],
    "bearish_reasons": [str, ...],
    "avoid_reasons": [str, ...]
  }
}
```

For warmup bars or ATR-unavailable bars, the slice is populated from
`empty_technical_confluence()` so the schema is shape-stable across the
full trace.

## What it detects

- **Dow / structure**: reuses `patterns.PatternResult` (swings,
  HH/HL/LH/LL, trend_state). Adds `bos_up` / `bos_down` (close beyond
  the most recent confirmed swing high / low).
- **Support / resistance**: confirmed swings serve as level proxies;
  distances are normalized by ATR; `near_*` is < 0.5 ATR; `breakout`,
  `fake_breakout`, `pullback`, `role_reversal` are computed via
  conservative close-based comparisons.
- **Candlestick**: pinbar (wick / body ratio + body / range ratio),
  engulfing (body comparison vs previous bar), harami (current body
  inside previous body), strong body (body / range), rejection wick
  (wick / range against close direction).
- **Indicator context**: RSI regime label (range vs trend danger),
  Bollinger lifecycle (squeeze / expansion / band-walk via rolling
  band-width percentile), MACD momentum sign / cross, MA trend
  support (SMA20 vs SMA50).
- **Risk plan**: structure stop = last swing low (BUY) / last swing
  high (SELL); ATR stop distance = stop_atr_mult; RR computed both ways;
  `invalidation_clear` when structure stop is at least 0.5 ATR away.
- **Vote breakdown**: replays the 4-source `technical_signal()` vote
  (RSI / MACD / SMA20-SMA50 / BB position) and surfaces it explicitly.
  Adds per-source alignment labels (structure, S/R, candle).
- **Final confluence label**: a simple scoring step over the section
  outputs producing one of `STRONG_BUY_SETUP / WEAK_BUY_SETUP /
  STRONG_SELL_SETUP / WEAK_SELL_SETUP / NO_TRADE / AVOID_TRADE / UNKNOWN`
  plus reason lists. The exact thresholds live in
  `technical_confluence._final_confluence_label` and are intentionally
  observation-grade — not tuned, not optimized, not used for entry.

## What it does NOT do

- Not connected to `decision_engine.decide_action`.
- Not connected to `risk_gate`.
- Not used by any `RuleCheck`.
- Does not change `BarDecisionTrace.decision.final_action`.
- Does not detect head_and_shoulders / flag / wedge / triangle.
- Does not detect horizontal levels via clustering — uses confirmed
  swings only.
- Does not score patterns by confidence beyond what `patterns.analyse`
  already provides.
- Thresholds (e.g. wick / body ratios) are observation constants that
  live in `technical_confluence.py` only. They do **not** appear in
  `parameter_defaults.PARAMETER_BASELINE_V1` (would change baseline
  hash) and they do **not** flow through `runtime_parameters.py`.

## Future decision-gate criteria

Connecting any subset of these fields to `decide_action` requires:

1. Multi-symbol × multi-period evidence that the chosen field separates
   WIN from LOSS at sufficient_n (≥ 30 per cell).
2. The connection lives behind an opt-in CLI flag (e.g.
   `--apply-confluence-filter`), with default-off mode pinned
   byte-identical to current behavior.
3. The resulting PR follows the
   [`royal_road_gap_audit_v1.md`](royal_road_gap_audit_v1.md) §7
   ordering: PR-C2 / C3 / C4 → evidence → PR-D.

Until then, read this slice **only** through summary aggregations and
R-candidate analyses. Do not introduce code paths in
`decision_engine.py`, `risk_gate.py`, or any rule_check that read it.

## Files touched

- `src/fx/technical_confluence.py` (new)
- `src/fx/decision_trace.py` (added `TechnicalConfluenceSlice`,
  optional field on `BarDecisionTrace`, `to_dict` extension)
- `src/fx/decision_trace_build.py` (`_confluence_slice` helper, wired
  into `build_full_trace` and `build_atr_unavailable_trace`)
- `tests/test_technical_confluence_candlestick.py` (new)
- `tests/test_technical_confluence_support_resistance.py` (new)
- `tests/test_technical_confluence_trace_observation_only.py` (new)
- `docs/technical_confluence_v1.md` (this file)
- `docs/technical_baseline_royal_road.md` (implementation-status note)

## Files NOT touched

- `src/fx/decision_engine.py` (decide / decide_action — unchanged)
- `src/fx/risk_gate.py` (unchanged)
- `src/fx/indicators.py` (`technical_signal` — unchanged)
- `src/fx/parameter_defaults.py` (`PARAMETER_BASELINE_V1` — unchanged)
- `src/fx/runtime_parameters.py` (no new connected section)
- `src/fx/cli.py` (no new flag introduced for the confluence slice)
- live / OANDA / paper paths (untouched)
- `runs/` and `libs/` (no commits)
