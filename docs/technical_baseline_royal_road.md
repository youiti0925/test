# FX Technical Baseline: Royal Road Method

Created: 2026-05-01
Source materials:

- `docs/reference/FX_Technical_Analysis_Masterclass.pptx`
- `docs/reference/FX_Trading_Masterclass_Roadmap.pptx`

> Note: The PPTX files are the original reference materials. This Markdown file is the operational baseline used for implementation planning, gap audits, trace design, and future PR reviews.

---

## 1. Purpose

This document defines the technical baseline for the FX verification project.

The goal is not to create PRs, run backtests endlessly, or optimize a single parameter in isolation.

The goal is to build a trading research system that can identify:

- conditions with higher probability of winning,
- conditions with higher probability of losing,
- setups where large losses should be avoided,
- environments where a technical setup is valid,
- environments where a technical setup is dangerous,
- whether a stop / take-profit / invalidation plan is structurally justified.

This baseline should be used as the reference when evaluating whether the current system is aligned with common FX trading fundamentals.

---

## 2. Royal Road Procedure

The materials define a practical FX decision flow. The order matters.

### Step 1: Fundamental and macro context

Before technical entry, check the broad environment.

Examples:

- interest rates,
- central bank policy,
- FOMC / BOJ / ECB / BOE,
- CPI / PCE,
- NFP / employment data,
- GDP,
- important speeches,
- currency strength / weakness,
- DXY,
- VIX,
- risk-on / risk-off environment,
- high-impact scheduled events.

Purpose:

- identify large directional pressure,
- avoid trading immediately before or during major event volatility,
- avoid technical entries that directly conflict with a strong macro catalyst.

### Step 2: Higher-timeframe environment recognition

Start from higher timeframes before considering entry.

Examples:

- weekly / daily / 4H / 1H trend,
- major support / resistance on higher timeframe,
- whether the intended trade follows or fights the higher-timeframe direction,
- whether the current price is in a higher-timeframe reaction zone.

Principle:

- move from macro to micro,
- avoid treating a lower-timeframe signal as valid when higher-timeframe structure is against it.

### Step 3: Dow theory / market structure

Classify the market structure.

Core concepts:

- HH: higher high,
- HL: higher low,
- LH: lower high,
- LL: lower low,
- uptrend,
- downtrend,
- range,
- transition,
- break of structure,
- last swing low,
- last swing high,
- failure swing,
- non-failure swing,
- structural invalidation.

Purpose:

- determine whether the market is trending, ranging, or transitioning,
- avoid using range tools in strong trends without confirmation,
- identify whether a proposed trade is continuation or reversal.

### Step 4: Line analysis / support and resistance

Identify levels where price is likely to react.

Core concepts:

- support,
- resistance,
- horizontal levels,
- trendlines,
- neckline,
- breakout,
- pullback,
- role reversal,
- fake breakout,
- candle reaction around levels.

Purpose:

- avoid entries in the middle of nowhere,
- identify clear invalidation levels,
- distinguish breakout entries from pullback entries,
- avoid chasing after a move has already extended.

### Step 5: Chart pattern setup

Look for structural setups.

Examples:

- double top,
- double bottom,
- head and shoulders,
- inverse head and shoulders,
- ascending flag,
- descending flag,
- ascending triangle,
- descending triangle,
- symmetrical triangle,
- wedge,
- neckline breakout,
- retest after breakout,
- second-wave entry.

Important:

- a pattern candidate is not enough,
- confirmation matters,
- neckline break should be based on close, not only wick,
- stop and target should be derived from the structure when possible.

### Step 6: Lower-timeframe entry confirmation

Use lower-timeframe price action to confirm entry.

Examples:

- bullish pin bar,
- bearish pin bar,
- bullish engulfing,
- bearish engulfing,
- harami,
- strong bullish body,
- strong bearish body,
- rejection wick,
- reversal candle,
- lower-timeframe double bottom / top in a higher-timeframe zone.

Purpose:

- avoid entering only because a higher-timeframe zone exists,
- wait for actual reaction,
- improve entry location and stop placement.

### Step 7: Indicators as secondary confirmation

Indicators are support tools, not the main decision authority.

Examples:

- SMA,
- EMA,
- Granville-style moving average behavior,
- Bollinger Band squeeze,
- Bollinger Band expansion,
- Bollinger Band walk,
- RSI,
- MACD,
- RSI divergence,
- MACD divergence,
- MACD cross,
- MACD histogram change.

Important warnings:

- RSI 70/30 should not be used identically in every regime,
- RSI mean reversion is more suitable in ranges,
- RSI countertrend entries can be dangerous in strong trends,
- Bollinger should not be reduced to band position only,
- MACD should not be reduced to positive/negative only,
- indicators should confirm structure, not replace it.

### Step 8: Confluence check

A trade setup should have multiple reasons pointing in the same direction.

Potential confluence sources:

- macro direction,
- higher-timeframe trend,
- Dow structure,
- support / resistance,
- chart pattern,
- candlestick confirmation,
- indicator confirmation,
- risk-reward,
- clear invalidation line.

Principle:

- if signals conflict, prefer HOLD,
- if the setup is unclear, prefer HOLD,
- missing a trade is better than entering a low-quality trade.

### Step 9: Risk plan before entry

Before entry, define loss and profit conditions.

Core concepts:

- OCO-style plan,
- stop loss,
- take profit,
- risk-reward 1:2 or better where possible,
- structural invalidation,
- recent high / low stop,
- neckline stop,
- support / resistance break stop,
- ATR stop,
- do not move stop emotionally,
- small losses, larger wins.

Important:

- ATR stop is useful, but not the same as structure-based invalidation,
- the system should eventually compare ATR stop and structure stop,
- entry quality and stop quality must be evaluated together.

### Step 10: Post-trade verification

For every trade or missed trade, evaluate:

- why the trade won,
- why the trade lost,
- whether the entry location was poor,
- whether the stop location was poor,
- whether the system fought higher-timeframe trend,
- whether there was no clear level,
- whether macro/event risk invalidated the setup,
- whether the system entered before confirmation,
- whether the setup lacked confluence.

---

## 3. Gap Audit Against Current System

Current system strengths:

- risk gate exists,
- backtest engine exists,
- decision trace exists,
- future outcome is recorded after the decision loop,
- macro / DXY / VIX can be observed,
- waveform bias exists as advisory / observation,
- ATR stop / TP exists,
- parameter A/B infrastructure exists,
- event calendar determinism exists,
- R-candidate summaries exist.

Current system weakness:

The current BUY / SELL direction is still too dependent on indicator voting:

- RSI,
- MACD,
- SMA20 vs SMA50,
- Bollinger position.

This means the system is not yet a full royal-road FX decision system.

The following concepts are missing or weak:

- horizontal support / resistance,
- role reversal,
- pullback after breakout,
- fake breakout,
- pin bar,
- engulfing candle,
- harami,
- strong body / rejection wick,
- flag,
- wedge,
- triangle,
- structure-based stop,
- invalidation line,
- multi-level confluence score,
- RSI regime filter,
- Bollinger lifecycle,
- lower-timeframe trigger.

---

## 4. Implementation Direction

Do not connect these concepts directly to BUY / SELL / HOLD first.

Correct path:

1. Add royal-road features to trace as observation-only.
2. Compare winning and losing trades.
3. Identify clearly bad conditions that should become HOLD candidates.
4. Identify clearly good conditions that may strengthen entries.
5. Run A/B tests only after sufficient sample size exists.
6. Connect to decision logic only if multi-symbol and multi-period evidence supports it.

---

## 5. Proposed `technical_confluence_v1`

Initial implementation should be observation-only.

Suggested trace fields:

```json
{
  "technical_confluence_v1": {
    "market_regime": "TREND_UP | TREND_DOWN | RANGE | VOLATILE | TRANSITION | UNKNOWN",
    "dow_structure": "HH_HL | LH_LL | MIXED | BREAK_OF_STRUCTURE_UP | BREAK_OF_STRUCTURE_DOWN | UNKNOWN",
    "mtf_context": {
      "upper_trend": "UP | DOWN | RANGE | UNKNOWN",
      "middle_structure": "UP | DOWN | RANGE | MIXED | UNKNOWN",
      "lower_trigger": "BULLISH | BEARISH | NONE | UNKNOWN",
      "alignment": "ALIGNED | MIXED | COUNTER | UNKNOWN"
    },
    "support_resistance": {
      "near_support": false,
      "near_resistance": false,
      "breakout": false,
      "pullback": false,
      "role_reversal": false,
      "fake_breakout": false,
      "distance_to_level_atr": null
    },
    "candlestick_signal": {
      "bullish_pinbar": false,
      "bearish_pinbar": false,
      "bullish_engulfing": false,
      "bearish_engulfing": false,
      "harami": false,
      "strong_bull_body": false,
      "strong_bear_body": false,
      "rejection_wick": false
    },
    "chart_pattern": {
      "double_top": false,
      "double_bottom": false,
      "head_and_shoulders": false,
      "inverse_head_and_shoulders": false,
      "flag": false,
      "wedge": false,
      "triangle": false
    },
    "indicator_context": {
      "rsi_range_valid": null,
      "rsi_trend_danger": null,
      "bb_squeeze": null,
      "bb_expansion": null,
      "bb_band_walk": null,
      "macd_momentum_up": null,
      "macd_momentum_down": null,
      "ma_trend_support": null
    },
    "risk_plan": {
      "rr_ok": null,
      "atr_stop": null,
      "structure_stop": null,
      "invalidation_clear": null,
      "invalidation_price": null
    },
    "final_confluence": {
      "label": "STRONG_BUY_SETUP | WEAK_BUY_SETUP | STRONG_SELL_SETUP | WEAK_SELL_SETUP | NO_TRADE | AVOID_TRADE | UNKNOWN",
      "score": null,
      "bullish_reasons": [],
      "bearish_reasons": [],
      "avoid_reasons": []
    }
  }
}
```

---

## 6. PR Planning Guidance

Recommended staged PRs:

### PR-A: Documentation and audit only

- Add this baseline document.
- Add PPTX references.
- Do not change trading logic.

### PR-B: Observation-only candlestick and support/resistance features

- Add candlestick detectors.
- Add simple level detection.
- Add trace fields.
- Do not connect to `decide_action`.

### PR-C: Confluence summary and R-candidate expansion

- Add `technical_confluence_v1` summary.
- Add win/loss analysis cells.
- Add sufficient_n flags.

### PR-D: Decision impact experiment

Only after enough evidence:

- test HOLD filters for clearly bad setups,
- test entry reinforcement for clearly good setups,
- compare against current runtime.

---

## 7. Non-Negotiable Guardrails

Do not do these without explicit approval:

- create or merge PRs automatically,
- change `decision_engine.py`,
- change `decide` / `decide_action`,
- connect waveform directly to trading decisions,
- connect DXY / VIX / SMA / event windows directly to risk gate or decision logic,
- create USDJPY-only rules,
- touch live / OANDA / paper trading,
- commit `runs/`,
- commit `libs/`,
- call a low-n result conclusive,
- treat current_runtime results as literature_baseline results.

---

## 8. Current Priority

Before implementing royal-road decision changes:

1. Check why PR #22 is open.
2. Resolve USDJPY `n_traces` mismatch in 90d A/B.
3. Finalize 90d A/B interpretation.
4. Use this document for a complete gap audit.
5. Design `technical_confluence_v1` observation-only.
6. Only after evidence, consider decision changes.

---

## 9. Final Recommendation

The existing system should not be discarded.

The current foundation is useful for research, traceability, safety, and A/B testing.

However, the decision core should move from indicator-voting-centered logic toward royal-road confluence logic.

First step: observe royal-road concepts in trace.

Second step: verify which concepts separate winners from losers.

Third step: only then connect proven concepts to decision logic.

---

## 10. Implementation Status

- `technical_confluence_v1` observation-only implementation added.
  See [`docs/technical_confluence_v1.md`](technical_confluence_v1.md).
- Not connected to `decision_engine.decide_action`.
- Not connected to `risk_gate`.
- Intended for win/loss analysis and future A/B design only.
- Decision pipeline output is byte-identical to PR #21 (pinned by
  `tests/test_technical_confluence_trace_observation_only.py`).
