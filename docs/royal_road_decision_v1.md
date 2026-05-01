# royal_road_decision_v1

Created: 2026-05-01.
Branch: `feat/royal-road-decision-v1`.
Baseline: [`docs/technical_baseline_royal_road.md`](technical_baseline_royal_road.md).
Predecessor PRs: PR-C1 (technical_confluence_v1 trace), PR-C2 (summary + R12-R16).

## Purpose

`royal_road_decision_v1` is an **opt-in decision profile** that replaces
the indicator-vote-centric `current_runtime` BUY/SELL/HOLD logic with
royal-road rules driven by `technical_confluence_v1`.

Its goal is not to ship a final trading rule. Its goal is:

1. Use the royal-road procedure as the **basis** for entry decisions
   (not just for observation).
2. Keep `current_runtime` as the comparison baseline so we can ask
   *did the royal road outperform the legacy votes? in which conditions?
   what conditions blocked royal-road from a winning trade?*
3. Surface bar-by-bar comparison data (`compared_to_current_runtime`)
   so post-hoc analyses can identify trades where the two profiles
   diverged — a discovery surface for finding setups where royal road
   outperforms or underperforms the legacy approach.

## Why this exists

The royal-road audit (`docs/royal_road_gap_audit_v1.md`) found that
`current_runtime` decides direction with a 4-way indicator vote (RSI /
MACD / SMA20-50 / BB position). Steps 4 (S/R), 6 (candlestick), and 8
(confluence) — the entry-quality core of the royal-road procedure —
are absent from the decision path.

PR-C1 added a `technical_confluence_v1` slice to the trace as
**observation-only**. PR-C2 added summary aggregation and R-candidate
cells. This PR uses that observed payload as the **decision input**
for an opt-in profile, keeping the legacy default intact.

## Difference from current_runtime

| Aspect | current_runtime | royal_road_decision_v1 |
|---|---|---|
| Direction source | `indicators.technical_signal` 4-way vote | `technical_confluence.final_confluence.label` |
| Required label | (not consulted) | STRONG_BUY_SETUP / STRONG_SELL_SETUP |
| WEAK_BUY/SELL setups | (not consulted) | HOLD |
| invalidation_clear required | No | Yes |
| structure_stop required | No | Yes |
| near_resistance + BUY | Allowed (only htf veto) | Blocked |
| near_support + SELL | Allowed (only htf veto) | Blocked |
| serious avoid_reasons (fake_breakout, regime_volatile) | Ignored | Blocks the entry |
| Higher-TF counter-trend | Blocked (existing) | Blocked (matched) |
| Risk gate (event_high, etc.) | Always run | Always run |
| LLM advisory veto | Active | NOT consulted (v1 isolates royal-road from LLM) |
| Waveform bias veto | Active | NOT consulted (v1 isolates royal-road from waveform) |
| Stop / TP placement | ATR multiplier | ATR multiplier (UNCHANGED — v1 does not modify the actual stop) |

## Royal-road decision rules (v1, intentionally strict)

Implemented in `src/fx/royal_road_decision.py::decide_royal_road`.

```
risk gate must pass (event_high, daily_loss_cap, etc.)
   ↓
require: invalidation_clear == True
require: structure_stop_price is not None
require: risk_reward >= 1.5
require: no serious avoid_reasons (currently: fake_breakout, regime_volatile)
   ↓
if final_confluence.label == "STRONG_BUY_SETUP":
    require: not near_resistance
    require: bullish structure OR near_support OR bullish candlestick
    require: higher-TF != DOWNTREND
    → BUY  (confidence = 0.55 + 0.30 * |score|, capped at 0.95)

elif final_confluence.label == "STRONG_SELL_SETUP":
    require: not near_support
    require: bearish structure OR near_resistance OR bearish candlestick
    require: higher-TF != UPTREND
    → SELL (confidence as above)

elif final_confluence.label in ("WEAK_BUY_SETUP", "WEAK_SELL_SETUP"):
    → HOLD

elif final_confluence.label in ("NO_TRADE", "AVOID_TRADE", "UNKNOWN"):
    → HOLD

else:
    → HOLD (defensive)
```

Bullish structure = `market_regime == TREND_UP` OR `structure_code in (HH, HL)`
OR `bos_up` OR `double_bottom` OR `triple_bottom` OR `inverse_head_and_shoulders`.
Bearish structure is the mirror.

Bullish candle = `bullish_pinbar` OR `bullish_engulfing` OR `strong_bull_body`.
Bearish candle is the mirror.

V1 deliberately does not consult LLM advisory or waveform bias — keeping the royal-road profile self-contained makes it easier to attribute outcomes to the rule chain.

## How to run A/B

Default (`current_runtime`, byte-identical to PR #21 main):

```
python -m src.fx.cli backtest-engine \
  --symbol EURUSD=X --interval 1h --period 90d \
  --trace-out runs/cur
```

Royal-road profile:

```
python -m src.fx.cli backtest-engine \
  --symbol EURUSD=X --interval 1h --period 90d \
  --decision-profile royal_road_decision_v1 \
  --trace-out runs/royal
```

Then compare:
- `runs/cur/summary.json` vs `runs/royal/summary.json` for n_trades / WR / total_return / PF
- `runs/royal/decision_traces.jsonl` for `royal_road_decision.compared_to_current_runtime` per bar

Per-bar `difference_type` taxonomy:
- `same` — both profiles agree
- `current_buy_royal_hold` — legacy entered BUY, royal blocked
- `current_sell_royal_hold` — legacy entered SELL, royal blocked
- `current_hold_royal_buy` — legacy held, royal entered BUY
- `current_hold_royal_sell` — legacy held, royal entered SELL
- `opposite_direction` — legacy and royal point opposite ways
- `other` — anything else (unexpected; investigate)

For the discovery use case, the most useful pivots are:
- "trades where current_runtime entered and royal-road blocked" → which conditions did royal-road catch?
- "trades where royal-road entered and current_runtime did not" → does the royal-road label have lift?
- "opposite_direction" cells → are these typically right for royal or for current?

## What it still does not cover

- `support_resistance` is still **swing proxy** (not horizontal-level clustering)
- `fake_breakout` / `pullback` / `role_reversal` are still simple close-based observations
- `head_and_shoulders` / `flag` / `wedge` / `triangle` are schema placeholders only — never trigger STRONG_*
- Lower-timeframe trigger (15m / 5m) is not implemented
- Stop / TP order placement is unchanged — still ATR multiplier
- Position sizing is unchanged — full-cash convention as in current_runtime
- LLM and waveform veto are not consulted by royal-road v1 — this is **intentional** for clean attribution
- The score-to-confidence map is heuristic and has not been calibrated against historical outcomes

## How it can discover better-than-royal-road conditions

Each trace carries:
- the `technical_confluence_v1` payload (`market_regime`, `dow_structure`, `support_resistance`, `candlestick_signal`, etc.)
- the `royal_road_decision` (action / score / reasons / block_reasons)
- the `compared_to_current_runtime` block

Combined with PR #20 `r_candidates_summary` (R1–R11) and PR-C2 R12–R16, this enables:

- "*royal_road blocked but the trade would have won*"
  → look at `current_buy_royal_hold` cells filtered by `shadow_outcome.blocked_outcome_direction == UP`
  → tells us which royal-road blocks are **costing** us

- "*royal_road took a setup that current_runtime missed*"
  → look at `current_hold_royal_buy` and `current_hold_royal_sell` filtered by trade WIN / LOSS
  → tells us which setups royal-road **adds** that current_runtime misses

- "*both agreed and won / both agreed and lost*"
  → look at `same` cells × outcome
  → reveals conditions where both methods agree but lose, suggesting a third missing dimension (macro / session / regime)

- "*better-than-royal-road combinations*"
  → cross R12 (label) × R13 (S/R) × R14 (candlestick) × R15 (RSI regime) × R16 (structure stop) × outcome
  → if a sub-cell of WEAK_BUY_SETUP + bullish_engulfing + RSI range_valid wins consistently at sufficient_n,
    that's a candidate rule that beats royal-road v1 on entry frequency

The system is therefore not "royal-road is correct"; it is "royal-road
is the **comparison baseline** for searching adjacent rule sets that
might beat it on the user's actual objectives (PF, drawdown, large-loss
avoidance)".

## Safety guardrails

- Default profile remains `current_runtime`; engine output is byte-identical to PR #21 main when the flag is omitted.
- `decision_engine.py` / `risk_gate.py` / `indicators.py` are unchanged.
- live / OANDA / paper paths do not call `run_engine_backtest`. Pinned by source-grep tests in `tests/test_royal_road_decision_engine_invariant.py`.
- `parameter_defaults.PARAMETER_BASELINE_V1` is unchanged. Royal-road v1 thresholds live as module constants in `royal_road_decision.py` and do **not** flow through `runtime_parameters.py` or the literature catalog hash.
- USDJPY-specific rules: NONE introduced.
- runs/ and libs/: NOT committed.
- The royal-road profile is opt-in via `--decision-profile royal_road_decision_v1`; rejected with `ValueError` for any other profile name except `current_runtime`.
- byte-identical pin: `tests/test_royal_road_decision_engine_invariant.py::test_default_profile_byte_identical_to_pr21_pin`.

## Files touched

- `src/fx/royal_road_decision.py` (new)
- `src/fx/decision_trace.py` (new `RoyalRoadDecisionSlice` + optional field on `BarDecisionTrace`)
- `src/fx/decision_trace_build.py` (`build_full_trace` accepts `royal_road_decision` / `royal_road_compare`)
- `src/fx/backtest_engine.py` (new `decision_profile` kwarg + per-bar dispatch)
- `src/fx/cli.py` (new `--decision-profile` flag on `backtest-engine` subcommand)
- `tests/test_royal_road_decision_rules.py` (rule chain unit tests)
- `tests/test_royal_road_decision_engine_invariant.py` (byte-identical default + opt-in fires + safety pins)
- `tests/test_royal_road_decision_compare_trace.py` (comparison trace metadata)
- `docs/royal_road_decision_v1.md` (this file)
- `docs/technical_baseline_royal_road.md` (status note)

## Files NOT touched

- `src/fx/decision_engine.py` (legacy `decide_action` chain — unchanged)
- `src/fx/risk_gate.py` (unchanged; reused by royal-road for Step 1)
- `src/fx/indicators.py` (`technical_signal` — unchanged)
- `src/fx/parameter_defaults.py` (baseline hash unchanged)
- `src/fx/runtime_parameters.py` (no new connected section)
- `src/fx/oanda.py` / `src/fx/broker.py` (live paths — untouched)
