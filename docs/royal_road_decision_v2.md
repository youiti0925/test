# royal_road_decision_v2

Created: 2026-05-01.
Branch: `feat/royal-road-complete-v2`.
Predecessor: `royal_road_decision_v1` (`docs/royal_road_decision_v1.md`).
Threshold audit: `docs/royal_road_thresholds_v1.md`.

## Layer separation

- `technical_confluence_v1` — observation layer (read-only trace).
- `royal_road_decision_v1` — opt-in research decision profile (modes:
  strict / balanced / exploratory).
- `royal_road_decision_v2` — opt-in research decision profile that
  ALSO consults a richer set of structural / macro / lower-TF observations.
- All three modes are heuristic and unvalidated. Default profile remains `current_runtime`.

## What v2 adds over v1

| v1 | v2 |
|---|---|
| Confluence dict (technical_confluence_v1) | + S/R level clustering (`support_resistance.py`) |
| | + 2-anchor trendline fitting (`trendlines.py`) |
| | + Real chart patterns: H&S / inverse / flag / wedge / triangle (`chart_patterns.py`) |
| | + Lower-TF trigger 15m / 5m (`lower_timeframe_trigger.py`) |
| | + Symbol-aware macro alignment (`macro_alignment.py`) |
| Stop = ATR multiplier always | + `--stop-mode {atr, structure, hybrid}` (`stop_modes.py`) |
| Single trace slice `royal_road_decision` | + Separate slice `royal_road_decision_v2` with all v2 fields |
| `r_candidates_summary` (R12-R16) | + `royal_road_v2_summary` aggregator (action / label / mode / block_reasons / cautions / evidence_axes counts / SR strength / trendline / chart_pattern / lower_tf / macro / stop_mode / RR distributions) |

## Modes

v2 reuses the same modes as v1 (strict / balanced / exploratory) via
`src/fx/royal_road_decision_modes.py`. `--royal-road-mode` selects.
Default: `balanced`.

## Stop modes (`--stop-mode`)

| mode | behaviour | default? |
|---|---|---|
| `atr` | entry +/- (stop_atr_mult * ATR). Byte-identical to PR #21 main. | Yes |
| `structure` | entry +/- structure_stop_price (most recent confirmed swing low for BUY, swing high for SELL). HOLD if missing. | No |
| `hybrid` | structure if 0.5..3.0 ATR away, ATR fallback if too far, HOLD if too close (<0.5 ATR). | No |

## Trace audit (v2 slice)

`BarDecisionTrace.royal_road_decision_v2` is `None` for default /
v1-only runs and populated when `--decision-profile royal_road_decision_v2`
is active. Per-bar payload includes:

```
profile / mode / mode_status / mode_needs_validation
action / confidence / score
reasons / block_reasons / cautions
evidence_axes (bullish/bearish dicts) + counts + min_evidence_axes_required
support_resistance_v2  (level clustering with broken / false_breakout / role_reversal counts)
trendline_context      (ascending_support / descending_resistance + bullish_signal / bearish_signal)
chart_pattern_v2       (H&S / inverse H&S / flag / wedge / triangle, each with formation/impulse/consolidation/upper_line/lower_line/convergence_score/breakout_direction/pattern_quality_score/reason/retested)
lower_tf_trigger       (interval / available / breakout / retest / micro_double_top|bottom / pinbar / engulfing)
macro_alignment        (symbol-aware DXY/yield/VIX/risk_on_off/currency_bias/macro_score, plus event_tone=UNKNOWN with unavailable_reason)
structure_stop_plan    (chosen_mode / outcome / stop_price / take_profit_price / rr_realized / structure_stop_distance_atr / atr_stop_price)
compared_to_current_runtime  (closed taxonomy)
compared_to_royal_road_v1    (closed taxonomy)
```

## Decision rules (algorithm)

```
risk gate must pass (event_high, daily_loss_cap, etc.)
   ↓
universal:
  invalidation_clear missing      → BLOCK if mode requires; else caution
  structure_stop_price missing    → BLOCK if mode requires; else caution
  risk_reward < min_rr            → BLOCK
  serious avoid_reasons (mode-configurable) → BLOCK
   ↓
direction (from final_confluence.label):
  STRONG_BUY_SETUP / STRONG_SELL_SETUP → side, axes_required = strong
  WEAK_BUY_SETUP / WEAK_SELL_SETUP →
    if mode allows weak: side, axes_required = weak
    else: BLOCK weak_setup
  NO_TRADE / AVOID_TRADE / UNKNOWN → BLOCK (label)
   ↓
v2-specific block reasons:
  macro_strong_against == side                 → BLOCK
  macro_score <= -_MACRO_BLOCK_THRESHOLD vs BUY → BLOCK
  macro_score >= +_MACRO_BLOCK_THRESHOLD vs SELL → BLOCK
  near_strong_resistance + BUY                 → BLOCK
  near_strong_support + SELL                   → BLOCK
  htf_counter_trend (mode-controlled)          → BLOCK
  sr_fake_breakout                             → BLOCK
  axes_count < axes_required                   → BLOCK
  stop_plan_invalid (when stop_mode != "atr")  → BLOCK
   ↓
if no block: enter with confidence = 0.55 + 0.30 * |score| (cap 0.95)
otherwise:   HOLD with reason = block_reasons joined
```

## Evidence axes (v2 set)

Bullish (any True counts as one axis):
- `bullish_structure`
- `near_strong_support`
- `ascending_trendline_intact`
- `descending_trendline_broken_up`
- `chart_pattern_bullish_breakout`
- `lower_tf_bullish_trigger`
- `macro_buy_score` (macro_score >= 0.5)
- `sr_role_reversal_bullish`
- `sr_pullback_bullish`

Bearish: mirror.

## Refactors / consolidations in v2.1

- `retest_detection.py`: shared retest helper. Both `chart_patterns`
  (neckline retest) and `lower_timeframe_trigger` (level retest) now
  consult the same `detect_retest()` function. Same closed-taxonomy
  RetestResult dataclass. wick-based detection added on top of
  close-based via `wick_allowed=True`.
- chart_patterns: flag detector is no longer N=3 fixed. It walks
  candidate consolidation lengths (5..30 bars) and impulse lengths
  (3..20 bars) to find an adaptive impulse + consolidation pair
  satisfying `consol_range / impulse_range <= 0.6` and
  `impulse_range >= 1.5 ATR`. Each pattern now also reports
  `formation_start_index / formation_end_index / formation_bars /
  impulse_bars / consolidation_bars / upper_line / lower_line /
  convergence_score / breakout_direction / pattern_quality_score /
  reason`.
- support_resistance: `broken_count`, `false_breakout_count`, and
  `role_reversal_count` are now real per-level counts derived from
  the visible close history. `strength_score` reflects them
  (false_breakouts and broken-but-not-reversed reduce strength;
  genuine role reversals raise it).

## What v2 still does NOT cover

- **Dynamic trailing stop is NOT implemented.** v2's stop placement
  is fixed at entry (ATR / structure / hybrid). A v3 expansion may
  add trailing — out of scope here.
- **Central-bank event tone (hawkish / dovish) is NOT extracted.**
  `macro_alignment.event_tone` always reports `"UNKNOWN"` with
  `event_tone_unavailable_reason="tone_extraction_not_implemented"`.
  Implementing FOMC / ECB statement classification is future work.
- Trendline fitting is 2-anchor only. 3+ anchor confirmation, log-scale
  fits, dynamic re-anchoring are deferred.
- Chart patterns: H&S detector requires 3 highs / 3 lows in a fixed
  window; multi-touch broadening of shoulder symmetry is deferred.
- Triangle classification differentiates ascending / descending /
  symmetric only — pennant / rectangle are out of scope.
- Lower-TF trigger uses a single ATR proxy (stdev-based) for the
  shared retest helper; ATR(14) on the lower-TF series is not yet
  computed. Acceptable observation-grade signal.
- Position sizing remains full-cash (Kelly is implemented in `risk.py`
  but never called).
- LLM advisory and waveform bias are not consulted in any v2 mode by
  design (clean attribution against royal-road alone).

## Adoption guardrails (unchanged from v1)

- Default decision profile remains `current_runtime`.
- `decision_engine.py` / `risk_gate.py` / `indicators.py` are unchanged.
- Live / OANDA / paper paths are not connected.
- `parameter_defaults.PARAMETER_BASELINE_V1` hash is unchanged.
- v2 thresholds are heuristic constants in their respective modules
  (`support_resistance._CLUSTER_BUCKET_ATR`, `trendlines._NEAR_LINE_ATR`,
  `chart_patterns._FLAG_*`, `retest_detection._DEFAULT_TOLERANCE_ATR`,
  `macro_alignment._VIX_HIGH_LEVEL`, etc.). They do not flow through
  `runtime_parameters.py` and changing them does NOT change the
  baseline catalog hash.
- All modes flagged `mode_needs_validation: True`. 90d single-window
  results are NOT a basis for adoption.

## Files (cumulative through v2.1)

- `src/fx/support_resistance.py` (level clustering + per-level scan counts)
- `src/fx/trendlines.py` (2-anchor fit)
- `src/fx/chart_patterns.py` (H&S / inverse / flag adaptive / wedge / triangle, retest via shared helper)
- `src/fx/retest_detection.py` (NEW — shared close + wick retest)
- `src/fx/lower_timeframe_trigger.py` (15m/5m candle + breakout + shared retest)
- `src/fx/macro_alignment.py` (symbol-aware DXY/VIX/yield + event_tone=UNKNOWN)
- `src/fx/stop_modes.py` (atr / structure / hybrid)
- `src/fx/royal_road_decision_v2.py` (decision aggregator)
- `src/fx/royal_road_v2_summary.py` (NEW — summary aggregator)
- `src/fx/decision_trace.py` (RoyalRoadDecisionV2Slice)
- `src/fx/decision_trace_build.py` (slice wiring)
- `src/fx/decision_trace_io.py` (royal_road_v2_summary fail-soft block)
- `src/fx/backtest_engine.py` (`royal_road_mode`, `df_lower_tf`, `lower_tf_interval`, `stop_mode` kwargs)
- `src/fx/cli.py` (`--decision-profile {current_runtime, royal_road_decision_v1, royal_road_decision_v2}`, `--royal-road-mode`, `--lower-tf`, `--stop-mode`)

## Files NOT touched

- `src/fx/decision_engine.py`
- `src/fx/risk_gate.py`
- `src/fx/indicators.py`
- `src/fx/parameter_defaults.py` (baseline hash invariant)
- `src/fx/runtime_parameters.py` (no new connected section)
- `src/fx/oanda.py` / `src/fx/broker.py` (live paths)
