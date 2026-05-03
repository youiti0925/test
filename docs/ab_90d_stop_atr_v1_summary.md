# 90d A/B Stop ATR Summary

Status: provisional. Created: 2026-05-01.
Branch: `docs/add-fx-masterclass-reference`. Code reference: `main` = `a2a4417` (PR #21 merged).

## Important framing

This A/B test is **not** a full literature-parameter comparison.

At PR #21, the practical runtime difference between current_runtime and `literature_baseline_v1` is effectively limited to a single field — `stop_atr_mult`. All other PARAMETER_BASELINE_V1 sections that are connected at runtime (rsi / macd / bollinger / atr period / risk.max_hold_bars) coincide with current_runtime defaults; the remaining sections (events / waveform / sma / vix / dxy / signal_voting / long_term) are deliberately skipped (`runtime_parameters.SKIPPED_SECTIONS_PR21`).

Therefore the empirical comparison is:

```
A:
  - current_runtime
  - stop_atr_mult = 2.0

B:
  - literature_baseline_v1 applied runtime
  - stop_atr_mult = 1.5
```

The interpretation must be:

- 90d A/B = **2.0 ATR stop vs 1.5 ATR stop**
- not "full literature baseline is better"

Calling B "literature baseline" without this qualifier conflates a single-parameter delta with the broader literature catalog and is explicitly forbidden by `docs/technical_baseline_royal_road.md` §7 ("treat current_runtime results as literature_baseline results").

## Current observation (provisional)

- B improved pooled total return at 90d level.
- avg_loss improved across symbols on B (consistent with a tighter stop closing losing trades earlier).
- EURUSD=X worsened on B (losing trades that would have recovered under 2.0 ATR were stopped out at 1.5 ATR).
- USDJPY=X has unresolved n_traces mismatch (see `docs/usdjpy_90d_ab_n_traces_mismatch.md`). Until that is explained, USDJPY=X-level and POOL-level (POOL contains USDJPY=X) interpretations remain provisional.
- All stop trades on B show stop distance ≈ 0.75x of A's stop distance (1.5 / 2.0 = 0.75) — consistent with stop_atr_mult being the sole runtime change.
- PF improved 1.31 → 1.56 at POOL level. This number includes USDJPY=X with the mismatch and should be re-checked once mismatch is resolved.
- Conclusion: **provisional**. Sample is 90d only; royal-road §7 prohibits treating low-n results as conclusive.

## Royal-road positioning

`stop_atr_mult` lives entirely within Step 9 of the royal-road procedure (risk plan / invalidation). It is one knob inside the risk-plan layer.

- Step 9 also calls for **structure-based stops** (recent swing / neckline / S/R break) which are not implemented today.
- Steps 4 (S/R) and Step 6 (candlestick) — which gate **entry** quality — are absent.

Optimizing only Step 9's ATR multiplier without addressing Step 4 / Step 6 / Step 8 (confluence) reduces an exit-only knob inside an entry-quality-blind decision pipeline. The +0.61pp pool-level improvement is consistent with "letting losers cut sooner" rather than "winning more often" — and therefore is bounded.

See `docs/royal_road_gap_audit_v1.md` for the full gap audit.

## Do not (until explicitly approved)

- Adopt 1.5 ATR as the new runtime default.
- Change decision logic.
- Change risk gate.
- Proceed to 180d / 365d before USDJPY n_traces mismatch is documented and (separately) resolved or accepted.
- Remove this provisional flag from any downstream report.
- Treat this as evidence that "literature baseline beats current runtime".

## What is allowed

- Reference this document when discussing 90d numbers.
- Re-read the per-symbol breakdown alongside `usdjpy_90d_ab_n_traces_mismatch.md` to scope which numbers are firm and which are provisional.
- Use this finding to motivate (but not to validate) further work on Step 9 — only after Steps 4 / 6 / 8 are observation-only on trace.

## Next-step gate

To upgrade this from provisional to confirmed:

1. Resolve USDJPY n_traces mismatch (separate doc).
2. Add multi-period evidence (180d / 365d) — but only after `technical_confluence_v1` observation fields are in trace, so we can ask "did 1.5 ATR help in the same setups, or only in some setup labels?".
3. Run the comparison on at least one out-of-sample period not used in the 90d window.

Without all three, this remains a 90d single-period single-pool reading.
