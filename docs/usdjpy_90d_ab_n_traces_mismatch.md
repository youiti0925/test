# USDJPY 90d A/B n_traces mismatch

Status: investigation memo only. Not a root-cause conclusion.
Created: 2026-05-01
Branch: `docs/add-fx-masterclass-reference`

## Context

- main = `a2a4417` (PR #21 merged)
- A = current_runtime, stop_atr_mult = 2.0
- B = literature_baseline_v1 applied_runtime, stop_atr_mult = 1.5
- symbol = USDJPY=X
- interval = 1h
- test period = 2026-01-30 to 2026-04-29
- context_days = 365
- events enabled
- waveform library: same id and path between A and B (expected)

## Observed

- A n_traces = 1415
- B n_traces = 1409
- difference = 6 traces
- Other 3 symbols (EURUSD=X, GBPUSD=X, AUDUSD=X) had matching n_traces between A and B in the same 90d run.

## Why this matters

- Under identical data and test window, n_traces should normally match between A and B.
- A `stop_atr_mult` change does not generate or remove bars on its own — it only affects exit timing of opened positions, not whether a bar produces a trace.
- Until explained, the USDJPY=X 90d A/B comparison and any POOL-level interpretation that includes USDJPY=X must remain **provisional**.
- Other 3 symbols with matching counts can still be compared, but the POOL aggregation should be flagged.

## Things to compare (when investigation resumes)

1. Exact CLI command lines used for A and B (argv-by-argv diff).
2. `run_metadata.bar_range` (first / last bar timestamp) for A vs B.
3. `run_metadata.context.n_test_bars` for A vs B.
4. First and last decision_trace timestamp for A vs B.
5. Set diff of decision_trace timestamps to identify the 6 missing rows.
6. `data_snapshot_hash` in run_metadata for A vs B.
   - Working hypothesis: yfinance 1h fetches are non-deterministic across processes.
   - In the prior 90d run, all 4 symbols other than USDJPY=X had matching `data_snapshot_hash`.
   - USDJPY=X showed `819f576f` for one run and `78440c0b` for another.
7. yfinance fetch row count for the test window (raw vs after preprocessing).
8. `events_file_sha8` in run_metadata for A vs B.
9. `event_window_policy_version` for A vs B.
10. waveform library id and on-disk path for A vs B.
11. `test_start` / `test_end` timezone handling (UTC vs local) and exclusive-end vs inclusive-end semantics.
12. `run_metadata.parameters` (literature_baseline_v1 application audit) for A vs B.
13. Whether A and B fetched data in the same process or in separate yfinance calls (process boundary affects yfinance cache state).

## Working hypotheses (in priority order)

1. **yfinance 1h non-determinism (most likely).** Different rows returned across separate fetches near the test-period boundary; affects bar count even with identical CLI inputs. Other 3 symbols may have been more cache-stable during the same window.
2. **Indicator warmup boundary.** ATR / RSI / SMA / BB warmup gates can cause early bars to be skipped at trace level. If a single missing high-impact bar shifts warmup completion by N bars, n_traces drops.
3. **Event coverage gap.** Even with identical `events.json`, calendar materialization can differ if the calendar policy version changed between the runs (less likely — both are post-PR-#18).
4. **A trade open at test_start in B closing earlier under stop=1.5.** This would not change `n_traces` (each bar still produces a trace) but should still be checked since "n_traces" definition assumes one trace per bar.

## What is NOT to be done now

- Do NOT re-run 90d A/B yet.
- Do NOT touch decision_engine / risk_gate.
- Do NOT change waveform connection.
- Do NOT push to main.
- Do NOT change USDJPY-specific behavior.

## What this document is for

- Pin the exact comparison conditions and the priority of comparison fields, so when (and only when) re-investigation is approved, it can be done with one targeted run rather than another full 4-symbol re-execution.
- Keep the open question visible so it does not silently disappear into a "POOL improved" headline.

## Next-step gate

Re-investigation requires explicit user approval and should be scoped to USDJPY=X only, not the full 4-symbol pool.
