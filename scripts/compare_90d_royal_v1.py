"""Single-process 4-symbol x 2-profile 90d comparison.

Spec contract:
  - One Python process. One yfinance fetch per symbol for the test
    window, one for the context window. Same df, same df_context,
    same events tuple, same waveform library are passed to BOTH
    profiles for that symbol so n_traces / data_snapshot_hash MUST
    match — re-running with two CLI invocations was the source of
    the prior USDJPY=X mismatch and is forbidden here.
  - Verifies hash / n_traces / bar_range / first/last ts / events
    sha8 / event_window_policy_version per symbol. On mismatch the
    symbol is marked provisional/invalid in notes; NO retry.
  - Writes runs/ab_90d_royal_v1/<sym>/ + pool / verification /
    difference_type / royal_blocked / royal_added / block_reasons /
    R12-R16 JSON. runs/ is git-ignored — never committed.

This is research-only diagnostic output. 90d single-window. Adoption
decisions are NOT made by this script. user is the only judge.
"""
from __future__ import annotations

import json
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Make `src.fx.*` importable when this script is run as
# `python scripts/compare_90d_royal_v1.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.fx.backtest_engine import run_engine_backtest
from src.fx.calendar import load_events
from src.fx.data import fetch_ohlcv
from src.fx.decision_trace_build import compute_library_id
from src.fx.decision_trace_io import export_run
from src.fx.macro import fetch_macro_snapshot
from src.fx.waveform_library import read_library


TEST_START   = "2026-01-30"
TEST_END     = "2026-04-29"
INTERVAL     = "1h"
CONTEXT_DAYS = 365
EVENTS_PATH  = Path("data/events.json")
OUT_ROOT     = Path("runs/ab_90d_royal_v1")
SYMBOLS      = ["USDJPY=X", "EURUSD=X", "GBPUSD=X", "AUDUSD=X"]
PROFILES     = ["current_runtime", "royal_road_decision_v1"]

LIB_BY_SYMBOL = {
    "USDJPY=X": Path("libs/usdjpy_1h_train_v2.jsonl"),
    "EURUSD=X": Path("libs/eurusd_1h_train_v2.jsonl"),
    "GBPUSD=X": Path("libs/gbpusd_1h_train_v2.jsonl"),
    "AUDUSD=X": Path("libs/audusd_1h_train_v2.jsonl"),
}


def _safe(v):
    """Round / coerce numeric for clean JSON output."""
    if isinstance(v, float):
        return round(v, 6)
    return v


def _metrics_block(res):
    m = res.metrics()
    trades = res.trades
    n = len(trades)
    wins = [t for t in trades if (t.return_pct or 0) > 0]
    losses = [t for t in trades if (t.return_pct or 0) <= 0]
    avg_win = (
        sum(t.return_pct for t in wins) / len(wins) if wins else None
    )
    avg_loss = (
        sum(t.return_pct for t in losses) / len(losses) if losses else None
    )
    max_streak = 0
    cur_streak = 0
    for t in trades:
        if (t.return_pct or 0) <= 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
    avg_holding = sum(t.bars_held for t in trades) / n if n else None
    exit_reasons = Counter(t.exit_reason for t in trades)
    return {
        "n_traces": len(res.decision_traces),
        "n_trades": n,
        "win_rate":         _safe(m.get("win_rate")),
        "profit_factor":    _safe(m.get("profit_factor")),
        "total_return_pct": _safe(m.get("total_return_pct")),
        "max_drawdown_pct": _safe(m.get("max_drawdown_pct")),
        "avg_win_pct":      _safe(avg_win),
        "avg_loss_pct":     _safe(avg_loss),
        "max_consecutive_losses": int(max_streak),
        "avg_holding_bars":  _safe(avg_holding),
        "exit_reasons":     dict(exit_reasons),
        "hold_reasons":     dict(res.hold_reasons),
    }


def _ts_iso(t):
    if hasattr(t, "isoformat"):
        return t.isoformat()
    return str(t)


def _verify(res_a, res_b) -> dict:
    rm_a = res_a.run_metadata
    rm_b = res_b.run_metadata
    cal_a = rm_a.calendar or {}
    cal_b = rm_b.calendar or {}
    ctx_a = rm_a.context or {}
    ctx_b = rm_b.context or {}
    first_a = res_a.decision_traces[0].timestamp if res_a.decision_traces else None
    first_b = res_b.decision_traces[0].timestamp if res_b.decision_traces else None
    last_a = res_a.decision_traces[-1].timestamp if res_a.decision_traces else None
    last_b = res_b.decision_traces[-1].timestamp if res_b.decision_traces else None
    checks = {
        "data_snapshot_hash":      rm_a.data_snapshot_hash == rm_b.data_snapshot_hash,
        "n_traces":                len(res_a.decision_traces) == len(res_b.decision_traces),
        "bar_range":               rm_a.bar_range == rm_b.bar_range,
        "n_test_bars":             ctx_a.get("n_test_bars") == ctx_b.get("n_test_bars"),
        "first_trace_ts":          first_a == first_b,
        "last_trace_ts":           last_a == last_b,
        "events_file_sha8":        cal_a.get("events_file_sha8") == cal_b.get("events_file_sha8"),
        "event_window_policy_version":
            cal_a.get("event_window_policy_version") == cal_b.get("event_window_policy_version"),
    }
    valid = all(checks.values())
    return {
        "status":                  "valid" if valid else "invalid",
        "checks":                  checks,
        "data_snapshot_hash_a":    rm_a.data_snapshot_hash,
        "data_snapshot_hash_b":    rm_b.data_snapshot_hash,
        "n_traces_a":              len(res_a.decision_traces),
        "n_traces_b":              len(res_b.decision_traces),
        "bar_range_a":             dict(rm_a.bar_range),
        "bar_range_b":             dict(rm_b.bar_range),
        "first_trace_ts_a":        first_a,
        "first_trace_ts_b":        first_b,
        "last_trace_ts_a":         last_a,
        "last_trace_ts_b":         last_b,
        "events_file_sha8_a":      cal_a.get("events_file_sha8"),
        "events_file_sha8_b":      cal_b.get("events_file_sha8"),
        "event_window_policy_version_a":
            cal_a.get("event_window_policy_version"),
        "event_window_policy_version_b":
            cal_b.get("event_window_policy_version"),
    }


def _classify_outcome(t):
    if (t.return_pct or 0) > 0:
        return "won"
    return "lost"


def _diff_by_ts(res_b) -> dict:
    out = {}
    for tr in res_b.decision_traces:
        rrd = tr.royal_road_decision
        if rrd is not None:
            out[tr.timestamp] = rrd.compared_to_current_runtime["difference_type"]
    return out


def _royal_blocked_outcomes(res_a, diff_by_ts) -> dict:
    won = [
        t for t in res_a.trades
        if diff_by_ts.get(_ts_iso(t.entry_ts)) in
            ("current_buy_royal_hold", "current_sell_royal_hold")
        and (t.return_pct or 0) > 0
    ]
    lost = [
        t for t in res_a.trades
        if diff_by_ts.get(_ts_iso(t.entry_ts)) in
            ("current_buy_royal_hold", "current_sell_royal_hold")
        and (t.return_pct or 0) <= 0
    ]
    return {
        "n":                  len(won) + len(lost),
        "won_n":              len(won),
        "lost_n":             len(lost),
        "won_sum_return_pct": _safe(sum(t.return_pct for t in won)),
        "lost_sum_return_pct": _safe(sum(t.return_pct for t in lost)),
    }


def _royal_added_outcomes(res_b, diff_by_ts) -> dict:
    won = [
        t for t in res_b.trades
        if diff_by_ts.get(_ts_iso(t.entry_ts)) in
            ("current_hold_royal_buy", "current_hold_royal_sell")
        and (t.return_pct or 0) > 0
    ]
    lost = [
        t for t in res_b.trades
        if diff_by_ts.get(_ts_iso(t.entry_ts)) in
            ("current_hold_royal_buy", "current_hold_royal_sell")
        and (t.return_pct or 0) <= 0
    ]
    return {
        "n":                  len(won) + len(lost),
        "won_n":              len(won),
        "lost_n":             len(lost),
        "won_sum_return_pct": _safe(sum(t.return_pct for t in won)),
        "lost_sum_return_pct": _safe(sum(t.return_pct for t in lost)),
    }


def _royal_block_reasons(res_b) -> dict:
    single = Counter()
    combo = Counter()
    for tr in res_b.decision_traces:
        rrd = tr.royal_road_decision
        if rrd is None:
            continue
        br = rrd.block_reasons or []
        for r in br:
            single[r] += 1
        if br:
            combo[tuple(sorted(br))] += 1
    return {
        "single_top": [
            {"reason": r, "count": c} for r, c in single.most_common(20)
        ],
        "combo_top": [
            {"combo": list(c), "count": n} for c, n in combo.most_common(10)
        ],
    }


def _r12_to_r16_from_summary(summary: dict) -> dict:
    rc = summary.get("r_candidates_summary", {})
    return {
        k: rc.get(k) for k in (
            "r12_confluence_label_outcome",
            "r13_sr_proximity_outcome",
            "r14_candlestick_confirmation_outcome",
            "r15_rsi_regime_danger_outcome",
            "r16_structure_stop_distance_outcome",
        )
    }


def run_one_symbol(sym: str) -> dict:
    sym_dir = OUT_ROOT / sym.replace("=", "_")
    sym_dir.mkdir(parents=True, exist_ok=True)
    diff_dir = sym_dir / "diff"
    diff_dir.mkdir(parents=True, exist_ok=True)

    notes_lines: list[str] = []
    print(f"[{sym}] fetching test window {TEST_START}..{TEST_END} ...", flush=True)
    df = fetch_ohlcv(sym, interval=INTERVAL, start=TEST_START, end=TEST_END)
    print(f"[{sym}]   df: {len(df)} bars, {df.index.min()} .. {df.index.max()}", flush=True)
    if len(df) == 0:
        raise RuntimeError(f"{sym}: empty test fetch")

    test_start_ts = pd.Timestamp(df.index.min())
    ctx_end = test_start_ts.strftime("%Y-%m-%d")
    ctx_start = (
        test_start_ts - pd.Timedelta(days=CONTEXT_DAYS)
    ).strftime("%Y-%m-%d")
    print(f"[{sym}] fetching context {ctx_start}..{ctx_end} ...", flush=True)
    df_context = fetch_ohlcv(
        sym, interval=INTERVAL, start=ctx_start, end=ctx_end,
    )
    if len(df_context) > 0:
        df_context = df_context[df_context.index < df.index.min()]
    print(f"[{sym}]   df_context: {len(df_context)} bars", flush=True)

    print(f"[{sym}] fetching macro ...", flush=True)
    try:
        macro = fetch_macro_snapshot(df.index, interval="1d", period="2y")
    except Exception as e:
        macro = None
        notes_lines.append(f"macro fetch failed: {e}")
        print(f"[{sym}]   macro fetch FAILED: {e}", flush=True)

    lib_path = LIB_BY_SYMBOL[sym]
    print(f"[{sym}] loading waveform library {lib_path} ...", flush=True)
    wf_lib = read_library(lib_path)
    wf_lib_id = compute_library_id(wf_lib, str(lib_path))
    print(f"[{sym}]   wf samples: {len(wf_lib)}, id: {wf_lib_id}", flush=True)

    events_list = load_events(EVENTS_PATH)
    events_tuple = tuple(events_list)
    print(f"[{sym}] events loaded: {len(events_list)}", flush=True)

    common = dict(
        interval=INTERVAL,
        events=events_tuple,
        df_context=df_context,
        context_days=CONTEXT_DAYS,
        macro=macro,
        waveform_library=wf_lib,
        waveform_library_id=wf_lib_id,
    )

    print(f"[{sym}] running profile A: current_runtime ...", flush=True)
    res_a = run_engine_backtest(
        df, sym, **common, decision_profile="current_runtime",
    )
    print(f"[{sym}]   A: n_traces={len(res_a.decision_traces)} n_trades={len(res_a.trades)}", flush=True)

    print(f"[{sym}] running profile B: royal_road_decision_v1 ...", flush=True)
    res_b = run_engine_backtest(
        df, sym, **common, decision_profile="royal_road_decision_v1",
    )
    print(f"[{sym}]   B: n_traces={len(res_b.decision_traces)} n_trades={len(res_b.trades)}", flush=True)

    out_a = sym_dir / "current_runtime"
    out_b = sym_dir / "royal_road_decision_v1"
    out_a.mkdir(exist_ok=True)
    out_b.mkdir(exist_ok=True)
    export_run(result=res_a, out_dir=out_a, overwrite=True)
    export_run(result=res_b, out_dir=out_b, overwrite=True)

    verification = _verify(res_a, res_b)
    if verification["status"] != "valid":
        bad = [k for k, v in verification["checks"].items() if not v]
        notes_lines.append(
            f"verification INVALID. failing checks: {bad}. "
            "Symbol marked provisional. NO retry per spec."
        )

    metrics = {
        "current_runtime":         _metrics_block(res_a),
        "royal_road_decision_v1":  _metrics_block(res_b),
    }
    diff_by_ts = _diff_by_ts(res_b)
    diff_dist = Counter(diff_by_ts.values())
    blocked = _royal_blocked_outcomes(res_a, diff_by_ts)
    added = _royal_added_outcomes(res_b, diff_by_ts)
    block_reasons = _royal_block_reasons(res_b)

    sum_a = json.loads((out_a / "summary.json").read_text())
    sum_b = json.loads((out_b / "summary.json").read_text())
    r12_to_r16 = {
        "current_runtime":        _r12_to_r16_from_summary(sum_a),
        "royal_road_decision_v1": _r12_to_r16_from_summary(sum_b),
    }
    label_dist = {
        "current_runtime": (
            sum_a.get("technical_confluence_summary") or {}
        ).get("label_distribution"),
        "royal_road_decision_v1": (
            sum_b.get("technical_confluence_summary") or {}
        ).get("label_distribution"),
    }

    notes_path = diff_dir / "notes.md"
    notes_body = (
        f"# {sym} 90d compare notes\n\n"
        f"Status: {verification['status']}\n\n"
        + (
            "\n".join(f"- {ln}" for ln in notes_lines) + "\n\n"
            if notes_lines else ""
        )
        + f"verification:\n```json\n{json.dumps(verification, indent=2, default=str)}\n```\n"
    )
    notes_path.write_text(notes_body)

    return {
        "verification": verification,
        "metrics": metrics,
        "diff_type": dict(diff_dist),
        "blocked": blocked,
        "added": added,
        "block_reasons": block_reasons,
        "r12_to_r16": r12_to_r16,
        "label_distribution": label_dist,
        "notes": notes_lines,
    }


def aggregate_pool(per_symbol: dict) -> dict:
    pool = {p: {
        "n_traces": 0,
        "n_trades": 0,
        "sum_total_return_pct": 0.0,
        "exit_reasons": Counter(),
        "hold_reasons": Counter(),
    } for p in PROFILES}
    for sym, payload in per_symbol.items():
        for p in PROFILES:
            m = payload["metrics"][p]
            pool[p]["n_traces"] += m["n_traces"]
            pool[p]["n_trades"] += m["n_trades"]
            pool[p]["sum_total_return_pct"] += m["total_return_pct"] or 0.0
            pool[p]["exit_reasons"].update(m["exit_reasons"])
            pool[p]["hold_reasons"].update(m["hold_reasons"])
    for p in PROFILES:
        pool[p]["sum_total_return_pct"] = round(
            pool[p]["sum_total_return_pct"], 6
        )
        pool[p]["exit_reasons"] = dict(pool[p]["exit_reasons"])
        pool[p]["hold_reasons"] = dict(pool[p]["hold_reasons"])
    return pool


def aggregate_diff_types(per_symbol: dict) -> dict:
    out = {"per_symbol": {}, "pool": Counter()}
    for sym, payload in per_symbol.items():
        out["per_symbol"][sym] = dict(payload["diff_type"])
        out["pool"].update(payload["diff_type"])
    out["pool"] = dict(out["pool"])
    return out


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    print(f"start: {started}")
    print(f"out_root: {OUT_ROOT.resolve()}")

    per_symbol: dict[str, dict] = {}
    fatal: list[str] = []
    for sym in SYMBOLS:
        try:
            per_symbol[sym] = run_one_symbol(sym)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[{sym}] FATAL: {e}\n{tb}", file=sys.stderr, flush=True)
            fatal.append(f"{sym}: {type(e).__name__}: {e}")
            continue

    # Per-symbol metrics dump
    per_symbol_metrics = {
        sym: payload["metrics"] for sym, payload in per_symbol.items()
    }
    (OUT_ROOT / "per_symbol_metrics.json").write_text(
        json.dumps(per_symbol_metrics, indent=2, default=str)
    )

    # Pool aggregate
    pool = aggregate_pool(per_symbol)
    (OUT_ROOT / "pool_summary.json").write_text(
        json.dumps(pool, indent=2, default=str)
    )

    # Verification summary
    ver = {sym: payload["verification"] for sym, payload in per_symbol.items()}
    (OUT_ROOT / "verification_summary.json").write_text(
        json.dumps(ver, indent=2, default=str)
    )

    # Diff type summary
    diff_summary = aggregate_diff_types(per_symbol)
    (OUT_ROOT / "difference_type_summary.json").write_text(
        json.dumps(diff_summary, indent=2, default=str)
    )

    # Royal blocked / added trade outcomes
    (OUT_ROOT / "royal_blocked_trade_outcomes.json").write_text(
        json.dumps(
            {sym: p["blocked"] for sym, p in per_symbol.items()},
            indent=2, default=str,
        )
    )
    (OUT_ROOT / "royal_added_trade_outcomes.json").write_text(
        json.dumps(
            {sym: p["added"] for sym, p in per_symbol.items()},
            indent=2, default=str,
        )
    )
    (OUT_ROOT / "royal_block_reasons.json").write_text(
        json.dumps(
            {sym: p["block_reasons"] for sym, p in per_symbol.items()},
            indent=2, default=str,
        )
    )
    (OUT_ROOT / "r12_to_r16_compare.json").write_text(
        json.dumps(
            {sym: p["r12_to_r16"] for sym, p in per_symbol.items()},
            indent=2, default=str,
        )
    )
    (OUT_ROOT / "label_distribution_compare.json").write_text(
        json.dumps(
            {sym: p["label_distribution"] for sym, p in per_symbol.items()},
            indent=2, default=str,
        )
    )

    # Top-level notes
    finished = datetime.now(timezone.utc).isoformat()
    lines = [
        "# 90d royal_road_decision_v1 compare — top-level notes",
        f"started:  {started}",
        f"finished: {finished}",
        f"symbols:  {SYMBOLS}",
        f"profiles: {PROFILES}",
        "",
        "## Per-symbol verification status:",
    ]
    for sym in SYMBOLS:
        if sym in per_symbol:
            v = per_symbol[sym]["verification"]
            lines.append(f"- {sym}: {v['status']}")
            for ln in per_symbol[sym]["notes"]:
                lines.append(f"    note: {ln}")
        else:
            lines.append(f"- {sym}: ERROR (see fatal section)")

    if fatal:
        lines.append("")
        lines.append("## Fatal errors:")
        for f in fatal:
            lines.append(f"- {f}")

    (OUT_ROOT / "notes.md").write_text("\n".join(lines) + "\n")
    print("done.")
    return 0 if not fatal else 1


if __name__ == "__main__":
    sys.exit(main())
