"""Disk export for `decision_trace v1` outputs.

Writes one run's BarDecisionTrace records as a JSONL file alongside
`run_metadata.json` and `summary.json`. Pure I/O — no decision logic
runs here, the schema in `decision_trace.py` is the source of truth.

Layout
------
    out_dir/
        run_metadata.json
        decision_traces.jsonl
        summary.json

Strict overwrite guard
----------------------
`export_run` refuses to write if ANY of the three target paths already
exists (unless `overwrite=True`). Half-overwriting one file but leaving
the others stale would silently desynchronise run_id across the trio,
so the check is intentionally all-or-nothing.
"""
from __future__ import annotations

import gzip as _gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .decision_trace import TRACE_SCHEMA_VERSION

if TYPE_CHECKING:
    from .backtest_engine import EngineBacktestResult


_RUN_METADATA_NAME = "run_metadata.json"
_DECISION_TRACES_NAME = "decision_traces.jsonl"
_DECISION_TRACES_GZ_NAME = "decision_traces.jsonl.gz"
_SUMMARY_NAME = "summary.json"


def _decision_traces_filename(gzip_enabled: bool) -> str:
    return _DECISION_TRACES_GZ_NAME if gzip_enabled else _DECISION_TRACES_NAME


def _build_summary(
    result: "EngineBacktestResult",
    *,
    output_files: dict[str, str],
    gzip_enabled: bool,
) -> dict:
    rm = result.run_metadata
    assert rm is not None  # validated by caller
    summary = {
        "run_id": rm.run_id,
        "symbol": rm.symbol,
        "interval": rm.timeframe,
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bars_processed": result.bars_processed,
        "n_traces": len(result.decision_traces),
        "n_trades": len(result.trades),
        "metrics": result.metrics(),
        "output_files": dict(output_files),
        "export_gzip": bool(gzip_enabled),
    }
    # PR #20: r_candidates_summary (R1-R11 hypothesis cells). Computed
    # from the in-memory traces + reconstructed entry-trace-bound
    # trades. Pure observation; never read by the engine. Fail-soft —
    # any exception during compute is recorded as `r_candidates_error`
    # and the rest of summary.json is unaffected.
    try:
        from .decision_trace_r_candidates import compute_r_candidates_summary
        bound_trades = _build_trades_with_traces(result)
        test_period = None
        bar_range = getattr(rm, "bar_range", None) or {}
        if bar_range.get("start") and bar_range.get("end"):
            test_period = {
                "start": str(bar_range["start"]),
                "end":   str(bar_range["end"]),
            }
        summary["r_candidates_summary"] = compute_r_candidates_summary(
            traces=result.decision_traces,
            trades=bound_trades,
            test_period=test_period,
        )
    except Exception as e:  # noqa: BLE001
        summary["r_candidates_error"] = f"{type(e).__name__}: {e}"
    # PR-C2: technical_confluence_summary. Same fail-soft pattern as
    # r_candidates above — purely diagnostic, never read by decide().
    try:
        from .technical_confluence_summary import (
            compute_technical_confluence_summary,
        )
        summary["technical_confluence_summary"] = (
            compute_technical_confluence_summary(
                traces=result.decision_traces,
            )
        )
    except Exception as e:  # noqa: BLE001
        summary["technical_confluence_error"] = f"{type(e).__name__}: {e}"
    return summary


def _build_trades_with_traces(result: "EngineBacktestResult") -> list[dict]:
    """Bind each `result.trades[*]` to its entry-bar BarDecisionTrace.

    The R-helper needs `entry_trace` so it can pivot trades by macro /
    waveform / long-term observations at entry. Lookups are O(N+M)
    via a `(symbol, entry_ts)` → trace map.
    """
    trace_index: dict[tuple, "BarDecisionTrace"] = {}
    for tr in result.decision_traces:
        ts = getattr(tr, "timestamp", None)
        sym = getattr(tr, "symbol", None)
        if ts and sym:
            trace_index[(sym, str(ts))] = tr
    out: list[dict] = []
    for trade in result.trades:
        entry_ts = getattr(trade, "entry_ts", None)
        sym = getattr(trade, "symbol", None) or result.run_metadata.symbol
        entry_ts_iso = str(entry_ts) if entry_ts is not None else None
        entry_trace_obj = trace_index.get((sym, entry_ts_iso))
        # Pass the trace as a dict so the R-helper's `_safe_get` works
        # uniformly (real BarDecisionTrace exposes attrs; serialised dict
        # exposes keys; we accept both via _safe_get).
        out.append({
            "sym": sym,
            "side": getattr(trade, "side", None),
            "entry_ts": entry_ts_iso,
            "exit_ts": str(getattr(trade, "exit_ts", "")) or None,
            "exit_reason": getattr(trade, "exit_reason", None),
            "return_pct": float(getattr(trade, "return_pct", 0.0) or 0.0),
            "bars_held": int(getattr(trade, "bars_held", 0) or 0),
            "entry_trace": (
                entry_trace_obj.to_dict() if entry_trace_obj is not None
                else None
            ),
        })
    return out


def export_run(
    result: "EngineBacktestResult",
    *,
    out_dir: Path | str,
    overwrite: bool = False,
    gzip: bool = False,
) -> dict[str, Path]:
    """Write the three decision_trace artefacts for one backtest run.

    Parameters
    ----------
    result:
        The `EngineBacktestResult` from `run_engine_backtest(...,
        capture_traces=True)`. Must carry `run_metadata` and at least one
        BarDecisionTrace.
    out_dir:
        Destination directory. Created (with parents) if missing.
    overwrite:
        When False (default), raises FileExistsError if ANY of
        `run_metadata.json` / `decision_traces.jsonl` / `summary.json`
        already exists in `out_dir`. All-or-nothing on purpose so a
        partial overwrite cannot leave the trio out of sync.
    gzip:
        When True, decision traces are written as
        `decision_traces.jsonl.gz` (gzip-compressed JSONL). The summary's
        `output_files` reflects the chosen filename.

    Returns
    -------
    dict[str, Path] with keys "run_metadata", "decision_traces",
    "summary" pointing to the resolved absolute paths.

    Raises
    ------
    ValueError
        if `result.run_metadata is None` (capture_traces was False)
        or `len(result.decision_traces) == 0` (nothing to export).
    FileExistsError
        if `overwrite=False` and any target file already exists.
    """
    if result.run_metadata is None:
        raise ValueError(
            "export_run: result.run_metadata is None. "
            "Run with capture_traces=True before exporting."
        )
    if len(result.decision_traces) == 0:
        raise ValueError(
            "export_run: result.decision_traces is empty. "
            "Nothing to export — check warmup vs len(df)."
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces_filename = _decision_traces_filename(gzip)
    metadata_path = out_dir / _RUN_METADATA_NAME
    traces_path = out_dir / traces_filename
    summary_path = out_dir / _SUMMARY_NAME

    if not overwrite:
        existing = [p for p in (metadata_path, traces_path, summary_path) if p.exists()]
        if existing:
            names = ", ".join(p.name for p in existing)
            raise FileExistsError(
                f"export_run: refusing to overwrite existing file(s) in "
                f"{out_dir}: {names}. Pass overwrite=True to replace."
            )

    # 1. run_metadata.json — single dict
    metadata_dict = result.to_run_metadata_dict()
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata_dict, fp, ensure_ascii=False, default=str, indent=2)

    # 2. decision_traces.jsonl — one record per line, streamed
    if gzip:
        traces_fp = _gzip.open(traces_path, "wt", encoding="utf-8")  # type: ignore[assignment]
    else:
        traces_fp = traces_path.open("w", encoding="utf-8")
    try:
        for trace in result.decision_traces:
            record = trace.to_dict()
            traces_fp.write(json.dumps(record, ensure_ascii=False, default=str))
            traces_fp.write("\n")
    finally:
        traces_fp.close()

    # 3. summary.json — last so the other two are guaranteed to be on
    #    disk before the index pointing at them is written.
    output_files = {
        "run_metadata": _RUN_METADATA_NAME,
        "decision_traces": traces_filename,
        "summary": _SUMMARY_NAME,
    }
    summary_dict = _build_summary(
        result, output_files=output_files, gzip_enabled=gzip
    )
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summary_dict, fp, ensure_ascii=False, default=str, indent=2)

    return {
        "run_metadata": metadata_path.resolve(),
        "decision_traces": traces_path.resolve(),
        "summary": summary_path.resolve(),
    }


__all__ = ["export_run"]
