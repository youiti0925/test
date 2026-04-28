"""Tests for cmd_trade live decision_trace export (PR #12)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ohlcv(n: int, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {"open": close, "high": close + 0.3, "low": close - 0.3,
         "close": close, "volume": [1000] * n},
        index=idx,
    )


def _build_trade_namespace(**overrides):
    """Minimum argparse.Namespace cmd_trade reads."""
    ns = argparse.Namespace(
        symbol="USDJPY=X",
        interval="1h",
        period="60d",
        capital=10_000.0,
        risk_pct=None,
        stop_atr=None,
        tp_atr=None,
        broker="paper",
        dry_run=True,
        confirm_demo=False,
        trace_out=None,
        trace_out_default=False,
        overwrite=False,
        # _add_context_flags defaults — just enough to make
        # _gather_inputs not blow up:
        no_news=True,
        news_limit=0,
        no_correlation=True,
        no_events=False,        # honour data/events.json
        event_window_hours=48,
        no_lessons=True,
        lesson_limit=0,
        no_sentiment=True,
        no_higher_tf=True,      # avoid network fetch in tests
        no_llm=True,
        strategy_config=None,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def cmd_trade_with_stub(monkeypatch, tmp_path):
    """Stub network calls and storage so cmd_trade runs in tmp_path.

    Returns (cmd_fn, storage, cwd, cfg).
    """
    from src.fx import cli as cli_mod

    df = _ohlcv(200, seed=11)

    def fake_fetch_ohlcv(symbol, interval, period=None, start=None, end=None):
        return df

    class FakeStorage:
        def relevant_postmortems(self, symbol, limit=0):
            return []

        def save_trade(self, **kwargs):
            return 1

    monkeypatch.setattr(cli_mod, "fetch_ohlcv", fake_fetch_ohlcv)
    monkeypatch.setattr(cli_mod, "_print", lambda *a, **k: None)

    monkeypatch.chdir(tmp_path)

    class Cfg:
        anthropic_api_key = None
        model = "claude-sonnet-4-6"
        effort = "medium"
        db_path = ":memory:"

    return cli_mod.cmd_trade, FakeStorage(), tmp_path, Cfg()


# ─── Trace export contract ─────────────────────────────────────────────────


def test_trade_dry_run_with_trace_out_default_writes_jsonl(cmd_trade_with_stub):
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace(
        dry_run=True, trace_out_default=True,
    )
    rc = cmd(args, cfg, storage)
    assert rc == 0
    runs_dir = cwd / "runs" / "live"
    assert runs_dir.exists()
    children = list(runs_dir.iterdir())
    assert len(children) == 1, (
        f"expected exactly one run subdir, got {[c.name for c in children]}"
    )
    run_dir = children[0]
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "decision_traces.jsonl").exists()


def test_trade_dry_run_jsonl_carries_exactly_one_line(cmd_trade_with_stub):
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace(trace_out_default=True)
    cmd(args, cfg, storage)
    run_dir = next((cwd / "runs" / "live").iterdir())
    lines = (run_dir / "decision_traces.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    lines = [ln for ln in lines if ln.strip()]
    assert len(lines) == 1


def test_trade_trace_carries_future_outcome_null(cmd_trade_with_stub):
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace(trace_out_default=True)
    cmd(args, cfg, storage)
    run_dir = next((cwd / "runs" / "live").iterdir())
    rec = json.loads((run_dir / "decision_traces.jsonl").read_text())
    assert rec["future_outcome"] is None, (
        "live trace must not pre-compute future_outcome — that's an "
        "offline-only enrichment"
    )


def test_run_metadata_execution_mode_distinguishes_dry_run(cmd_trade_with_stub):
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace(dry_run=True, trace_out_default=True)
    cmd(args, cfg, storage)
    run_dir = next((cwd / "runs" / "live").iterdir())
    rm = json.loads((run_dir / "run_metadata.json").read_text())
    assert rm["execution_mode"] == "live_dry_run"


def test_run_metadata_execution_mode_when_paper_actual(cmd_trade_with_stub):
    """Non-dry-run paper broker — engine still produces a trace; the
    decision happens to be HOLD in this fixture, but execution_mode
    must reflect the broker label, not the dry-run flag."""
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace(
        dry_run=False, broker="paper", trace_out_default=True,
    )
    cmd(args, cfg, storage)
    run_dir = next((cwd / "runs" / "live").iterdir())
    rm = json.loads((run_dir / "run_metadata.json").read_text())
    # When dry_run=False, execution_mode encodes the broker label.
    assert rm["execution_mode"] == "live_paper"


def test_trade_explicit_trace_out_wins(cmd_trade_with_stub):
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    explicit_dir = cwd / "explicit_live"
    args = _build_trade_namespace(
        trace_out=str(explicit_dir),
        trace_out_default=True,
    )
    cmd(args, cfg, storage)
    assert (explicit_dir / "decision_traces.jsonl").exists()
    # runs/live/<run_id>/ must NOT have been created since --trace-out won
    assert not (cwd / "runs" / "live").exists()


def test_trade_no_flag_does_not_export(cmd_trade_with_stub):
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace()  # neither flag
    rc = cmd(args, cfg, storage)
    assert rc == 0
    assert not (cwd / "runs").exists()
    assert not (cwd / "runs" / "live").exists()


# ─── HOLD vs directional path both emit a trace ────────────────────────────


def _make_hold_inducing_args(**overrides):
    """Args force HOLD via --no-events + a calm random fixture; with
    seed 11 the engine returns HOLD on the last bar (the existing
    backtest fixture verified this)."""
    return _build_trade_namespace(trace_out_default=True, **overrides)


def test_hold_path_writes_trace(cmd_trade_with_stub):
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _make_hold_inducing_args(dry_run=False)
    rc = cmd(args, cfg, storage)
    assert rc == 0
    run_dir = next((cwd / "runs" / "live").iterdir())
    rec = json.loads((run_dir / "decision_traces.jsonl").read_text())
    assert rec["decision"]["final_action"] == "HOLD"
    assert rec["execution_trace"]["entry_executed"] is False
    assert rec["execution_trace"]["entry_skipped_reason"] in (
        "decision_was_HOLD",
    )


def test_directional_dry_run_records_advisory_status(
    cmd_trade_with_stub, monkeypatch
):
    """Force the engine to return BUY by stubbing decide_action; the
    execution_trace must record entry_executed=False, but the
    entry_skipped_reason MUST stay within the closed taxonomy
    (ENTRY_SKIPPED_REASONS) — `dry_run` is recorded in
    decision.advisory.live_execution_status, NOT in entry_skipped_reason."""
    from src.fx import cli as cli_mod
    from src.fx.decision_engine import Decision
    from src.fx.decision_trace import ENTRY_SKIPPED_REASONS

    def stub_buy(*, technical_signal, pattern, higher_timeframe_trend,
                 risk_reward, risk_state, llm_signal=None,
                 waveform_bias=None, min_confidence=0.6, min_risk_reward=1.5):
        return Decision(
            action="BUY", confidence=0.9,
            reason="forced BUY for test",
            blocked_by=(),
            rule_chain=("risk_gate", "technical_directionality",
                        "pattern_check", "higher_tf_alignment",
                        "risk_reward_floor"),
            advisory={},
        )

    monkeypatch.setattr(cli_mod, "decide_action", stub_buy)

    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace(dry_run=True, trace_out_default=True)
    rc = cmd(args, cfg, storage)
    assert rc == 0
    run_dir = next((cwd / "runs" / "live").iterdir())
    rec = json.loads((run_dir / "decision_traces.jsonl").read_text())
    assert rec["decision"]["final_action"] == "BUY"
    assert rec["execution_trace"]["entry_executed"] is False
    # entry_skipped_reason MUST stay within the closed taxonomy.
    assert rec["execution_trace"]["entry_skipped_reason"] in ENTRY_SKIPPED_REASONS
    # The engine path was clean (BUY decision); broker layer skipped due
    # to dry-run, so the engine-level sentinel "entry_executed" applies.
    assert rec["execution_trace"]["entry_skipped_reason"] == "entry_executed"
    # Live-specific disposition lives in advisory.
    assert rec["decision"]["advisory"]["live_execution_status"] == "dry_run"
    assert rec["decision"]["advisory"]["live_dry_run"] is True
    assert rec["decision"]["advisory"]["live_broker"] == "paper"
    assert rec["decision"]["advisory"]["live_order_placed"] is False


def test_directional_paper_broker_records_executed(
    cmd_trade_with_stub, monkeypatch
):
    """With dry_run=False and a forced BUY decision, paper broker takes
    the order; trace must record entry_executed=True with the entry
    price coming through the fill object."""
    from src.fx import cli as cli_mod
    from src.fx.decision_engine import Decision

    def stub_buy(*, technical_signal, pattern, higher_timeframe_trend,
                 risk_reward, risk_state, llm_signal=None,
                 waveform_bias=None, min_confidence=0.6, min_risk_reward=1.5):
        return Decision(
            action="BUY", confidence=0.9,
            reason="forced BUY for test",
            blocked_by=(),
            rule_chain=("risk_gate", "technical_directionality",
                        "pattern_check", "higher_tf_alignment",
                        "risk_reward_floor"),
            advisory={},
        )

    monkeypatch.setattr(cli_mod, "decide_action", stub_buy)

    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace(dry_run=False, trace_out_default=True)
    rc = cmd(args, cfg, storage)
    assert rc == 0
    run_dir = next((cwd / "runs" / "live").iterdir())
    rec = json.loads((run_dir / "decision_traces.jsonl").read_text())
    assert rec["decision"]["final_action"] == "BUY"
    assert rec["execution_trace"]["entry_executed"] is True
    assert rec["execution_trace"]["entry_skipped_reason"] == "entry_executed"
    # advisory carries broker label and the matching live_execution_status
    assert rec["decision"]["advisory"]["live_broker"] == "paper"
    assert rec["decision"]["advisory"]["live_dry_run"] is False
    assert rec["decision"]["advisory"]["live_order_placed"] is True
    assert rec["decision"]["advisory"]["live_execution_status"] == "entry_executed"


# ─── trace-stats reads the live JSONL ──────────────────────────────────────


def test_trace_stats_reads_live_trace_jsonl(cmd_trade_with_stub):
    """Pipeline integration: live trace JSONL must be readable by the
    same trace-stats CLI used for backtest, including cross_stats."""
    from src.fx.decision_trace_stats import aggregate_stats

    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace(trace_out_default=True)
    cmd(args, cfg, storage)
    jsonl = next((cwd / "runs" / "live").iterdir()) / "decision_traces.jsonl"

    stats = aggregate_stats(jsonl)
    assert stats["metadata"]["n_traces"] == 1
    cs = stats["cross_stats"]
    # cross_stats schema must contain all PR #7 + PR #10 sections
    assert "hold_reason_outcome" in cs
    assert "blocked_by_outcome" in cs
    # blocked_by_outcome carries either no_block (most common) or a
    # real Risk Gate code if events.json caused a block on this bar.
    assert len(cs["blocked_by_outcome"]) >= 1


def test_run_id_consistent_between_metadata_and_jsonl(cmd_trade_with_stub):
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    args = _build_trade_namespace(trace_out_default=True)
    cmd(args, cfg, storage)
    run_dir = next((cwd / "runs" / "live").iterdir())
    rm = json.loads((run_dir / "run_metadata.json").read_text())
    rec = json.loads((run_dir / "decision_traces.jsonl").read_text())
    assert rm["run_id"] == rec["run_id"], (
        f"run_id mismatch: metadata={rm['run_id']!r} vs "
        f"trace={rec['run_id']!r}"
    )


# ─── Existing trace contracts unchanged ────────────────────────────────────


def test_existing_backtest_trace_export_unchanged(tmp_path):
    """Sanity: PR #5/PR #8 backtest export pipeline should still work
    end-to-end after the new live helpers landed."""
    from src.fx.backtest_engine import run_engine_backtest
    from src.fx.decision_trace_io import export_run

    df = _ohlcv(200, seed=42)
    res = run_engine_backtest(df, "X", interval="1h", warmup=60)
    paths = export_run(res, out_dir=tmp_path / "bt")
    assert paths["run_metadata"].exists()
    assert paths["decision_traces"].exists()
    assert paths["summary"].exists()


# ─── ENTRY_SKIPPED_REASONS taxonomy compliance ─────────────────────────────


def test_all_live_trace_paths_use_closed_entry_skipped_reason_taxonomy(
    cmd_trade_with_stub, monkeypatch
):
    """ALL live exit branches (HOLD / dry-run BUY / paper BUY actual) must
    write `entry_skipped_reason` values that are members of the closed
    taxonomy `ENTRY_SKIPPED_REASONS`. PR #12 review correction."""
    from src.fx import cli as cli_mod
    from src.fx.decision_engine import Decision
    from src.fx.decision_trace import ENTRY_SKIPPED_REASONS

    cmd, storage, cwd, cfg = cmd_trade_with_stub

    def _read_skip_reason(run_subdir: Path) -> str:
        rec = json.loads(
            (run_subdir / "decision_traces.jsonl").read_text(encoding="utf-8")
        )
        return rec["execution_trace"]["entry_skipped_reason"]

    # 1. HOLD path — engine returns HOLD on the seed=11 fixture
    args = _build_trade_namespace(dry_run=False, trace_out_default=True)
    cmd(args, cfg, storage)
    run_subdir = next((cwd / "runs" / "live").iterdir())
    hold_reason = _read_skip_reason(run_subdir)
    assert hold_reason in ENTRY_SKIPPED_REASONS, (
        f"HOLD path wrote {hold_reason!r}; not in ENTRY_SKIPPED_REASONS"
    )

    # Clear runs/live/ so the next path lands in a fresh subdir
    import shutil
    shutil.rmtree(cwd / "runs")

    def stub_buy(*, technical_signal, pattern, higher_timeframe_trend,
                 risk_reward, risk_state, llm_signal=None,
                 waveform_bias=None, min_confidence=0.6, min_risk_reward=1.5):
        return Decision(
            action="BUY", confidence=0.9, reason="t",
            blocked_by=(), rule_chain=("risk_gate",), advisory={},
        )

    monkeypatch.setattr(cli_mod, "decide_action", stub_buy)

    # 2. dry-run BUY path
    args = _build_trade_namespace(dry_run=True, trace_out_default=True)
    cmd(args, cfg, storage)
    run_subdir = next((cwd / "runs" / "live").iterdir())
    dry_reason = _read_skip_reason(run_subdir)
    assert dry_reason in ENTRY_SKIPPED_REASONS, (
        f"dry-run BUY path wrote {dry_reason!r}; not in ENTRY_SKIPPED_REASONS"
    )

    shutil.rmtree(cwd / "runs")

    # 3. live paper BUY path
    args = _build_trade_namespace(dry_run=False, trace_out_default=True)
    cmd(args, cfg, storage)
    run_subdir = next((cwd / "runs" / "live").iterdir())
    live_reason = _read_skip_reason(run_subdir)
    assert live_reason in ENTRY_SKIPPED_REASONS, (
        f"live paper BUY path wrote {live_reason!r}; "
        f"not in ENTRY_SKIPPED_REASONS"
    )


def test_live_execution_status_distinguishes_dry_run_paper_hold(
    cmd_trade_with_stub, monkeypatch
):
    """The closed `entry_skipped_reason` taxonomy cannot tell live cases
    apart on its own — `decision.advisory.live_execution_status` must
    carry the disposition (`hold` / `dry_run` / `entry_executed` /
    `order_not_placed`)."""
    from src.fx import cli as cli_mod
    from src.fx.decision_engine import Decision

    cmd, storage, cwd, cfg = cmd_trade_with_stub
    import shutil

    def _read_status(run_subdir: Path) -> str:
        rec = json.loads(
            (run_subdir / "decision_traces.jsonl").read_text(encoding="utf-8")
        )
        return rec["decision"]["advisory"]["live_execution_status"]

    # HOLD case: status should be "hold"
    cmd(_build_trade_namespace(dry_run=False, trace_out_default=True),
        cfg, storage)
    assert _read_status(next((cwd / "runs" / "live").iterdir())) == "hold"
    shutil.rmtree(cwd / "runs")

    # Force BUY for the next two cases.
    def stub_buy(*, technical_signal, pattern, higher_timeframe_trend,
                 risk_reward, risk_state, llm_signal=None,
                 waveform_bias=None, min_confidence=0.6, min_risk_reward=1.5):
        return Decision(
            action="BUY", confidence=0.9, reason="t",
            blocked_by=(), rule_chain=("risk_gate",), advisory={},
        )

    monkeypatch.setattr(cli_mod, "decide_action", stub_buy)

    # dry-run BUY case: status should be "dry_run"
    cmd(_build_trade_namespace(dry_run=True, trace_out_default=True),
        cfg, storage)
    assert _read_status(next((cwd / "runs" / "live").iterdir())) == "dry_run"
    shutil.rmtree(cwd / "runs")

    # live paper BUY: status should be "entry_executed"
    cmd(_build_trade_namespace(dry_run=False, trace_out_default=True),
        cfg, storage)
    assert _read_status(next((cwd / "runs" / "live").iterdir())) == "entry_executed"


# ─── Trace export failure surfacing ───────────────────────────────────────


def test_trace_export_failure_returns_exit_2_when_existing_files(
    cmd_trade_with_stub
):
    """When --trace-out points at a directory that already contains
    decision_traces.jsonl and --overwrite is not set, the second run
    must exit 2 and emit `trace_export_error` in the payload — silent
    success on a requested-but-failed export is dangerous."""
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    explicit = cwd / "explicit_dest"
    captured = {}

    from src.fx import cli as cli_mod

    def capturing_print(payload):
        captured.update(payload)

    cli_mod._print = capturing_print  # capture _print payload

    # First run: succeeds and creates files.
    args1 = _build_trade_namespace(trace_out=str(explicit))
    rc1 = cmd(args1, cfg, storage)
    assert rc1 == 0
    captured.clear()

    # Second run with same --trace-out, no --overwrite: must fail.
    args2 = _build_trade_namespace(trace_out=str(explicit))
    rc2 = cmd(args2, cfg, storage)
    assert rc2 == 2, (
        f"second export with no --overwrite should exit 2; got {rc2}"
    )
    assert "trace_export_error" in captured, (
        f"trace_export_error should be in the printed payload; "
        f"keys={list(captured.keys())}"
    )
    assert "FileExistsError" in captured["trace_export_error"]


def test_trace_export_failure_is_recoverable_with_overwrite(
    cmd_trade_with_stub
):
    """When --overwrite is set, a second run to the same dir succeeds."""
    cmd, storage, cwd, cfg = cmd_trade_with_stub
    explicit = cwd / "explicit_dest"

    args1 = _build_trade_namespace(trace_out=str(explicit))
    assert cmd(args1, cfg, storage) == 0

    args2 = _build_trade_namespace(
        trace_out=str(explicit),
        overwrite=True,
    )
    assert cmd(args2, cfg, storage) == 0
    assert (explicit / "decision_traces.jsonl").exists()
