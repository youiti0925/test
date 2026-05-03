"""Command-line entry point.

Usage examples:
  python -m src.fx.cli analyze --symbol USDJPY=X --interval 1h
  python -m src.fx.cli backtest --symbol USDJPY=X --interval 1h --period 180d
  python -m src.fx.cli review --limit 50
  python -m src.fx.cli watch --symbol USDJPY=X --interval 15m --every 900
  python -m src.fx.cli trade --symbol USDJPY=X --dry-run
  python -m src.fx.cli calendar-seed        # create a placeholder events.json
  python -m src.fx.cli evaluate             # score pending predictions
  python -m src.fx.cli postmortem           # learn from wrong predictions
  python -m src.fx.cli lessons --symbol USDJPY=X
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .analyst import Analyst
from .attribution import attribute_move
from .backtest import run_backtest
from .backtest_engine import run_engine_backtest
from .calibration import SWEEP_KEYS, calibrate
from .event_overlay import KEYWORD_MAP, overlay_events
from .external import (
    aggregate_comparisons,
    compare_signals,
    read_external_csv,
)
from .macro import MACRO_SYMBOLS, fetch_macro_snapshot
from .market_timeline import build_timeline
from .strategy_config import StrategyConfig
from .waveform_backtest import waveform_lookup
from .waveform_library import build_library, read_library, write_library
from .waveform_matcher import compute_signature
from .broker import Broker, PaperBroker
from .calendar import (
    Event,
    calendar_freshness,
    events_file_sha8,
    load_events,
    load_events_with_diagnostics,
    seed_defaults,
    upcoming_for_symbol,
)
from .risk_gate import (
    EVENT_WINDOW_POLICY_VERSION,
    infer_event_kind,
)
from .config import Config
from .context import (
    MarketContext,
    detect_volume_spike,
    event_risk_level_from,
    keywords_from_sentiment,
)
from .correlation import build_correlation_snapshot, related_for
from .data import fetch_ohlcv
from .decision_engine import decide as decide_action
from .higher_timeframe import fetch_and_classify as fetch_higher_tf
from .indicators import build_snapshot, technical_signal
from .news import fetch_headlines
from .patterns import analyse as analyse_patterns
from .postmortem import Postmortem
from .prediction import evaluate_prediction, slice_bars_after
from .risk import atr, plan_trade
from .risk_gate import RiskState
from .sentiment import DEFAULT_PATH as SENTIMENT_PATH
from .sentiment import read_for as read_sentiment
from .storage import Storage
from src.sentiment.refresh import refresh as refresh_sentiment_snapshot
from src.sentiment.snapshot import load_snapshot as load_sentiment_snapshot

from src.notify import (
    Notification,
    build_notifier,
    format_lesson,
    format_signal,
    format_summary,
    format_verdict,
)

EVENTS_PATH = Path("data/events.json")


# PR #18 — calendar policy enum recorded in run_metadata.calendar.
# `backtest_warn_but_use` / `paper_warn_but_use`: stale is a warning,
# events are still loaded so historical results stay deterministic.
# `live_require_fresh`: stale fails closed (events disabled here, Risk
# Gate also blocks via require_calendar_fresh). `disabled_*` variants
# capture explicit / structural reasons events were not loaded.
_CALENDAR_POLICY_BACKTEST = "backtest_warn_but_use"
_CALENDAR_POLICY_PAPER    = "paper_warn_but_use"
_CALENDAR_POLICY_LIVE     = "live_require_fresh"
_CALENDAR_POLICY_DISABLED_EXPLICIT   = "disabled_explicit"
_CALENDAR_POLICY_DISABLED_MISSING    = "disabled_missing"
_CALENDAR_POLICY_DISABLED_UNREADABLE = "disabled_unreadable"
_CALENDAR_POLICY_DISABLED_EMPTY      = "disabled_empty"


# PR #19: Literature parameter baseline catalog. Always emitted into
# `run_metadata.parameters` from the research backtest path so a stored
# trace can be reconciled with the catalog version that produced it.
# PR #21: when `--apply-parameter-profile` is also supplied, the
# baseline values are RESOLVED into runtime kwargs and forwarded to
# `run_engine_backtest`. The metadata flips `applied_to_runtime=True`
# and emits applied_sections / skipped_sections / applied_values /
# diff_vs_default. `--parameter-profile X` ALONE remains metadata-only
# (PR #19 invariant preserved). Live `cmd_trade` never calls this
# helper, so live behaviour is unaffected.
def _build_parameters_metadata(
    requested_profile: str | None,
    *,
    apply_runtime: bool = False,
) -> tuple[dict, dict]:
    """Build the run_metadata.parameters block + runtime kwargs.

    Returns (parameters_metadata, runtime_kwargs).

    parameters_metadata is the dict folded into `run_metadata.parameters`.
    runtime_kwargs is what to splat into `run_engine_backtest(**kwargs)`
      to actually apply the profile to the engine. Empty dict when
      `apply_runtime` is False, in which case the engine uses its
      historical defaults (PR #20 main byte-identical).

    `requested_profile` comes from `--parameter-profile`. None +
    apply_runtime=True → ValueError (CLI must reject; this is a
    defensive guard).
    """
    from .parameter_defaults import (
        PARAMETER_BASELINE_V1,
        PARAMETER_BASELINE_ID,
        PARAMETER_BASELINE_VERSION,
        PARAMETER_SWEEP_SPACE_V1,
        baseline_payload_hash,
        KNOWN_PARAMETER_PROFILES,
    )
    from .runtime_parameters import resolve_runtime_parameters
    if (
        requested_profile is not None
        and requested_profile not in KNOWN_PARAMETER_PROFILES
    ):
        raise ValueError(
            f"unknown --parameter-profile: {requested_profile!r}. "
            f"Known profiles: {list(KNOWN_PARAMETER_PROFILES)}"
        )
    if apply_runtime and requested_profile is None:
        raise ValueError(
            "--apply-parameter-profile requires --parameter-profile to "
            "be set as well"
        )
    runtime_kwargs, runtime_audit = resolve_runtime_parameters(
        profile_baseline=PARAMETER_BASELINE_V1 if apply_runtime else None,
        apply_runtime=apply_runtime,
    )
    base = {
        "parameter_profile": requested_profile,
        "runtime_profile": (
            requested_profile if apply_runtime else "current_runtime"
        ),
        "baseline_id": PARAMETER_BASELINE_ID,
        "baseline_version": PARAMETER_BASELINE_VERSION,
        "baseline_payload_hash": baseline_payload_hash(PARAMETER_BASELINE_V1),
        "baseline_reference": PARAMETER_BASELINE_ID,
        "baseline_values": dict(PARAMETER_BASELINE_V1),
        "sweep_space_reference": PARAMETER_SWEEP_SPACE_V1.get("sweep_space_id"),
    }
    base.update(runtime_audit)
    return base, runtime_kwargs


def _load_events_for_run(
    *,
    events_path: Path,
    no_events: bool,
    policy: str,
    test_range: tuple | None = None,
) -> tuple[tuple[Event, ...], dict, list[str]]:
    """Load events for a research backtest / analyze / timeline run.

    PR #18. The freshness signal goes into the metadata + warnings, but
    `backtest_warn_but_use` deliberately uses events anyway so historical
    backtests are deterministic against the events.json content (not the
    file mtime). Returns:
        (events_tuple, calendar_metadata_dict, warning_strings)
    `calendar_metadata_dict` is the run_metadata.calendar block emission;
    callers should attach it to RunMetadata via the engine kwarg.

    `policy` is one of the `_CALENDAR_POLICY_*` constants. `test_range`
    is `(start, end)` of the run window (used to compute
    `test_range_within_coverage`). Pass None to skip that check.
    """
    warnings: list[str] = []

    def _build_metadata(
        *,
        events_loaded: bool,
        events: tuple[Event, ...],
        diagnostics,
        freshness,
        effective_policy: str,
        event_high_enabled: bool,
    ) -> dict:
        coverage_start = (
            min(e.when for e in events).isoformat() if events else None
        )
        coverage_end = (
            max(e.when for e in events).isoformat() if events else None
        )
        in_test_range = None
        within_coverage: bool | None = None
        if events and test_range is not None:
            ts_start, ts_end = test_range
            in_test_range = sum(
                1 for e in events
                if (e.impact or "medium").lower() == "high"
                and ts_start <= e.when <= ts_end
            )
            within_coverage = (
                min(e.when for e in events) <= ts_start
                and max(e.when for e in events) >= ts_end
            )
        return {
            "events_path": str(events_path),
            "events_path_basename": Path(events_path).name,
            "events_loaded": events_loaded,
            "events_count": len(events),
            "events_loaded_count": (
                diagnostics.loaded_count if diagnostics else 0
            ),
            "events_dropped_count": (
                diagnostics.dropped_count if diagnostics else 0
            ),
            "events_dropped_examples": (
                list(diagnostics.dropped_examples) if diagnostics else []
            ),
            "events_parse_errors": (
                list(diagnostics.parse_errors) if diagnostics else []
            ),
            "events_coverage_start": coverage_start,
            "events_coverage_end": coverage_end,
            "events_in_test_range_high_count": in_test_range,
            "test_range_within_coverage": within_coverage,
            "calendar_freshness_status": (
                freshness.status if freshness else None
            ),
            "calendar_freshness_age_hours": (
                freshness.age_hours if freshness else None
            ),
            "calendar_max_age_hours": (
                freshness.max_age_hours if freshness else None
            ),
            "calendar_policy": effective_policy,
            "event_high_enabled": event_high_enabled,
            "event_window_policy_version": EVENT_WINDOW_POLICY_VERSION,
            "events_file_sha8": events_file_sha8(events_path),
            "warnings": list(warnings),
        }

    if no_events:
        warnings.append("calendar disabled by --no-events")
        return (), _build_metadata(
            events_loaded=False, events=(), diagnostics=None,
            freshness=None,
            effective_policy=_CALENDAR_POLICY_DISABLED_EXPLICIT,
            event_high_enabled=False,
        ), warnings

    freshness = calendar_freshness(events_path)

    # Structural failure paths — independent of policy choice.
    if freshness.status == "missing":
        warnings.append(
            f"events.json not found at {events_path}; running without events"
        )
        return (), _build_metadata(
            events_loaded=False, events=(), diagnostics=None,
            freshness=freshness,
            effective_policy=_CALENDAR_POLICY_DISABLED_MISSING,
            event_high_enabled=False,
        ), warnings
    if freshness.status == "unreadable":
        warnings.append(
            f"events.json unreadable: {freshness.detail}; "
            "running without events"
        )
        return (), _build_metadata(
            events_loaded=False, events=(), diagnostics=None,
            freshness=freshness,
            effective_policy=_CALENDAR_POLICY_DISABLED_UNREADABLE,
            event_high_enabled=False,
        ), warnings

    events_list, diagnostics = load_events_with_diagnostics(events_path)
    if diagnostics.dropped_count > 0:
        warnings.append(
            f"events.json: {diagnostics.dropped_count} entries dropped "
            f"during parse (see events_dropped_examples)"
        )
    if not events_list:
        warnings.append("events.json contains zero events")
        return (), _build_metadata(
            events_loaded=True, events=(), diagnostics=diagnostics,
            freshness=freshness,
            effective_policy=_CALENDAR_POLICY_DISABLED_EMPTY,
            event_high_enabled=False,
        ), warnings

    # Live policy: stale fails closed. Research policies: stale is a
    # warning, events are still loaded → backtest stays deterministic
    # vs the events.json content.
    if freshness.status == "stale" and policy == _CALENDAR_POLICY_LIVE:
        warnings.append(
            f"calendar stale ({freshness.age_hours:.1f}h > "
            f"{freshness.max_age_hours:.0f}h) and policy={policy} → "
            "events disabled for this run"
        )
        return (), _build_metadata(
            events_loaded=False, events=(), diagnostics=diagnostics,
            freshness=freshness, effective_policy=policy,
            event_high_enabled=False,
        ), warnings

    if freshness.status == "stale":
        warnings.append(
            f"calendar stale ({freshness.age_hours:.1f}h > "
            f"{freshness.max_age_hours:.0f}h) — using events anyway "
            f"(policy={policy})"
        )

    events_tuple = tuple(events_list)
    return events_tuple, _build_metadata(
        events_loaded=True, events=events_tuple, diagnostics=diagnostics,
        freshness=freshness, effective_policy=policy,
        event_high_enabled=True,
    ), warnings


def _notify(storage: Storage, kind: str, text: str, min_confidence: float = 0.0) -> None:
    """Send `text` to every active subscriber that opted into `kind`.

    Failures are silent — the analysis path must never crash because
    LINE is down.
    """
    try:
        notifier = build_notifier()
        if notifier.name == "null":
            return
        subs = storage.active_subscribers(kind=kind)
        recipients = [
            s["user_id"]
            for s in subs
            if (s.get("min_confidence") or 0.0) <= min_confidence
        ]
        if not recipients and notifier.name == "log":
            recipients = ["broadcast"]
        if not recipients:
            return
        notifier.push(
            Notification(text=text, kind=kind, recipients=tuple(recipients))
        )
    except Exception:  # noqa: BLE001
        pass


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def _gather_inputs(args, cfg: Config, storage: Storage):
    """Common input fetch for analyze / trade / watch.

    Builds a MarketContext, runs the fixed-rule Decision Engine, and
    returns everything the callers and notifications need. The LLM is
    consulted but cannot override the engine.

    `args.strategy_config` (if set) is a `StrategyConfig` loaded from
    `--config`. Its risk + waveform thresholds flow into `decide()`
    so live tuning happens via the config file rather than code edits.
    """
    strategy: StrategyConfig | None = getattr(args, "strategy_config", None)
    df = fetch_ohlcv(args.symbol, interval=args.interval, period=args.period)
    snap = build_snapshot(args.symbol, df)
    tech = technical_signal(snap)
    pattern = analyse_patterns(df)
    atr_value = float(atr(df).iloc[-1])

    headlines = []
    if not args.no_news:
        headlines = fetch_headlines(args.symbol, limit=args.news_limit)

    correlation = None
    if not args.no_correlation:
        others = related_for(args.symbol)
        if others:
            try:
                correlation = build_correlation_snapshot(
                    args.symbol, others=others, interval="1d", period="60d"
                )
            except Exception as e:  # noqa: BLE001
                print(f"[warn] correlation fetch failed: {e}", file=sys.stderr)

    events = []
    cal_health = None
    if not args.no_events:
        cal_health = calendar_freshness(EVENTS_PATH)
        if cal_health.is_fresh:
            all_events = load_events(EVENTS_PATH)
            events = upcoming_for_symbol(
                all_events, args.symbol, within_hours=args.event_window_hours
            )
        else:
            print(
                f"[warn] calendar {cal_health.status}: "
                f"{cal_health.detail or 'unhealthy'}",
                file=sys.stderr,
            )

    past_lessons = []
    if not args.no_lessons:
        past_lessons = storage.relevant_postmortems(
            args.symbol, limit=args.lesson_limit
        )

    sentiment = None
    if not args.no_sentiment:
        sentiment = read_sentiment(args.symbol, max_age_s=12 * 3600)

    higher_tf = "UNKNOWN"
    if not getattr(args, "no_higher_tf", False):
        try:
            higher_tf = fetch_higher_tf(args.symbol, args.interval)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] higher-timeframe fetch failed: {e}", file=sys.stderr)

    # Risk-reward + stop distance derive from the StrategyConfig's
    # ratios (defaults: stop=2*ATR, TP=3*ATR → RR=1.5).
    if strategy is not None and strategy.risk.stop_atr > 0:
        rr_ratio = strategy.risk.take_profit_atr / strategy.risk.stop_atr
        stop_mult = strategy.risk.stop_atr
    else:
        rr_ratio = 1.5
        stop_mult = 2.0
    risk_reward = rr_ratio if atr_value > 0 else None
    atr_stop_distance = stop_mult * atr_value if atr_value > 0 else None

    context = MarketContext(
        symbol=args.symbol,
        interval=args.interval,
        snapshot=snap,
        pattern=pattern,
        higher_timeframe_trend=higher_tf,
        spread_state="UNKNOWN",  # yfinance has no bid/ask; left UNKNOWN
        spread_pct=None,
        event_risk_level=event_risk_level_from([e.to_dict() for e in events]),
        economic_events_nearby=[e.to_dict() for e in events],
        sentiment_score=(sentiment or {}).get("sentiment_score"),
        sentiment_velocity=(sentiment or {}).get("sentiment_velocity"),
        sentiment_volume_spike=detect_volume_spike(sentiment),
        top_sentiment_keywords=keywords_from_sentiment(sentiment),
        risk_reward=risk_reward,
        atr_stop_distance=atr_stop_distance,
        technical_signal=tech,
    )

    llm_signal = None
    if cfg.anthropic_api_key and not args.no_llm:
        analyst = Analyst(model=cfg.model, effort=cfg.effort)
        llm_signal = analyst.analyze(
            context.to_dict(),
            headlines=headlines,
            correlation=correlation,
            upcoming_events=events,
            past_postmortems=past_lessons,
            sentiment_snapshot=sentiment,
        )

    # `require_calendar_fresh` / `require_spread` are flipped on by trade callers
    # at decision time (see cmd_trade); analyze() leaves them False so research
    # workflows still produce useful output when feeds are stale.
    require_calendar_fresh = bool(getattr(args, "require_fresh_calendar", False))
    require_spread = bool(getattr(args, "require_spread", False))

    risk_state = RiskState(
        df=df,
        events=tuple(events),
        spread_pct=getattr(args, "_spread_pct_override", None),
        sentiment_snapshot=sentiment,
        calendar_freshness=cal_health,
        require_calendar_fresh=require_calendar_fresh,
        require_spread=require_spread,
    )

    # Decision Engine thresholds come from StrategyConfig when provided
    # — the function defaults are kept in sync but explicit is better
    # than implicit when the user has dialled in custom values.
    decide_kwargs = {}
    if strategy is not None:
        decide_kwargs["min_risk_reward"] = strategy.risk.min_risk_reward
        # waveform.min_confidence doubles as the LLM/min_confidence
        # floor in the engine — keep one knob, one effect.
        decide_kwargs["min_confidence"] = strategy.waveform.min_confidence

    decision = decide_action(
        technical_signal=tech,
        pattern=pattern,
        higher_timeframe_trend=higher_tf,
        risk_reward=risk_reward,
        risk_state=risk_state,
        llm_signal=llm_signal,
        **decide_kwargs,
    )

    return {
        "df": df,
        "snapshot": snap,
        "context": context,
        "pattern": pattern,
        "technical": tech,
        "headlines": headlines,
        "correlation": correlation,
        "events": events,
        "calendar_freshness": cal_health,
        "past_lessons": past_lessons,
        "sentiment": sentiment,
        "higher_tf": higher_tf,
        "llm": llm_signal,
        "decision": decision,
        "atr": atr_value,
    }


def cmd_analyze(args, cfg: Config, storage: Storage) -> int:
    ctx = _gather_inputs(args, cfg, storage)
    snap, tech, llm, decision = (
        ctx["snapshot"],
        ctx["technical"],
        ctx["llm"],
        ctx["decision"],
    )
    analysis_id = storage.save_analysis(
        symbol=args.symbol,
        snapshot=snap.to_dict(),
        technical_signal=tech,
        final_action=decision.action,
        llm_action=llm.action if llm else None,
        llm_confidence=llm.confidence if llm else None,
        llm_reason=llm.reason if llm else None,
    )

    prediction_id = None
    if llm is not None:
        # Spec §13: distinguish what the LLM advised, what the engine
        # decided, and what was actually executed. analyze() doesn't place
        # orders, so executed_action mirrors the engine's final action.
        ctx_obj = ctx["context"]
        events_json = json.dumps(
            [e.to_dict() for e in ctx["events"]], default=str
        ) if ctx["events"] else None
        prediction_id = storage.save_prediction(
            analysis_id=analysis_id,
            symbol=args.symbol,
            interval=args.interval,
            entry_price=snap.last_close,
            action=llm.action,
            confidence=llm.confidence,
            reason=llm.reason,
            expected_direction=llm.expected_direction,
            expected_magnitude_pct=llm.expected_magnitude_pct,
            horizon_bars=llm.horizon_bars,
            invalidation_price=llm.invalidation_price,
            final_decision_action=decision.action,
            executed_action=decision.action,
            blocked_by=",".join(decision.blocked_by) if decision.blocked_by else None,
            final_reason=decision.reason,
            rule_chain=",".join(decision.rule_chain) if decision.rule_chain else None,
            risk_reward=ctx_obj.risk_reward,
            detected_pattern=(
                ctx["pattern"].detected_pattern if ctx["pattern"] else None
            ),
            trend_state=(
                ctx["pattern"].trend_state.value if ctx["pattern"] else None
            ),
            higher_timeframe_trend=ctx["higher_tf"],
            event_risk_level=ctx_obj.event_risk_level,
            economic_events_nearby=events_json,
            sentiment_score=ctx_obj.sentiment_score,
            sentiment_volume_spike=(
                1 if ctx_obj.sentiment_volume_spike else 0
            ),
            spread_at_entry=ctx_obj.spread_pct,
        )

    # Push notification when LLM produced a directional, confident signal.
    if llm and decision.action in ("BUY", "SELL"):
        _notify(
            storage,
            kind="signal",
            min_confidence=llm.confidence,
            text=format_signal(
                symbol=args.symbol,
                action=llm.action,
                confidence=llm.confidence,
                expected_direction=llm.expected_direction,
                expected_magnitude_pct=llm.expected_magnitude_pct,
                horizon_bars=llm.horizon_bars,
                invalidation_price=llm.invalidation_price,
                reason=llm.reason,
                analysis_id=analysis_id,
            ),
        )

    _print(
        {
            "analysis_id": analysis_id,
            "prediction_id": prediction_id,
            "snapshot": snap.to_dict(),
            "technical_signal": tech,
            "atr_14": round(ctx["atr"], 6),
            "correlation": (
                ctx["correlation"].to_dict() if ctx["correlation"] else None
            ),
            "upcoming_events": [e.to_dict() for e in ctx["events"]],
            "headlines": [h.to_dict() for h in ctx["headlines"]],
            "past_lessons_used": len(ctx["past_lessons"]),
            "llm": (
                {
                    "action": llm.action,
                    "confidence": llm.confidence,
                    "reason": llm.reason,
                    "key_risks": llm.key_risks,
                    "expected_direction": llm.expected_direction,
                    "expected_magnitude_pct": llm.expected_magnitude_pct,
                    "horizon_bars": llm.horizon_bars,
                    "invalidation_price": llm.invalidation_price,
                }
                if llm
                else None
            ),
            "decision": {
                "action": decision.action,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
        }
    )
    return 0


def cmd_trade(args, cfg: Config, storage: Storage) -> int:
    """Run an analysis and, if the decision is directional, place a trade.

    Default broker is PaperBroker. Real-broker support is explicitly opt-in
    via --broker oanda (requires environment guards, see src/fx/oanda.py).

    Live brokers (oanda) trigger two extra guards on top of the analyst path:
      * `require_calendar_fresh`: events.json must be fresh (spec §11)
      * `require_spread`: broker must return a Bid/Ask quote (spec §12)
    Both flow through Risk Gate, so the Decision Engine returns HOLD when
    either is missing — no order is placed.
    """
    is_live_broker = args.broker == "oanda"

    # Build broker first so we can query its quote BEFORE running the
    # Decision Engine. Quote is what produces spread_pct.
    broker: Broker = _build_broker(args)
    quote = None
    spread_pct = None
    if is_live_broker:
        try:
            quote = broker.quote(args.symbol)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] broker.quote failed: {e}", file=sys.stderr)
        if quote is not None:
            spread_pct = quote.spread_pct

    # Make the gathering step aware of live-broker enforcement. We reuse
    # _gather_inputs by stamping the args with the override fields it
    # checks via getattr — keeps backward compat for analyze/watch.
    args._spread_pct_override = spread_pct
    args.require_fresh_calendar = is_live_broker
    args.require_spread = is_live_broker

    ctx = _gather_inputs(args, cfg, storage)
    snap, decision, atr_value = ctx["snapshot"], ctx["decision"], ctx["atr"]

    # Live trace plumbing — single helper that the three exit branches
    # (HOLD / dry-run / live order) all call so every cmd_trade invocation
    # produces exactly one BarDecisionTrace. When --trace-out / --trace-out-default
    # is given but export fails, we surface the error to the caller via
    # `trace_export_error` and the exit branches return code 2 — silent
    # success on a requested-but-failed export would be dangerous.
    trace_export_paths: dict[str, str] | None = None
    trace_export_error: str | None = None

    def _emit_live_trace(*, order_placed: bool, fill_obj: object | None) -> None:
        nonlocal trace_export_paths, trace_export_error
        out_dir = _resolve_live_trace_out_dir(args, ctx)
        if out_dir is None:
            return
        try:
            trace_export_paths = _write_live_trace(
                ctx=ctx,
                args=args,
                broker_label=args.broker,
                dry_run=bool(getattr(args, "dry_run", False)),
                order_placed=order_placed,
                fill_obj=fill_obj,
                out_dir=out_dir,
                overwrite=bool(getattr(args, "overwrite", False)),
            )
        except (FileExistsError, ValueError, OSError) as exc:
            trace_export_error = f"{type(exc).__name__}: {exc}"
            print(
                f"[error] live trace export failed: {trace_export_error}",
                file=sys.stderr,
            )

    if decision.action == "HOLD":
        _emit_live_trace(order_placed=False, fill_obj=None)
        payload_hold = {
            "action": "HOLD",
            "reason": decision.reason,
            "blocked_by": list(decision.blocked_by),
            "spread_pct": spread_pct,
            "calendar": (
                ctx["calendar_freshness"].to_dict()
                if ctx.get("calendar_freshness") else None
            ),
        }
        if trace_export_paths:
            payload_hold["trace_export"] = trace_export_paths
        if trace_export_error:
            payload_hold["trace_export_error"] = trace_export_error
        _print(payload_hold)
        return 2 if trace_export_error else 0

    if atr_value <= 0:
        print("[warn] ATR is non-positive; cannot size trade.", file=sys.stderr)
        return 1

    # Strategy config can override the per-broker stop/TP/risk_pct
    # defaults if the user supplied --config and didn't pass an
    # explicit --stop-atr/--tp-atr/--risk-pct on the command line.
    s_cfg = getattr(args, "strategy_config", None)
    stop_atr = args.stop_atr if args.stop_atr is not None else (
        s_cfg.risk.stop_atr if s_cfg else 2.0
    )
    tp_atr = args.tp_atr if args.tp_atr is not None else (
        s_cfg.risk.take_profit_atr if s_cfg else 3.0
    )
    risk_pct = args.risk_pct if args.risk_pct is not None else (
        s_cfg.risk.risk_pct if s_cfg else 0.01
    )

    plan = plan_trade(
        side=decision.action,
        entry=snap.last_close,
        atr_value=atr_value,
        capital=args.capital,
        stop_mult=stop_atr,
        tp_mult=tp_atr,
        risk_pct=risk_pct,
    )

    if args.dry_run:
        _emit_live_trace(order_placed=False, fill_obj=None)
        payload_dry = {
            "mode": "dry-run",
            "symbol": args.symbol,
            "decision": {
                "action": decision.action,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
            "plan": plan.to_dict(),
            "spread_pct": spread_pct,
            "would_place_order": True,
        }
        if trace_export_paths:
            payload_dry["trace_export"] = trace_export_paths
        if trace_export_error:
            payload_dry["trace_export_error"] = trace_export_error
        _print(payload_dry)
        return 2 if trace_export_error else 0

    pos = broker.place_order(
        symbol=args.symbol,
        side=plan.side,
        size=plan.size,
        price=plan.entry,
        stop=plan.stop,
        take_profit=plan.take_profit,
    )

    # Persist the trade row with the full execution audit. The fill
    # object is None for old broker implementations that haven't been
    # upgraded — we still save the trade, just without the audit.
    fill = broker.last_execution_fill()
    # Emit the live trace BEFORE persisting the trade so save_trade() can
    # record the resolved decision_traces.jsonl path. If trace export is
    # not requested (no --trace-out / --trace-out-default), trace_export_paths
    # stays None and the trade row gets NULL for trace_jsonl_path. If trace
    # export is requested but fails, we still persist the trade (the order
    # was already placed by the broker) with NULL path, and exit code 2 is
    # surfaced via trace_export_error below.
    _emit_live_trace(order_placed=True, fill_obj=fill)
    trace_jsonl_path = (
        trace_export_paths.get("decision_traces") if trace_export_paths else None
    )
    trade_id = storage.save_trade(
        symbol=args.symbol,
        side=plan.side,
        entry=fill.actual_fill_price if fill else plan.entry,
        size=plan.size,
        opened_at=fill.fill_time if fill else pos.opened_at,
        analysis_id=None,
        note=decision.reason,
        broker=args.broker,
        expected_entry_price=fill.expected_entry_price if fill else plan.entry,
        actual_fill_price=fill.actual_fill_price if fill else None,
        slippage=fill.slippage if fill else None,
        bid_at_entry=fill.bid if fill else None,
        ask_at_entry=fill.ask if fill else None,
        spread_pct_at_entry=fill.spread_pct if fill else None,
        broker_order_id=fill.broker_order_id if fill else None,
        order_request_time=fill.order_request_time if fill else None,
        order_response_time=fill.order_response_time if fill else None,
        fill_time=fill.fill_time if fill else None,
        execution_latency_ms=fill.execution_latency_ms if fill else None,
        trace_jsonl_path=trace_jsonl_path,
    )
    payload_live = {
        "mode": "live",
        "broker": args.broker,
        "trade_id": trade_id,
        "position": {
            "id": pos.id,
            "symbol": pos.symbol,
            "side": pos.side,
            "entry": pos.entry,
            "size": pos.size,
            "stop": pos.stop,
            "take_profit": pos.take_profit,
        },
        "fill": fill.to_dict() if fill else None,
        "plan": plan.to_dict(),
    }
    if trace_export_paths:
        payload_live["trace_export"] = trace_export_paths
    if trace_export_error:
        payload_live["trace_export_error"] = trace_export_error
    _print(payload_live)
    return 2 if trace_export_error else 0


def _resolve_live_trace_out_dir(args, ctx) -> "Path | None":
    """Decide the live trace output directory based on CLI flags.

    --trace-out wins over --trace-out-default. When --trace-out-default is
    on but the run_id helper has not yet been generated, we mint one here
    so the directory name is deterministic for the rest of the call.
    """
    if getattr(args, "trace_out", None):
        return Path(args.trace_out)
    if getattr(args, "trace_out_default", False):
        from .decision_trace_build import build_run_id
        if "_live_run_id" not in ctx:
            ctx["_live_run_id"] = build_run_id(args.symbol)
        return Path("runs") / "live" / ctx["_live_run_id"]
    return None


def _write_live_trace(
    *,
    ctx: dict,
    args,
    broker_label: str,
    dry_run: bool,
    order_placed: bool,
    fill_obj,
    out_dir: "Path",
    overwrite: bool,
) -> dict[str, str]:
    """Write run_metadata.json + decision_traces.jsonl for one cmd_trade run.

    `decision_traces.jsonl` carries exactly one line (one bar) — live
    invocations are single-bar by definition. Returns the resolved file
    paths so the CLI can echo them in the JSON payload.
    """
    from .decision_trace_build import (
        build_live_decision_trace,
        build_live_run_metadata,
        build_run_id,
    )

    run_id = ctx.get("_live_run_id") or build_run_id(args.symbol)
    df = ctx["df"]
    ts = df.index[-1]
    snap = ctx["snapshot"]
    pattern = ctx["pattern"]
    technical_action = ctx["technical"]
    higher_tf = ctx["higher_tf"]
    decision = ctx["decision"]
    atr_value = ctx["atr"]

    # Re-derive the same RiskState the engine saw — _gather_inputs builds
    # it inline and discards it; re-build with the same inputs to feed
    # rule_checks_full. This matches the values used by decide_action.
    from datetime import timezone as _tz
    from .risk_gate import RiskState as _RiskState
    cal_health = ctx.get("calendar_freshness")
    risk_state = _RiskState(
        df=df,
        events=tuple(ctx.get("events", [])),
        spread_pct=getattr(args, "_spread_pct_override", None),
        sentiment_snapshot=ctx.get("sentiment"),
        calendar_freshness=cal_health,
        require_calendar_fresh=bool(getattr(args, "require_fresh_calendar", False)),
        require_spread=bool(getattr(args, "require_spread", False)),
        now=ts.to_pydatetime() if ts.tzinfo else ts.tz_localize(_tz.utc).to_pydatetime(),
    )

    risk_reward = ctx.get("context").risk_reward if ctx.get("context") else None
    if risk_reward is None:
        risk_reward = 0.0

    trace = build_live_decision_trace(
        run_id=run_id,
        df=df,
        ts=ts,
        symbol=args.symbol,
        interval=args.interval,
        snap=snap,
        atr_value=atr_value if atr_value and atr_value > 0 else None,
        technical_action=technical_action,
        pattern=pattern,
        higher_tf=higher_tf,
        use_higher_tf=not bool(getattr(args, "no_higher_tf", False)),
        risk_state=risk_state,
        llm_signal=ctx.get("llm"),
        decision=decision,
        risk_reward=float(risk_reward),
        broker_label=broker_label,
        dry_run=dry_run,
        order_placed=order_placed,
        fill=fill_obj,
    )

    run_metadata = build_live_run_metadata(
        run_id=run_id,
        symbol=args.symbol,
        interval=args.interval,
        broker_label=broker_label,
        dry_run=dry_run,
        config_payload={
            "broker": broker_label,
            "dry_run": dry_run,
            "interval": args.interval,
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "run_metadata.json"
    traces_path = out_dir / "decision_traces.jsonl"

    if not overwrite:
        existing = [p for p in (metadata_path, traces_path) if p.exists()]
        if existing:
            names = ", ".join(p.name for p in existing)
            raise FileExistsError(
                f"_write_live_trace: refusing to overwrite existing file(s) "
                f"in {out_dir}: {names}. Pass --overwrite to replace."
            )

    metadata_path.write_text(
        json.dumps(run_metadata.to_dict(), ensure_ascii=False,
                   default=str, indent=2),
        encoding="utf-8",
    )
    with traces_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(trace.to_dict(), ensure_ascii=False, default=str))
        fp.write("\n")

    return {
        "run_metadata": str(metadata_path.resolve()),
        "decision_traces": str(traces_path.resolve()),
    }


def _build_broker(args) -> Broker:
    if args.broker == "paper":
        return PaperBroker(initial_cash=args.capital)
    if args.broker == "oanda":
        from .oanda import OANDABroker, OANDAConfig  # lazy import

        cfg = OANDAConfig.from_env()
        return OANDABroker(cfg, confirm_demo=args.confirm_demo)
    raise ValueError(f"Unknown broker: {args.broker}")


def cmd_backtest(args, cfg: Config, storage: Storage) -> int:
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )
    result = run_backtest(df, symbol=args.symbol, warmup=args.warmup)
    metrics = result.metrics()
    start_date = str(df.index[0])
    end_date = str(df.index[-1])
    storage.save_backtest(
        symbol=args.symbol,
        interval=args.interval,
        start_date=start_date,
        end_date=end_date,
        strategy="technical_consensus",
        metrics=metrics,
    )
    _print(
        {
            "symbol": args.symbol,
            "interval": args.interval,
            "period": f"{start_date} -> {end_date}",
            "bars": len(df),
            "metrics": metrics,
            "last_5_trades": [
                {
                    "side": t.side,
                    "entry": round(t.entry, 6),
                    "exit": round(t.exit, 6),
                    "return_pct": round(t.return_pct, 3),
                    "entry_ts": str(t.entry_ts),
                    "exit_ts": str(t.exit_ts),
                }
                for t in result.trades[-5:]
            ],
        }
    )
    return 0


def cmd_backtest_engine(args, cfg: Config, storage: Storage) -> int:
    """Backtest using the full Decision Engine + Risk Gate chain (spec §4)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )

    # PR #17: optional historical context — feeds long_term_trend_slice
    # only, NOT the judgement path. Fetched as a separate yfinance call
    # ending strictly before df.index.min(). The 1h-specific 730d total-
    # span limit is enforced here because the user's combined window
    # (context + test) must fit; longer intervals (1d / 1wk) have no
    # such limit and we don't blanket-apply it.
    df_context = None
    context_days = int(getattr(args, "context_days", 0) or 0)
    if context_days > 0:
        if len(df) == 0:
            print(
                "[error] --context-days given but the test fetch is empty",
                file=sys.stderr,
            )
            return 2
        test_start_ts = pd.Timestamp(df.index.min())
        # Combined-window guard for 1h: yfinance hard-stops at ~730 days.
        if args.interval == "1h":
            test_span_days = (df.index.max() - df.index.min()).days + 1
            if context_days + test_span_days > 730:
                print(
                    f"[error] --context-days {context_days} + test span "
                    f"{test_span_days}d > 730d (yfinance 1h hard limit). "
                    "Reduce --context-days or shorten test window.",
                    file=sys.stderr,
                )
                return 2
        ctx_end = test_start_ts.strftime("%Y-%m-%d")
        ctx_start = (
            test_start_ts - pd.Timedelta(days=context_days)
        ).strftime("%Y-%m-%d")
        try:
            df_context = fetch_ohlcv(
                args.symbol,
                interval=args.interval,
                period=None,
                start=ctx_start,
                end=ctx_end,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[error] context fetch failed: {e}",
                file=sys.stderr,
            )
            return 2
        # Drop any rows that landed at or after test_start (yfinance can
        # over-deliver near the boundary). Strict-less-than is rechecked
        # by the engine, but we trim defensively here so the engine
        # doesn't reject the run on what is really a data-source quirk.
        if len(df_context) > 0:
            df_context = df_context[df_context.index < df.index.min()]
        if len(df_context) == 0:
            print(
                "[warn] context fetch returned 0 bars after the strict-"
                "before filter — proceeding without context",
                file=sys.stderr,
            )
            df_context = None

    # PR #18 — backtest deterministic policy: stale events.json is a
    # WARNING, not a reason to drop events. Same events.json content +
    # different mtime must yield the same trades. Live path uses
    # `live_require_fresh` separately.
    test_range_for_calendar = (
        (df.index.min().to_pydatetime(), df.index.max().to_pydatetime())
        if len(df) > 0 else None
    )
    events, calendar_metadata, calendar_warnings = _load_events_for_run(
        events_path=EVENTS_PATH,
        no_events=args.no_events,
        policy=_CALENDAR_POLICY_BACKTEST,
        test_range=test_range_for_calendar,
    )
    for w in calendar_warnings:
        print(f"[warn] {w}", file=sys.stderr)

    # Macro context (DXY / US10Y / VIX / indices). Fetched once per run
    # and attached to the engine; build_full_trace samples each slot
    # point-in-time per bar via Series.asof. Failure is non-fatal —
    # backtest still produces the trace, macro_context just stays None.
    macro = None
    if not getattr(args, "no_macro", False):
        try:
            macro_period = getattr(args, "macro_period", None) or "2y"
            macro = fetch_macro_snapshot(
                df.index, interval="1d", period=macro_period,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[warn] macro fetch failed: {e} — trace will have "
                "macro_context=null",
                file=sys.stderr,
            )

    # Waveform library (PR #15): user-built JSONL of past WaveformSample.
    # Read failure is fatal here — if you asked for a library and we
    # cannot load it, you'd unknowingly run a backtest without it.
    # Decision logic is unchanged either way; this is observability only.
    waveform_library = None
    waveform_library_id = None
    if getattr(args, "waveform_library", None):
        from .waveform_library import read_library
        from .decision_trace_build import compute_library_id
        lib_path = Path(args.waveform_library)
        try:
            waveform_library = read_library(lib_path)
        except Exception as e:  # noqa: BLE001
            print(
                f"[error] failed to read waveform library {lib_path}: {e}",
                file=sys.stderr,
            )
            return 2
        if not waveform_library:
            print(
                f"[error] waveform library {lib_path} is empty",
                file=sys.stderr,
            )
            return 2
        waveform_library_id = compute_library_id(
            waveform_library, str(lib_path),
        )

    # PR #19: build the literature_baseline_v1 metadata block. This is
    # ALWAYS emitted (regardless of --parameter-profile) so a stored
    # trace records which catalog version was current at run time.
    # PR #21: when `--apply-parameter-profile` is also supplied, the
    # baseline values are RESOLVED into runtime kwargs that flow to
    # `run_engine_backtest`. Without that flag, runtime is unchanged
    # (PR #19 invariant preserved).
    try:
        parameters_metadata, runtime_kwargs = _build_parameters_metadata(
            getattr(args, "parameter_profile", None),
            apply_runtime=bool(getattr(args, "apply_parameter_profile", False)),
        )
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    # When PR #21 runtime apply is on, the resolved profile values
    # override stop_atr_mult / tp_atr_mult / max_holding_bars. CLI args
    # (--stop-atr etc.) take precedence ONLY when the user did NOT
    # request profile apply — otherwise the profile wins.
    _stop_atr_mult = (
        runtime_kwargs.get("stop_atr_mult", args.stop_atr)
        if runtime_kwargs else args.stop_atr
    )
    _tp_atr_mult = (
        runtime_kwargs.get("tp_atr_mult", args.tp_atr)
        if runtime_kwargs else args.tp_atr
    )
    _max_holding_bars = (
        runtime_kwargs.get("max_holding_bars", args.max_holding_bars)
        if runtime_kwargs else args.max_holding_bars
    )

    # v2 lower-TF fetch (opt-in via --lower-tf). Failure is non-fatal —
    # engine treats unavailable lower-TF as `not_attached` and emits
    # the unavailable_reason in trace.
    df_lower_tf_for_v2 = None
    lower_tf_interval_for_v2 = getattr(args, "lower_tf", None)
    if lower_tf_interval_for_v2 and len(df) > 0:
        try:
            df_lower_tf_for_v2 = fetch_ohlcv(
                args.symbol,
                interval=lower_tf_interval_for_v2,
                period=None,
                start=df.index.min().strftime("%Y-%m-%d"),
                end=(df.index.max() + pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                ),
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"[warn] lower-TF fetch failed: {e} — v2 lower_tf_trigger "
                "will report unavailable_reason",
                file=sys.stderr,
            )
            df_lower_tf_for_v2 = None

    result = run_engine_backtest(
        df,
        symbol=args.symbol,
        interval=args.interval,
        warmup=args.warmup,
        stop_atr_mult=_stop_atr_mult,
        tp_atr_mult=_tp_atr_mult,
        max_holding_bars=_max_holding_bars,
        events=events,
        use_higher_tf=not args.no_higher_tf,
        macro=macro,
        waveform_library=waveform_library,
        waveform_library_id=waveform_library_id,
        df_context=df_context,
        context_days=context_days,
        calendar=calendar_metadata,
        parameters=parameters_metadata,
        # PR #21 indicator-period overrides (None when apply_runtime=False
        # → engine uses its existing defaults)
        rsi_period=runtime_kwargs.get("rsi_period"),
        rsi_overbought=runtime_kwargs.get("rsi_overbought"),
        rsi_oversold=runtime_kwargs.get("rsi_oversold"),
        macd_fast=runtime_kwargs.get("macd_fast"),
        macd_slow=runtime_kwargs.get("macd_slow"),
        macd_signal_period=runtime_kwargs.get("macd_signal"),
        bb_period=runtime_kwargs.get("bb_period"),
        bb_std=runtime_kwargs.get("bb_std"),
        atr_period=runtime_kwargs.get("atr_period"),
        decision_profile=getattr(args, "decision_profile", "current_runtime"),
        royal_road_mode=getattr(args, "royal_road_mode", "balanced"),
        integrated_mode=getattr(args, "integrated_mode", "integrated_balanced"),
        df_lower_tf=df_lower_tf_for_v2,
        lower_tf_interval=lower_tf_interval_for_v2,
        stop_mode=getattr(args, "stop_mode", "atr"),
    )
    metrics = result.metrics()
    start_date = str(df.index[0])
    end_date = str(df.index[-1])
    storage.save_backtest(
        symbol=args.symbol,
        interval=args.interval,
        start_date=start_date,
        end_date=end_date,
        strategy="decision_engine",
        metrics=metrics,
    )
    payload = {
        "symbol": args.symbol,
        "interval": args.interval,
        "period": f"{start_date} -> {end_date}",
        "bars": len(df),
        "metrics": metrics,
        "last_5_trades": [
            {
                "side": t.side,
                "entry": round(t.entry, 6),
                "exit": round(t.exit, 6),
                "return_pct": round(t.return_pct, 3),
                "bars_held": t.bars_held,
                "exit_reason": t.exit_reason,
                "entry_ts": str(t.entry_ts),
                "exit_ts": str(t.exit_ts),
            }
            for t in result.trades[-5:]
        ],
    }

    if args.trace_out:
        trace_out_dir = Path(args.trace_out)
    elif args.trace_out_default:
        if result.run_metadata is None:
            print(
                "[error] --trace-out-default requires capture_traces=True "
                "(run_metadata is None)",
                file=sys.stderr,
            )
            return 2
        trace_out_dir = Path("runs") / result.run_metadata.run_id
    else:
        trace_out_dir = None

    if trace_out_dir is not None:
        from .decision_trace_io import export_run
        try:
            paths = export_run(
                result,
                out_dir=trace_out_dir,
                overwrite=args.overwrite,
                gzip=args.gzip,
            )
        except (FileExistsError, ValueError) as exc:
            print(f"[error] trace export failed: {exc}", file=sys.stderr)
            return 2
        payload["trace_export"] = {k: str(v) for k, v in paths.items()}

    _print(payload)
    return 0


def cmd_build_timeline(args, cfg: Config, storage: Storage) -> int:
    """Build a point-in-time market timeline (spec §5)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )

    # PR #18: timeline is a research path — same deterministic policy
    # as backtest-engine so the event columns are reproducible.
    test_range_for_calendar = (
        (df.index.min().to_pydatetime(), df.index.max().to_pydatetime())
        if len(df) > 0 else None
    )
    events, _calendar_metadata, calendar_warnings = _load_events_for_run(
        events_path=EVENTS_PATH,
        no_events=args.no_events,
        policy=_CALENDAR_POLICY_BACKTEST,
        test_range=test_range_for_calendar,
    )
    for w in calendar_warnings:
        print(f"[warn] {w}", file=sys.stderr)

    macro = None
    if not args.no_macro:
        try:
            macro = fetch_macro_snapshot(
                df.index,
                interval="1d",
                period=args.macro_period,
                slots=args.macro_slots if args.macro_slots else None,
            )
            if macro.fetch_errors:
                print(
                    f"[warn] macro fetch errors: {macro.fetch_errors}",
                    file=sys.stderr,
                )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] macro fetch failed entirely: {e}", file=sys.stderr)

    timeline = build_timeline(
        df,
        symbol=args.symbol,
        interval=args.interval,
        warmup=args.warmup,
        events=events,
        macro=macro,
        use_higher_tf=not args.no_higher_tf,
    )

    if args.output:
        out_path = Path(args.output)
        frame = timeline.to_frame()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".parquet":
            frame.to_parquet(out_path)
        else:
            # Default to CSV; nested cols get JSON-stringified for safety
            for col in frame.columns:
                if frame[col].dtype == object:
                    frame[col] = frame[col].apply(
                        lambda x: json.dumps(x, default=str)
                        if isinstance(x, (list, dict)) else x
                    )
            frame.to_csv(out_path)

    summary = {
        "symbol": args.symbol,
        "interval": args.interval,
        "rows": len(timeline),
        "first_ts": (
            str(timeline.rows[0].timestamp) if timeline.rows else None
        ),
        "last_ts": (
            str(timeline.rows[-1].timestamp) if timeline.rows else None
        ),
        "macro_slots": (
            list(macro.series.keys()) if macro else []
        ),
        "macro_errors": (
            macro.fetch_errors if macro else {}
        ),
        "events_used": len(events),
        "output": args.output,
        "preview": [
            timeline.rows[i].to_dict()
            for i in (0, len(timeline) // 2, len(timeline) - 1)
            if 0 <= i < len(timeline)
        ],
    }
    _print(summary)
    return 0


def cmd_event_overlay(args, cfg: Config, storage: Storage) -> int:
    """Aggregate pre/post-event price action (spec §6)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )

    cal_health = calendar_freshness(EVENTS_PATH)
    if not cal_health.is_fresh:
        print(
            f"[warn] calendar {cal_health.status}: "
            f"{cal_health.detail or 'unhealthy'}",
            file=sys.stderr,
        )
    events = load_events(EVENTS_PATH) if EVENTS_PATH.exists() else []
    if not events:
        _print({"error": "no events available", "calendar": cal_health.to_dict()})
        return 1

    result = overlay_events(
        df, events,
        keyword=args.event,
        impact=args.impact,
    )
    _print({
        "symbol": args.symbol,
        "interval": args.interval,
        "filter_keyword": args.event,
        "filter_impact": args.impact,
        "n_events_matched": len(result.rows),
        "aggregate": result.aggregate(),
        "events": [r.to_dict() for r in result.rows[: args.limit]],
    })
    return 0


def cmd_waveform_build_library(args, cfg: Config, storage: Storage) -> int:
    """Build a waveform sample library from a long history (spec §7.2)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
        start=args.start,
        end=args.end,
    )
    samples = build_library(
        df,
        symbol=args.symbol,
        timeframe=args.interval,
        window_bars=args.window_bars,
        step_bars=args.step_bars,
        forward_horizons=tuple(args.horizons),
        normalize=args.normalize,
    )
    out_path = Path(args.output)
    n = write_library(out_path, samples, append=args.append)
    _print({
        "symbol": args.symbol,
        "interval": args.interval,
        "window_bars": args.window_bars,
        "step_bars": args.step_bars,
        "horizons": args.horizons,
        "samples_written": n,
        "output": str(out_path),
        "appended": bool(args.append),
        "first_sample_end": (
            samples[0].end_ts.isoformat() if samples else None
        ),
        "last_sample_end": (
            samples[-1].end_ts.isoformat() if samples else None
        ),
    })
    return 0


def cmd_waveform_match(args, cfg: Config, storage: Storage) -> int:
    """Match the current window against a library and report a bias (spec §7.3)."""
    df = fetch_ohlcv(
        args.symbol,
        interval=args.interval,
        period=args.period,
    )
    if len(df) < args.window:
        print(f"[err] need ≥ {args.window} bars; got {len(df)}", file=sys.stderr)
        return 1
    target_window = df.iloc[-args.window:]
    target_sig = compute_signature(target_window, method=args.normalize)

    library = read_library(Path(args.library))
    if not library:
        print(
            f"[err] empty/missing library at {args.library} — "
            "run `waveform-build-library` first",
            file=sys.stderr,
        )
        return 1

    bias, matches = waveform_lookup(
        target_sig,
        library,
        horizon_bars=args.horizon,
        method=args.method,
        top_k=args.top_k,
        min_score=args.min_score,
        min_sample_count=args.min_samples,
        min_directional_share=args.min_share,
    )

    _print({
        "symbol": args.symbol,
        "interval": args.interval,
        "window_bars": args.window,
        "library": args.library,
        "library_size": len(library),
        "method": args.method,
        "horizon_bars": args.horizon,
        "bias": bias.to_dict(),
        "top_matches": [m.to_dict() for m in matches[: args.show]],
    })
    return 0


def cmd_attribution_report(args, cfg: Config, storage: Storage) -> int:
    """Run attribution over the largest moves in `period` (spec §8)."""
    df = fetch_ohlcv(
        args.symbol, interval=args.interval, period=args.period,
        start=args.start, end=args.end,
    )
    closes = df["close"].astype(float).values
    rets_pct = (closes[1:] / closes[:-1] - 1.0) * 100.0
    bars = list(df.index[1:])

    events = []
    if not args.no_events:
        cal_health = calendar_freshness(EVENTS_PATH)
        if cal_health.is_fresh:
            events = load_events(EVENTS_PATH)
        else:
            print(
                f"[warn] calendar {cal_health.status}: "
                f"{cal_health.detail or 'unhealthy'}", file=sys.stderr,
            )

    # Pull macro deltas if available — single fetch for the full period
    macro = None
    if not args.no_macro:
        try:
            macro = fetch_macro_snapshot(
                df.index, interval="1d", period=args.macro_period,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] macro fetch failed: {e}", file=sys.stderr)

    # Pick the top-N moves by absolute return.
    indexed = sorted(
        enumerate(rets_pct), key=lambda t: abs(t[1]), reverse=True,
    )[: args.top_n]

    reports = []
    from datetime import timedelta
    for raw_i, ret in indexed:
        ts = bars[raw_i]
        # Window: events within ±6h of the move.
        nearby = [
            e.to_dict() for e in events
            if abs((e.when - ts.to_pydatetime()).total_seconds()) <= 6 * 3600
        ]
        # Macro deltas: change over the previous 24h.
        macro_chg: dict = {}
        if macro is not None:
            for slot in macro.series.keys():
                v_now = macro.value_at(slot, ts)
                v_prev = macro.value_at(slot, ts - timedelta(hours=24))
                if v_now is not None and v_prev is not None:
                    if slot in ("us10y", "us_short_yield_proxy"):
                        # Yields are quoted in %; report basis-point change
                        macro_chg[slot] = float((v_now - v_prev) * 100.0)
                    else:
                        macro_chg[slot] = (
                            float((v_now / v_prev - 1.0) * 100.0)
                            if v_prev != 0 else None
                        )
        result = attribute_move(
            float(ret),
            events_nearby=nearby,
            macro_changes_pct=macro_chg,
            notable_threshold_pct=args.notable_threshold,
        )
        reports.append({
            "ts": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "result": result.to_dict(),
        })

    _print({
        "symbol": args.symbol,
        "interval": args.interval,
        "n_bars": len(df),
        "top_n": args.top_n,
        "reports": reports,
    })
    return 0


def cmd_calibrate(args, cfg: Config, storage: Storage) -> int:
    """Sweep parameters and rank by stability_score (spec §10)."""
    df = fetch_ohlcv(
        args.symbol, interval=args.interval, period=args.period,
        start=args.start, end=args.end,
    )
    base = (
        StrategyConfig.load(Path(args.config)) if args.config
        else StrategyConfig(symbol=args.symbol, interval=args.interval)
    )

    ranges: dict = {}
    if args.stop_atr:
        ranges["stop_atr"] = [float(x) for x in args.stop_atr]
    if args.tp_atr:
        ranges["tp_atr"] = [float(x) for x in args.tp_atr]
    if args.max_holding_bars:
        ranges["max_holding_bars"] = [int(x) for x in args.max_holding_bars]
    if not ranges:
        # Sensible default: explore stop_atr and tp_atr
        ranges = {
            "stop_atr": [1.5, 2.0, 2.5],
            "tp_atr": [2.0, 3.0, 4.0],
        }

    report = calibrate(
        df,
        symbol=args.symbol,
        interval=args.interval,
        base=base,
        ranges=ranges,
    )
    _print(report.to_dict())
    return 0


def cmd_compare_external(args, cfg: Config, storage: Storage) -> int:
    """Compare engine decisions against an external CSV (spec §9)."""
    externals = read_external_csv(Path(args.external_csv), source=args.source)
    if not externals:
        _print({"n": 0, "note": "no external rows parsed"})
        return 0

    # Pull self decisions from the predictions table within the time window.
    if args.symbol:
        rows = storage.list_predictions(symbol=args.symbol, limit=args.limit)
    else:
        rows = storage.list_predictions(limit=args.limit)

    self_decisions = [
        {
            "symbol": r["symbol"],
            "ts": r["ts"],
            "action": r.get("final_decision_action") or r.get("action"),
            "advisory": {
                "event_risk_level": r.get("event_risk_level"),
                "sentiment_score": r.get("sentiment_score"),
                "pattern": r.get("detected_pattern"),
            },
            "entry_price": r.get("entry_price"),
            "stop_loss": None,
            "take_profit": None,
        }
        for r in rows
    ]
    pair_within = pd.Timedelta(args.pair_within)
    cmps = compare_signals(
        self_decisions, externals,
        pair_within=pair_within.to_pytimedelta(),
    )
    _print({
        "n_external": len(externals),
        "n_paired": len(cmps),
        "aggregate": aggregate_comparisons(cmps),
        "samples": [c.to_dict() for c in cmps[: args.show]],
    })
    return 0


def cmd_trace_stats(args, cfg: Config, storage: Storage) -> int:
    """Aggregate a `decision_traces.jsonl` (or .jsonl.gz) into a JSON summary.

    Errors raised by `aggregate_stats` (FileNotFoundError / ValueError)
    propagate as stderr + exit 2. When parsing succeeded but malformed
    lines were skipped, the stats JSON is still emitted to stdout AND
    exit code is 2 so a downstream caller can detect the partial read.
    """
    from .decision_trace_stats import aggregate_stats
    try:
        stats = aggregate_stats(
            args.path,
            top_n_hold_reasons=args.top_hold_reasons,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] trace-stats: {exc}", file=sys.stderr)
        return 2

    if args.pretty:
        out = json.dumps(stats, indent=2, ensure_ascii=False, default=str)
    else:
        out = json.dumps(stats, ensure_ascii=False, default=str)
    print(out)

    errors_total = stats["consistency_checks"]["errors_total"]
    if errors_total > 0:
        print(
            f"[warn] trace-stats: {errors_total} error(s) recorded; "
            f"see consistency_checks.errors",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_trace_stats_multi(args, cfg: Config, storage: Storage) -> int:
    """Aggregate multiple `decision_traces.jsonl` files at once.

    Per-run stats and a global pooled aggregate are emitted as a single
    JSON document. Failure semantics:
      * A `ValueError` from `aggregate_many` (no input paths or every
        path failed) is printed to stderr and the command exits 2.
      * If at least one path succeeded but `consistency_checks.errors_total`
        is non-zero (per-run errors or failed_runs), the JSON is still
        written to stdout AND the command exits 2 so the caller can
        detect the partial read.
    """
    from .decision_trace_stats import aggregate_many
    try:
        stats = aggregate_many(
            list(args.paths),
            top_n_hold_reasons=args.top_hold_reasons,
        )
    except ValueError as exc:
        print(f"[error] trace-stats-multi: {exc}", file=sys.stderr)
        return 2

    if args.pretty:
        out = json.dumps(stats, indent=2, ensure_ascii=False, default=str)
    else:
        out = json.dumps(stats, ensure_ascii=False, default=str)
    print(out)

    errors_total = stats["consistency_checks"]["errors_total"]
    if errors_total > 0:
        print(
            f"[warn] trace-stats-multi: {errors_total} error(s) recorded; "
            f"see consistency_checks.errors / failed_runs",
            file=sys.stderr,
        )
        return 2
    return 0


def cmd_review(args, cfg: Config, storage: Storage) -> int:
    if not cfg.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set; cannot run review.", file=sys.stderr)
        return 1
    trades = storage.recent_trades(limit=args.limit)
    analyses = storage.recent_analyses(limit=args.limit)
    analyst = Analyst(model=cfg.model, effort="high")
    print(analyst.review(trades or analyses))
    return 0


def cmd_watch(args, cfg: Config, storage: Storage) -> int:
    while True:
        try:
            cmd_analyze(args, cfg, storage)
        except Exception as e:  # noqa: BLE001
            print(f"[watch] error: {e}", file=sys.stderr)
        time.sleep(args.every)


def cmd_evaluate(args, cfg: Config, storage: Storage) -> int:
    """Score every PENDING prediction whose horizon has elapsed."""
    pending = storage.pending_predictions()
    if not pending:
        _print({"evaluated": 0, "note": "No pending predictions."})
        return 0

    summary = {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0,
               "INCONCLUSIVE": 0, "INSUFFICIENT_DATA": 0}
    details = []

    by_symbol: dict[tuple[str, str], list[dict]] = {}
    for p in pending:
        by_symbol.setdefault((p["symbol"], p["interval"]), []).append(p)

    for (symbol, interval), preds in by_symbol.items():
        try:
            df = fetch_ohlcv(symbol, interval=interval, period=args.period)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not fetch {symbol} {interval}: {e}", file=sys.stderr)
            continue
        for p in preds:
            after = slice_bars_after(df, p["ts"])
            verdict = evaluate_prediction(p, after)
            if verdict.status == "INSUFFICIENT_DATA":
                summary["INSUFFICIENT_DATA"] += 1
                continue
            storage.update_prediction_evaluation(
                prediction_id=p["id"],
                status=verdict.status,
                actual_direction=verdict.actual_direction,
                actual_magnitude_pct=verdict.actual_magnitude_pct,
                max_favorable_pct=verdict.max_favorable_pct,
                max_adverse_pct=verdict.max_adverse_pct,
                invalidation_hit=verdict.invalidation_hit,
                note=verdict.note,
            )
            summary[verdict.status] = summary.get(verdict.status, 0) + 1
            details.append(
                {
                    "prediction_id": p["id"],
                    "symbol": p["symbol"],
                    "action": p["action"],
                    "verdict": verdict.to_dict(),
                }
            )
            # Push verdict for terminal outcomes only (not INSUFFICIENT_DATA).
            if verdict.status in ("CORRECT", "PARTIAL", "WRONG"):
                _notify(
                    storage,
                    kind="verdict",
                    text=format_verdict(
                        symbol=p["symbol"],
                        action=p["action"],
                        status=verdict.status,
                        actual_direction=verdict.actual_direction,
                        actual_magnitude_pct=verdict.actual_magnitude_pct,
                        note=verdict.note,
                        analysis_id=p.get("analysis_id"),
                    ),
                )

    if any(summary.values()):
        _notify(
            storage,
            kind="summary",
            text=format_summary(period="evaluate", counts=summary),
        )
    _print({"summary": summary, "details": details})
    return 0


def cmd_postmortem(args, cfg: Config, storage: Storage) -> int:
    """Run Claude post-mortems on WRONG predictions that don't have one yet."""
    if not cfg.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set; cannot run post-mortems.", file=sys.stderr)
        return 1

    wrongs = storage.wrong_predictions_without_postmortem(limit=args.limit)
    if not wrongs:
        _print({"analyzed": 0, "note": "No wrong predictions awaiting post-mortem."})
        return 0

    pm = Postmortem(model=cfg.model, effort="high")
    written = []
    for p in wrongs:
        verdict_view = {
            "status": p["status"],
            "actual_direction": p["actual_direction"],
            "actual_magnitude_pct": p["actual_magnitude_pct"],
            "max_favorable_pct": p["max_favorable_pct"],
            "max_adverse_pct": p["max_adverse_pct"],
            "invalidation_hit": bool(p["invalidation_hit"]),
            "note": p["evaluation_note"],
        }
        try:
            result = pm.analyze(p, verdict_view)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] post-mortem failed for prediction {p['id']}: {e}",
                  file=sys.stderr)
            continue
        # Spec §13: capture the diagnostic snapshot of what the engine
        # saw at decision time, so future readers see the WHY, not just
        # the prediction.
        pm_context = {
            "final_decision_action": p.get("final_decision_action"),
            "executed_action": p.get("executed_action"),
            "blocked_by": p.get("blocked_by"),
            "rule_chain": p.get("rule_chain"),
            "detected_pattern": p.get("detected_pattern"),
            "trend_state": p.get("trend_state"),
            "higher_timeframe_trend": p.get("higher_timeframe_trend"),
            "event_risk_level": p.get("event_risk_level"),
            "economic_events_nearby": p.get("economic_events_nearby"),
            "sentiment_score": p.get("sentiment_score"),
            "sentiment_volume_spike": p.get("sentiment_volume_spike"),
            "spread_at_entry": p.get("spread_at_entry"),
            "risk_reward": p.get("risk_reward"),
            "rule_version": p.get("rule_version"),
            "final_reason": p.get("final_reason"),
        }
        pm_id = storage.save_postmortem(
            prediction_id=p["id"],
            root_cause=result.root_cause,
            narrative=result.narrative,
            proposed_rule=result.proposed_rule,
            tags=",".join(result.tags),
            context=pm_context,
        )
        written.append(
            {
                "postmortem_id": pm_id,
                "prediction_id": p["id"],
                "root_cause": result.root_cause,
                "proposed_rule": result.proposed_rule,
            }
        )
        _notify(
            storage,
            kind="lesson",
            text=format_lesson(
                symbol=p["symbol"],
                root_cause=result.root_cause,
                proposed_rule=result.proposed_rule,
                narrative=result.narrative,
                analysis_id=p.get("analysis_id"),
            ),
        )

    _print({"written": written, "count": len(written)})
    return 0


def cmd_lessons(args, cfg: Config, storage: Storage) -> int:
    """Show what we've learned so far."""
    summary = storage.lesson_summary()
    recent = storage.relevant_postmortems(args.symbol, limit=args.limit) if args.symbol else []
    _print(
        {
            "root_cause_counts": summary,
            "recent_for_symbol": recent,
            "symbol": args.symbol,
        }
    )
    return 0


def cmd_sentiment_refresh(args, cfg: Config, storage: Storage) -> int:
    if not cfg.anthropic_api_key:
        print("ANTHROPIC_API_KEY not set; cannot score sentiment.", file=sys.stderr)
        return 1
    enable = {
        "reddit": not args.no_reddit,
        "stocktwits": not args.no_stocktwits,
        "tradingview": not args.no_tradingview,
        "twitter": not args.no_twitter,
        "rss": not args.no_rss,
    }
    result = refresh_sentiment_snapshot(
        symbols=args.symbols,
        output_path=SENTIMENT_PATH,
        enable=enable,
        translate=args.translate,
    )
    _print(
        {
            "elapsed_s": result.elapsed_s,
            "twitter_backend": result.twitter_backend,
            "counts_by_source": result.counts_by_source,
            "errors_by_source": result.errors_by_source,
            "snapshots": {
                sym: {
                    "mention_count_24h": s.mention_count_24h,
                    "sentiment_score": s.sentiment_score,
                    "sentiment_velocity": s.sentiment_velocity,
                }
                for sym, s in result.snapshots.items()
            },
        }
    )
    return 0


def cmd_sentiment_show(args, cfg: Config, storage: Storage) -> int:
    snaps = load_sentiment_snapshot(SENTIMENT_PATH)
    if args.symbol:
        snap = snaps.get(args.symbol)
        if snap is None:
            _print({"error": f"no snapshot for {args.symbol}"})
            return 1
        _print(snap.to_dict())
    else:
        _print(
            {
                "symbols": {sym: s.to_dict() for sym, s in snaps.items()},
            }
        )
    return 0


def cmd_calendar_seed(args, cfg: Config, storage: Storage) -> int:
    events = seed_defaults(EVENTS_PATH)
    _print(
        {
            "path": str(EVENTS_PATH),
            "seeded": [e.to_dict() for e in events],
            "note": "These are placeholder entries. Replace with a real feed.",
        }
    )
    return 0


def _add_context_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by analyze / watch / trade."""
    parser.add_argument("--symbol", default="USDJPY=X")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--no-news", action="store_true")
    parser.add_argument("--no-correlation", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--no-lessons", action="store_true",
                        help="Skip injecting past post-mortems")
    parser.add_argument("--no-sentiment", action="store_true",
                        help="Skip the cached crowd-sentiment snapshot")
    parser.add_argument("--no-higher-tf", action="store_true",
                        help="Skip higher-timeframe fetch (faster, less safe)")
    parser.add_argument("--lesson-limit", type=int, default=5)
    parser.add_argument("--news-limit", type=int, default=5)
    parser.add_argument("--event-window-hours", type=int, default=24)
    parser.add_argument(
        "--config", default=None,
        help="StrategyConfig YAML/JSON; overrides risk/waveform/min_confidence",
    )


def _load_strategy_config(args) -> None:
    """Stamp `args.strategy_config` from `--config` if provided.

    Mutates `args` in place — keeps every cmd_* function dependency-free
    on argparse plumbing.
    """
    path = getattr(args, "config", None)
    args.strategy_config = (
        StrategyConfig.load(Path(path)) if path else None
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fx", description="FX analysis toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Run a single analysis")
    _add_context_flags(a)
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("trade", help="Analyze and optionally place an order")
    _add_context_flags(t)
    t.add_argument("--broker", choices=["paper", "oanda"], default="paper")
    t.add_argument("--capital", type=float, default=10_000.0)
    # Defaults are None so the StrategyConfig (or hard-coded fallback)
    # supplies the actual values inside cmd_trade. This lets `--config
    # cfg.yaml` override stop/TP/risk_pct via the file.
    t.add_argument("--risk-pct", type=float, default=None,
                   help="Fraction of capital risked per trade "
                        "(falls back to StrategyConfig.risk.risk_pct, then 0.01)")
    t.add_argument("--stop-atr", type=float, default=None)
    t.add_argument("--tp-atr", type=float, default=None)
    t.add_argument("--dry-run", action="store_true",
                   help="Print the plan but do not place the order")
    t.add_argument("--confirm-demo", action="store_true",
                   help="Required for OANDA broker; acknowledges demo-only use")
    t.add_argument("--trace-out", default=None,
                   help="Directory to write run_metadata.json and "
                        "decision_traces.jsonl (one line for this trade). "
                        "Skip the flag to disable live trace export.")
    t.add_argument("--trace-out-default", action="store_true",
                   help="When --trace-out is omitted, write the live trace "
                        "to runs/live/<run_id>/. --trace-out wins if both.")
    t.add_argument("--overwrite", action="store_true",
                   help="Allow live trace export to overwrite existing "
                        "files in the target dir.")
    t.set_defaults(func=cmd_trade)

    b = sub.add_parser("backtest", help="Backtest technical strategy")
    b.add_argument("--symbol", default="USDJPY=X")
    b.add_argument("--interval", default="1h")
    b.add_argument("--period", default="180d")
    b.add_argument("--start", default=None)
    b.add_argument("--end", default=None)
    b.add_argument("--warmup", type=int, default=50)
    b.set_defaults(func=cmd_backtest)

    bt = sub.add_parser(
        "build-timeline",
        help="Build a point-in-time market timeline (spec §5)",
    )
    bt.add_argument("--symbol", default="USDJPY=X")
    bt.add_argument("--interval", default="1h")
    bt.add_argument("--period", default="180d")
    bt.add_argument("--start", default=None)
    bt.add_argument("--end", default=None)
    bt.add_argument("--warmup", type=int, default=50)
    bt.add_argument("--no-events", action="store_true")
    bt.add_argument("--no-macro", action="store_true",
                    help="Skip yields/DXY/VIX/equities fetch")
    bt.add_argument(
        "--macro-slots", nargs="*", default=None,
        help=f"Subset of {sorted(MACRO_SYMBOLS.keys())}; default = all",
    )
    bt.add_argument("--macro-period", default="2y",
                    help="yfinance period for macro fetch")
    bt.add_argument("--no-higher-tf", action="store_true")
    bt.add_argument(
        "--output", default=None,
        help="Optional path; .csv or .parquet — written if set",
    )
    bt.set_defaults(func=cmd_build_timeline)

    wbl = sub.add_parser(
        "waveform-build-library",
        help="Slide a window over history and write a JSONL waveform library (spec §7.2)",
    )
    wbl.add_argument("--symbol", default="USDJPY=X")
    wbl.add_argument("--interval", default="1h")
    wbl.add_argument("--period", default="2y")
    wbl.add_argument("--start", default=None)
    wbl.add_argument("--end", default=None)
    wbl.add_argument("--window-bars", type=int, default=60)
    wbl.add_argument("--step-bars", type=int, default=5)
    wbl.add_argument("--horizons", type=int, nargs="+", default=[4, 12, 24])
    wbl.add_argument("--normalize",
                     choices=["z_score", "start_price", "min_max"],
                     default="z_score")
    wbl.add_argument("--output", default="data/waveforms/library.jsonl")
    wbl.add_argument("--append", action="store_true",
                     help="Append to an existing library instead of overwriting")
    wbl.set_defaults(func=cmd_waveform_build_library)

    wm = sub.add_parser(
        "waveform-match",
        help="Match the most recent window against a library; report bias (spec §7.3)",
    )
    wm.add_argument("--symbol", default="USDJPY=X")
    wm.add_argument("--interval", default="1h")
    wm.add_argument("--period", default="60d")
    wm.add_argument("--window", type=int, default=60,
                    help="Bars at the right edge of the chart used as the target")
    wm.add_argument("--library", default="data/waveforms/library.jsonl")
    wm.add_argument("--method", choices=["dtw", "cosine", "correlation"],
                    default="dtw")
    wm.add_argument("--horizon", type=int, default=24)
    wm.add_argument("--top-k", type=int, default=30)
    wm.add_argument("--min-score", type=float, default=0.55)
    wm.add_argument("--min-samples", type=int, default=20)
    wm.add_argument("--min-share", type=float, default=0.6)
    wm.add_argument("--show", type=int, default=5,
                    help="How many top matches to print")
    wm.add_argument("--normalize",
                    choices=["z_score", "start_price", "min_max"],
                    default="z_score")
    wm.set_defaults(func=cmd_waveform_match)

    ar = sub.add_parser(
        "attribution-report",
        help="Rank candidate causes for the largest moves (spec §8)",
    )
    ar.add_argument("--symbol", default="USDJPY=X")
    ar.add_argument("--interval", default="1h")
    ar.add_argument("--period", default="180d")
    ar.add_argument("--start", default=None)
    ar.add_argument("--end", default=None)
    ar.add_argument("--top-n", type=int, default=10)
    ar.add_argument("--no-events", action="store_true")
    ar.add_argument("--no-macro", action="store_true")
    ar.add_argument("--macro-period", default="2y")
    ar.add_argument("--notable-threshold", type=float, default=0.5,
                    help="|return|%% above which a move is flagged as notable")
    ar.set_defaults(func=cmd_attribution_report)

    cb = sub.add_parser(
        "calibrate",
        help="Sweep parameter ranges and rank by stability (spec §10)",
    )
    cb.add_argument("--symbol", default="USDJPY=X")
    cb.add_argument("--interval", default="1h")
    cb.add_argument("--period", default="180d")
    cb.add_argument("--start", default=None)
    cb.add_argument("--end", default=None)
    cb.add_argument("--config", default=None,
                    help="Optional StrategyConfig YAML/JSON to use as the base")
    cb.add_argument("--stop-atr", nargs="*", default=None,
                    help="Values to try for stop_atr (e.g. 1.5 2.0 2.5)")
    cb.add_argument("--tp-atr", nargs="*", default=None,
                    help="Values to try for tp_atr")
    cb.add_argument("--max-holding-bars", nargs="*", default=None,
                    help="Values to try for max_holding_bars")
    cb.set_defaults(func=cmd_calibrate)

    ce = sub.add_parser(
        "compare-external",
        help="Compare engine predictions against an external-vendor CSV (spec §9)",
    )
    ce.add_argument("--external-csv", required=True,
                    help="Path to a vendor-exported CSV file")
    ce.add_argument("--source", required=True,
                    help="Vendor label (quantconnect, trendspider, ...)")
    ce.add_argument("--symbol", default=None,
                    help="Restrict self-side to this symbol; default = all")
    ce.add_argument("--limit", type=int, default=500,
                    help="Max self-predictions to read for pairing")
    ce.add_argument("--pair-within", default="1h",
                    help="Pandas-style timedelta for pairing tolerance")
    ce.add_argument("--show", type=int, default=10,
                    help="How many sample comparisons to print")
    ce.set_defaults(func=cmd_compare_external)

    eo = sub.add_parser(
        "event-overlay",
        help="Aggregate pre/post-event price action (spec §6)",
    )
    eo.add_argument("--symbol", default="USDJPY=X")
    eo.add_argument("--interval", default="1h")
    eo.add_argument("--period", default="2y")
    eo.add_argument("--start", default=None)
    eo.add_argument("--end", default=None)
    eo.add_argument(
        "--event", default=None,
        choices=[k for k, _ in KEYWORD_MAP],
        help="Restrict to events matching this keyword",
    )
    eo.add_argument(
        "--impact", default="high",
        choices=["high", "medium", "low"],
        help="Restrict to events of this impact (None to include all)",
    )
    eo.add_argument("--limit", type=int, default=20,
                    help="Max per-event rows to print (aggregate is full)")
    eo.set_defaults(func=cmd_event_overlay)

    be = sub.add_parser(
        "backtest-engine",
        help="Backtest with the full Decision Engine + Risk Gate chain (spec §4)",
    )
    be.add_argument("--symbol", default="USDJPY=X")
    be.add_argument("--interval", default="1h")
    be.add_argument("--period", default="180d")
    be.add_argument("--start", default=None)
    be.add_argument("--end", default=None)
    be.add_argument("--warmup", type=int, default=50)
    be.add_argument("--stop-atr", type=float, default=2.0)
    be.add_argument("--tp-atr", type=float, default=3.0)
    be.add_argument("--max-holding-bars", type=int, default=48)
    be.add_argument("--no-events", action="store_true",
                    help="Ignore data/events.json (skip macro-event gate)")
    be.add_argument("--no-higher-tf", action="store_true",
                    help="Skip higher-timeframe alignment check")
    be.add_argument("--no-macro", action="store_true",
                    help="Do not fetch / attach the macro snapshot (DXY, "
                         "US10Y, VIX, indices). trace.macro_context will "
                         "be null. Decision logic is unchanged either way "
                         "— this only affects what's recorded.")
    be.add_argument("--macro-period", default="2y",
                    help="yfinance period for the macro snapshot fetch "
                         "(default 2y). Must comfortably cover --period.")
    be.add_argument("--waveform-library", default=None,
                    help="Path to a waveform library JSONL produced by "
                         "`waveform-build-library`. When given, trace.waveform"
                         ".waveform_bias is populated per bar via a "
                         "horizon-aware look-ahead-safe lookup. Decision "
                         "logic is unchanged — bias is observability only. "
                         "REQUIRES a PR-#15+ library that records "
                         "`forward_return_end_ts` per horizon (older "
                         "libraries are rejected at lookup time and "
                         "logged as unavailable_reason; rebuild with the "
                         "current `waveform-build-library`). Recommended: "
                         "build the library from data PRIOR to the "
                         "backtest period (out-of-sample).")
    be.add_argument("--context-days", type=int, default=0,
                    help="Days of historical OHLCV to fetch BEFORE the "
                         "test window for long_term_trend / SMA200 / "
                         "monthly_trend computation. Observation-only — "
                         "the judgement path (ATR / patterns / "
                         "higher_tf / RSI / waveform target signature / "
                         "decide_action) NEVER sees this context, so "
                         "decisions are byte-identical with or without "
                         "it. Default 0 (legacy behaviour). For "
                         "`--interval 1h`, context_days + test span "
                         "must be ≤730 (yfinance hard limit).")
    be.add_argument("--parameter-profile", default=None,
                    help="Literature parameter catalog profile to record "
                         "in run_metadata.parameters. PR #19: "
                         "METADATA-ONLY by default. Even with "
                         "`--parameter-profile literature_baseline_v1` "
                         "alone the runtime gate / decide_action / "
                         "risk_gate values are unchanged — "
                         "`applied_to_runtime` stays False. trade list "
                         "/ metrics / hold_reasons / equity_curve are "
                         "byte-identical with or without this flag "
                         "(pinned by "
                         "test_parameter_profile_metadata_only_trade_"
                         "results_identical). To actually apply the "
                         "profile to backtest runtime, ALSO pass "
                         "--apply-parameter-profile. Unknown profile "
                         "names are rejected with exit code 2.")
    be.add_argument("--apply-parameter-profile", action="store_true",
                    help="PR #21: actually apply the catalog selected "
                         "by --parameter-profile to the backtest "
                         "engine's runtime. Backtest-only — live "
                         "(cmd_trade --broker oanda) and paper paths "
                         "are unaffected. Without this flag, "
                         "--parameter-profile stays metadata-only "
                         "(PR #19 invariant preserved). Requires "
                         "--parameter-profile X; passing this flag "
                         "alone is rejected with exit code 2. "
                         "literature_baseline_v1 differs from "
                         "current_runtime defaults ONLY on "
                         "stop_atr_mult (1.5 vs 2.0); other "
                         "indicator periods / thresholds already "
                         "match current defaults.")
    be.add_argument(
        "--decision-profile",
        default="current_runtime",
        choices=(
            "current_runtime",
            "royal_road_decision_v1",
            "royal_road_decision_v2",
            "royal_road_decision_v2_integrated",
        ),
        help=(
            "Which decision profile drives BUY/SELL/HOLD. Default "
            "`current_runtime` is byte-identical to PR #21 main. "
            "`royal_road_decision_v1` (opt-in) reads "
            "technical_confluence_v1 and applies royal-road rules "
            "with strictness selected by --royal-road-mode. "
            "`royal_road_decision_v2_integrated` (opt-in) elevates "
            "wave / W-lines / fib / candlestick / dow / MA / RSI / "
            "BB / MACD / invalidation / RR into first-class evidence "
            "the action is built from; mode selected by "
            "--integrated-mode. "
            "Backtest-only — live cmd_trade does not call "
            "run_engine_backtest, so this flag cannot affect live "
            "trading. The trace gains a `royal_road_decision` slice "
            "with comparison metadata when the profile is non-default."
        ),
    )
    be.add_argument(
        "--integrated-mode",
        default="integrated_balanced",
        choices=("integrated_balanced", "integrated_strict"),
        help=(
            "Strictness mode for `royal_road_decision_v2_integrated`. "
            "`integrated_balanced` (default): required-data missing "
            "(macro / daily_roadmap / symbol_macro_briefing) → WARN. "
            "`integrated_strict`: required-data missing → HOLD. RR / "
            "stop / invalidation / WNL still HOLD if missing in both "
            "modes. Only consulted when --decision-profile is "
            "royal_road_decision_v2_integrated."
        ),
    )
    from .royal_road_decision_modes import (
        DEFAULT_ROYAL_ROAD_MODE as _DEFAULT_ROYAL_ROAD_MODE,
        supported_royal_road_modes as _supported_royal_road_modes,
    )
    be.add_argument(
        "--royal-road-mode",
        default=_DEFAULT_ROYAL_ROAD_MODE,
        choices=_supported_royal_road_modes(),
        help=(
            "Strictness mode for `royal_road_decision_v1`. Only "
            "consulted when --decision-profile is "
            "royal_road_decision_v1. `balanced` (default) is the "
            "research candidate (STRONG enters with >=1 axis, WEAK "
            "with >=2 axes; invalidation_clear / structure_stop "
            "missing become cautions). `strict` is a diagnostic "
            "baseline (STRONG only, WEAK -> HOLD, missing "
            "invalidation/structure_stop -> block). `exploratory` "
            "is for discovering adjacent rules (WEAK with >=1 axis) "
            "and is NOT a candidate for adoption. All three modes "
            "are heuristic and pending validation."
        ),
    )
    be.add_argument(
        "--lower-tf",
        default=None,
        choices=("15m", "5m"),
        help=(
            "Optional lower-timeframe interval to fetch alongside the "
            "base interval. Used by royal_road_decision_v2 only "
            "(`--decision-profile royal_road_decision_v2`). When "
            "omitted, v2's lower_tf_trigger field reports "
            "unavailable_reason=not_attached. Future-leak-safe: only "
            "lower-TF bars with timestamp <= the base bar's close "
            "timestamp are visible. Backtest-only — has no effect on "
            "live cmd_trade."
        ),
    )
    be.add_argument(
        "--stop-mode",
        default="atr",
        choices=("atr", "structure", "hybrid"),
        help=(
            "Stop-placement mode. `atr` (default) = entry +/- "
            "stop_atr_mult * ATR (byte-identical to PR #21 main). "
            "`structure` = the v2 structure_stop_price (most recent "
            "confirmed swing low for BUY, swing high for SELL); HOLD "
            "if missing. `hybrid` = structure if 0.5..3.0 ATR away, "
            "ATR fallback if too far, HOLD if too close. Only "
            "consulted when --decision-profile is "
            "royal_road_decision_v2; otherwise the choice is recorded "
            "in metadata only. Backtest-only."
        ),
    )
    be.add_argument("--trace-out", default=None,
                    help="Directory to write run_metadata.json / "
                         "decision_traces.jsonl / summary.json. Skip the "
                         "flag to disable export (default behaviour).")
    be.add_argument("--trace-out-default", action="store_true",
                    help="When --trace-out is omitted, write the trace "
                         "artefacts to runs/<run_id>/. --trace-out wins "
                         "if both are given.")
    be.add_argument("--overwrite", action="store_true",
                    help="Allow --trace-out to overwrite existing files. "
                         "Without this flag, any pre-existing output file "
                         "in the target dir aborts the export.")
    be.add_argument("--gzip", action="store_true",
                    help="Write decision_traces.jsonl.gz instead of plain "
                         "JSONL (only meaningful with --trace-out).")
    be.set_defaults(func=cmd_backtest_engine)

    ts = sub.add_parser(
        "trace-stats",
        help="Aggregate a decision_traces.jsonl (or .gz) to a JSON summary",
    )
    ts.add_argument("path",
                    help="Path to decision_traces.jsonl or .jsonl.gz")
    ts.add_argument("--pretty", action="store_true",
                    help="Pretty-print the output JSON (indent=2). "
                         "Default is compact one-line JSON.")
    ts.add_argument("--top-hold-reasons", type=int, default=10,
                    help="Number of (reason, count) pairs to surface in "
                         "top_hold_reasons (default 10).")
    ts.set_defaults(func=cmd_trace_stats)

    tsm = sub.add_parser(
        "trace-stats-multi",
        help="Aggregate multiple decision_traces.jsonl files (per-run + global)",
    )
    tsm.add_argument("paths", nargs="+",
                     help="Paths to one or more decision_traces.jsonl(.gz). "
                          "Globs expand via the shell.")
    tsm.add_argument("--pretty", action="store_true",
                     help="Pretty-print the output JSON (indent=2).")
    tsm.add_argument("--top-hold-reasons", type=int, default=10,
                     help="Number of (reason, count) pairs to surface in "
                          "global.top_hold_reasons (default 10).")
    tsm.set_defaults(func=cmd_trace_stats_multi)

    r = sub.add_parser("review", help="Weekly self-review via Claude")
    r.add_argument("--limit", type=int, default=50)
    r.set_defaults(func=cmd_review)

    w = sub.add_parser("watch", help="Continuously analyze at a fixed interval")
    _add_context_flags(w)
    w.add_argument("--every", type=int, default=900, help="Seconds between analyses")
    w.set_defaults(func=cmd_watch)

    c = sub.add_parser("calendar-seed", help="Write a placeholder events.json")
    c.set_defaults(func=cmd_calendar_seed)

    e = sub.add_parser(
        "evaluate", help="Score pending predictions against actual price action"
    )
    e.add_argument(
        "--period", default="60d",
        help="Lookback window for price fetch (must cover the prediction's horizon)",
    )
    e.set_defaults(func=cmd_evaluate)

    pm = sub.add_parser(
        "postmortem", help="Run Claude post-mortems on WRONG predictions"
    )
    pm.add_argument("--limit", type=int, default=10)
    pm.set_defaults(func=cmd_postmortem)

    ls = sub.add_parser("lessons", help="Show accumulated post-mortem lessons")
    ls.add_argument("--symbol", default=None,
                    help="If set, also list recent lessons for this symbol")
    ls.add_argument("--limit", type=int, default=10)
    ls.set_defaults(func=cmd_lessons)

    sr = sub.add_parser("sentiment-refresh",
                        help="Collect + score crowd sentiment for the listed symbols")
    sr.add_argument("--symbols", nargs="+", default=["USDJPY=X", "EURUSD=X", "BTC-USD"])
    sr.add_argument("--no-reddit", action="store_true")
    sr.add_argument("--no-stocktwits", action="store_true")
    sr.add_argument("--no-tradingview", action="store_true")
    sr.add_argument("--no-twitter", action="store_true")
    sr.add_argument("--no-rss", action="store_true")
    sr.add_argument("--translate", action="store_true",
                    help="Translate non-English posts to English before scoring")
    sr.set_defaults(func=cmd_sentiment_refresh)

    ss = sub.add_parser("sentiment-show", help="Print the cached sentiment snapshot")
    ss.add_argument("--symbol", default=None)
    ss.set_defaults(func=cmd_sentiment_show)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = Config.load()
    storage = Storage(cfg.db_path)
    _load_strategy_config(args)
    return args.func(args, cfg, storage)


if __name__ == "__main__":
    sys.exit(main())
