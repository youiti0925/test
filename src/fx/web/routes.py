"""Routes for the FX dashboard."""
from __future__ import annotations

import json

from flask import (
    Blueprint,
    Response,
    abort,
    current_app,
    render_template,
    request,
    stream_with_context,
)

from ..correlation import RELATED_SYMBOLS
from .pipeline_stream import run_pipeline

bp = Blueprint("fx", __name__)


DEFAULT_SYMBOL_OPTIONS = [
    "USDJPY=X",
    "EURUSD=X",
    "GBPUSD=X",
    "AUDUSD=X",
    "EURJPY=X",
    "BTC-USD",
    "ETH-USD",
]
INTERVAL_OPTIONS = ["15m", "1h", "4h", "1d"]


@bp.route("/")
def dashboard():
    storage = current_app.config["FX_STORAGE"]
    cfg = current_app.config["FX_CONFIG"]
    stats = storage.dashboard_stats()
    recent = storage.recent_analyses(limit=15)
    lessons = storage.lesson_summary()
    return render_template(
        "dashboard.html",
        stats=stats,
        recent=recent,
        lessons=lessons,
        symbol_options=sorted(set(DEFAULT_SYMBOL_OPTIONS + list(RELATED_SYMBOLS))),
        config={"model": cfg.model, "effort": cfg.effort,
                "api_key_set": bool(cfg.anthropic_api_key)},
    )


@bp.route("/analyze")
def analyze_page():
    storage = current_app.config["FX_STORAGE"]
    cfg = current_app.config["FX_CONFIG"]
    return render_template(
        "analyze.html",
        symbol_options=sorted(set(DEFAULT_SYMBOL_OPTIONS + list(RELATED_SYMBOLS))),
        interval_options=INTERVAL_OPTIONS,
        events_configured=current_app.config["FX_EVENTS_PATH"].exists(),
        api_key_set=bool(cfg.anthropic_api_key),
    )


@bp.route("/analyze/stream")
def analyze_stream():
    storage = current_app.config["FX_STORAGE"]
    cfg = current_app.config["FX_CONFIG"]
    events_path = current_app.config["FX_EVENTS_PATH"]

    symbol = request.args.get("symbol", "USDJPY=X")
    interval = request.args.get("interval", "1h")
    period = request.args.get("period", "60d")
    want_llm = request.args.get("llm", "1") == "1"
    want_news = request.args.get("news", "1") == "1"
    want_correlation = request.args.get("correlation", "1") == "1"
    want_events = request.args.get("events", "1") == "1"
    want_lessons = request.args.get("lessons", "1") == "1"

    def stream():
        try:
            for event in run_pipeline(
                symbol=symbol,
                interval=interval,
                period=period,
                cfg=cfg,
                storage=storage,
                events_path=events_path,
                want_llm=want_llm,
                want_news=want_news,
                want_correlation=want_correlation,
                want_events=want_events,
                want_lessons=want_lessons,
            ):
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as e:  # noqa: BLE001
            yield (
                "data: "
                + json.dumps({"step": "fatal", "status": "error", "data": str(e)})
                + "\n\n"
            )

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@bp.route("/analysis/<int:analysis_id>")
def analysis_detail(analysis_id):
    storage = current_app.config["FX_STORAGE"]
    record = storage.get_analysis(analysis_id)
    if record is None:
        abort(404)
    analysis = record["analysis"]
    snapshot = json.loads(analysis["snapshot_json"])
    prediction = record["prediction"]
    return render_template(
        "analysis_detail.html",
        analysis=analysis,
        snapshot=snapshot,
        prediction=prediction,
    )


@bp.route("/predictions")
def predictions_list():
    storage = current_app.config["FX_STORAGE"]
    status = request.args.get("status") or None
    symbol = request.args.get("symbol") or None
    preds = storage.list_predictions(status=status, symbol=symbol, limit=200)
    statuses = ["PENDING", "CORRECT", "PARTIAL", "WRONG", "INCONCLUSIVE"]
    return render_template(
        "predictions.html",
        predictions=preds,
        status_filter=status,
        symbol_filter=symbol,
        statuses=statuses,
    )


@bp.route("/lessons")
def lessons_page():
    storage = current_app.config["FX_STORAGE"]
    summary = storage.lesson_summary()
    recent = storage.list_postmortems(limit=50)
    return render_template(
        "lessons.html",
        summary=summary,
        recent=recent,
    )


@bp.app_template_filter("fmtfloat")
def fmt_float(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


@bp.app_template_filter("fmtpct")
def fmt_pct(value, digits=2):
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


@bp.app_template_filter("statuscolor")
def status_color(status):
    return {
        "CORRECT": "emerald",
        "PARTIAL": "amber",
        "WRONG": "rose",
        "INCONCLUSIVE": "slate",
        "PENDING": "sky",
        "INSUFFICIENT_DATA": "slate",
    }.get(status, "slate")
