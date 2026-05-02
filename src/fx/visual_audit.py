"""Visual audit for royal_road_decision_v2 chart reconstruction.

Purpose
-------
A human looking at a chart sees zones, trendlines, patterns, lower-TF
triggers, and the structure stop / target plan. v2 already produces all
of those as data; this module turns them into a per-bar audit artefact:

  - sidecar JSON  (always emitted) — full v2 payload PLUS render
    metadata (the bar window actually used, future-leak boundary,
    overlay coordinates), so a human or another process can verify
    the system "saw" the same chart they would draw.
  - PNG image     (optional; only when matplotlib is available) — OHLC
    candlestick overlay of the visible window with the same overlays.
    When matplotlib is missing, an `.image_unavailable` marker file is
    written instead so audits can detect the gap explicitly.

This module is observation-only:
  - it never reads decide_action,
  - it never modifies the trace it's auditing,
  - it never touches live / OANDA / paper paths,
  - it never writes to runs/ or libs/ unless the caller explicitly
    points it at a writable directory.

Future-leak rule
----------------
The render uses ONLY bars with `ts <= parent_bar_ts`. The sidecar
records `parent_bar_ts`, `render_window_start_ts`, `render_window_end_ts`,
and `bars_used_in_render` so consumers can verify the boundary was
respected. Pinned by `tests/test_visual_audit.py::test_future_bars_not_in_render`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd


SCHEMA_VERSION: Final[str] = "visual_audit_v1"

# Maximum number of bars to feed into the renderer (so very-long
# windows don't produce unreadable PNGs). The sidecar records the
# truncated bar count too. Pure visualization choice — does not
# influence any decision.
_MAX_RENDER_BARS: Final[int] = 600


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _trace_v2_payload(trace: Any) -> dict | None:
    """Extract the royal_road_decision_v2 dict from a trace (object or
    dict). Returns None when the slice is absent."""
    sl = _safe_get(trace, "royal_road_decision_v2")
    if sl is None:
        return None
    if isinstance(sl, dict):
        return sl
    to_dict = getattr(sl, "to_dict", None)
    return to_dict() if callable(to_dict) else None


def _trace_timestamp(trace: Any) -> pd.Timestamp | None:
    ts = _safe_get(trace, "timestamp")
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return ts
    try:
        return pd.Timestamp(ts)
    except Exception:  # noqa: BLE001
        return None


def build_visual_audit_payload(
    *,
    trace: Any,
    df: pd.DataFrame,
) -> dict | None:
    """Build the audit sidecar JSON for one trace.

    Returns None when the trace has no `royal_road_decision_v2` slice
    (i.e. the default current_runtime profile or v1-only profile).
    Otherwise returns a dict matching `visual_audit_v1` schema.
    """
    v2 = _trace_v2_payload(trace)
    if v2 is None:
        return None
    parent_ts = _trace_timestamp(trace)
    if parent_ts is None:
        # Fall back to df's last index — still future-leak safe because
        # we never look past it.
        parent_ts = pd.Timestamp(df.index[-1]) if len(df) > 0 else None
    visible_full = (
        df[df.index <= parent_ts] if (parent_ts is not None and len(df) > 0)
        else df.iloc[0:0]
    )
    visible = visible_full.tail(_MAX_RENDER_BARS)
    n_visible = int(len(visible))
    render_start_ts = (
        visible.index[0].isoformat() if n_visible > 0 else None
    )
    render_end_ts = (
        visible.index[-1].isoformat() if n_visible > 0 else None
    )
    parent_iso = (
        parent_ts.isoformat() if isinstance(parent_ts, pd.Timestamp)
        else str(parent_ts) if parent_ts is not None else None
    )
    sr = v2.get("support_resistance_v2") or {}
    tl = v2.get("trendline_context") or {}
    cp = v2.get("chart_pattern_v2") or {}
    ltf = v2.get("lower_tf_trigger") or {}
    stop_plan = v2.get("structure_stop_plan") or {}
    rq = v2.get("reconstruction_quality") or {}
    best = v2.get("best_setup")

    # Per-overlay drawing coordinates that downstream renderers can
    # consume directly. All times are ISO strings; all prices are
    # floats.
    overlays = {
        "level_zones_selected": [
            {
                "kind": lvl.get("kind"),
                "zone_low": lvl.get("zone_low"),
                "zone_high": lvl.get("zone_high"),
                "price": lvl.get("price"),
                "first_touch_ts": lvl.get("first_touch_ts"),
                "last_touch_ts": lvl.get("last_touch_ts"),
                "touch_count": lvl.get("touch_count"),
                "confidence": lvl.get("confidence"),
                "reasons": lvl.get("reasons"),
            }
            for lvl in sr.get("selected_level_zones_top5", [])
        ],
        "level_zones_rejected": [
            {
                "kind": lvl.get("kind"),
                "zone_low": lvl.get("zone_low"),
                "zone_high": lvl.get("zone_high"),
                "price": lvl.get("price"),
                "reject_reason": lvl.get("reject_reason"),
            }
            for lvl in sr.get("rejected_level_zones", [])
        ],
        "trendlines_selected": [
            {
                "kind": t.get("kind"),
                "slope": t.get("slope"),
                "intercept": t.get("intercept"),
                "anchor_indices": t.get("anchor_indices"),
                "anchor_prices": t.get("anchor_prices"),
                "touch_count": t.get("touch_count"),
                "is_strong": t.get("is_strong"),
                "broken": t.get("broken"),
                "confidence": t.get("confidence"),
            }
            for t in tl.get("selected_trendlines_top3", [])
        ],
        "trendlines_rejected": [
            {
                "kind": (rj.get("trendline") or {}).get("kind"),
                "slope": (rj.get("trendline") or {}).get("slope"),
                "anchor_indices": (rj.get("trendline") or {}).get("anchor_indices"),
                "reject_reason": rj.get("reject_reason"),
            }
            for rj in tl.get("rejected_trendlines", [])
        ],
        "patterns_selected": [
            {
                "kind": p.get("kind"),
                "neckline": p.get("neckline"),
                "neckline_broken": p.get("neckline_broken"),
                "retested": p.get("retested"),
                "side_bias": p.get("side_bias"),
                "anchor_indices": p.get("anchor_indices"),
                "upper_line": p.get("upper_line"),
                "lower_line": p.get("lower_line"),
                "invalidation_price": p.get("invalidation_price"),
                "target_price": p.get("target_price"),
                "pattern_quality_score": p.get("pattern_quality_score"),
            }
            for p in cp.get("selected_patterns_top5", [])
        ],
        "patterns_rejected": [
            {
                "kind": (rj.get("pattern") or {}).get("kind"),
                "neckline": (rj.get("pattern") or {}).get("neckline"),
                "side_bias": (rj.get("pattern") or {}).get("side_bias"),
                "reject_reason": rj.get("reject_reason"),
            }
            for rj in cp.get("rejected_patterns", [])
        ],
        "lower_tf_trigger": (
            {
                "interval": ltf.get("interval"),
                "trigger_ts": ltf.get("trigger_ts"),
                "trigger_price": ltf.get("trigger_price"),
                "trigger_type": ltf.get("trigger_type"),
                "trigger_strength": ltf.get("trigger_strength"),
                "available": ltf.get("available"),
            }
            if ltf else None
        ),
        "structure_stop_plan": (
            {
                "chosen_mode": stop_plan.get("chosen_mode"),
                "outcome": stop_plan.get("outcome"),
                "stop_price": stop_plan.get("stop_price"),
                "take_profit_price": stop_plan.get("take_profit_price"),
                "rr_realized": stop_plan.get("rr_realized"),
                "structure_stop_price": stop_plan.get("structure_stop_price"),
                "atr_stop_price": stop_plan.get("atr_stop_price"),
            }
            if stop_plan else None
        ),
    }

    title_parts: list[str] = []
    if best is not None:
        title_parts.append(f"best={best.get('side')} score={best.get('score', 0.0):.3f}")
    else:
        title_parts.append("best=NONE")
    title_parts.append(
        f"quality={rq.get('total_reconstruction_score', 0.0):.3f}"
    )
    if v2.get("action"):
        title_parts.append(f"action={v2['action']}")
    title = " | ".join(title_parts)

    return {
        "schema_version": SCHEMA_VERSION,
        "parent_bar_ts": parent_iso,
        "render_window_start_ts": render_start_ts,
        "render_window_end_ts": render_end_ts,
        "bars_used_in_render": n_visible,
        "max_render_bars": _MAX_RENDER_BARS,
        "title": title,
        "profile": v2.get("profile"),
        "mode": v2.get("mode"),
        "action": v2.get("action"),
        "best_setup": best,
        "reconstruction_quality": rq,
        "overlays": overlays,
        # Full v2 payload preserved so the audit is self-contained
        # (no need to read the JSONL trace separately).
        "royal_road_decision_v2": v2,
    }


def _render_image_optional(
    *,
    payload: dict,
    df: pd.DataFrame,
    out_path: Path,
) -> dict:
    """Best-effort matplotlib render. Writes a `<out_path>.image_unavailable`
    marker file when matplotlib is missing or rendering fails.

    Returns a dict reporting `image_status` ∈ {"rendered", "matplotlib_missing", "render_error:<msg>"}.
    """
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as e:  # noqa: BLE001
        marker = out_path.with_suffix(".image_unavailable")
        marker.write_text(
            f"matplotlib_missing: {type(e).__name__}: {e}\n"
        )
        return {
            "image_status": "matplotlib_missing",
            "marker_file": str(marker),
        }

    try:
        n_render = int(payload.get("bars_used_in_render") or 0)
        if n_render <= 0 or len(df) == 0:
            marker = out_path.with_suffix(".image_unavailable")
            marker.write_text("no_visible_bars\n")
            return {
                "image_status": "render_error:no_visible_bars",
                "marker_file": str(marker),
            }
        end_ts = pd.Timestamp(payload["render_window_end_ts"])
        visible = df[df.index <= end_ts].tail(n_render)
        fig, ax = plt.subplots(figsize=(14, 8))
        # Simple OHLC: plot close as line + shade high/low band.
        ax.plot(visible.index, visible["close"], color="black", linewidth=0.8)
        ax.fill_between(
            visible.index, visible["low"], visible["high"],
            color="lightgray", alpha=0.3,
        )

        overlays = payload.get("overlays") or {}
        # SR zones — selected (green/red), rejected (gray)
        for lvl in overlays.get("level_zones_selected", []):
            zlow = lvl.get("zone_low")
            zhigh = lvl.get("zone_high")
            if zlow is None or zhigh is None:
                continue
            color = "green" if lvl.get("kind") == "support" else (
                "red" if lvl.get("kind") == "resistance" else "purple"
            )
            ax.axhspan(zlow, zhigh, color=color, alpha=0.15)
        for lvl in overlays.get("level_zones_rejected", []):
            zlow = lvl.get("zone_low")
            zhigh = lvl.get("zone_high")
            if zlow is None or zhigh is None:
                continue
            ax.axhspan(zlow, zhigh, color="gray", alpha=0.05)
        # Trendlines — selected (blue), rejected skipped to avoid clutter
        for t in overlays.get("trendlines_selected", []):
            slope = t.get("slope")
            intercept = t.get("intercept")
            if slope is None or intercept is None:
                continue
            anchors = t.get("anchor_indices") or []
            if len(anchors) >= 2:
                xs = list(range(anchors[0], anchors[-1] + 1))
                ys = [slope * x + intercept for x in xs]
                # Map anchor indices to visible timestamps where possible.
                if anchors[-1] < len(visible):
                    ts_xs = [visible.index[x] for x in xs if x < len(visible)]
                    ys = ys[: len(ts_xs)]
                    ax.plot(
                        ts_xs, ys, color="blue", linewidth=0.7,
                        alpha=0.6 if t.get("broken") else 0.85,
                    )
        # Patterns — neckline as horizontal line at neckline price
        for p in overlays.get("patterns_selected", []):
            nl = p.get("neckline")
            if nl is None:
                continue
            color = "darkgreen" if p.get("side_bias") == "BUY" else (
                "darkred" if p.get("side_bias") == "SELL" else "darkorange"
            )
            ax.axhline(
                nl, color=color, linestyle="--", linewidth=0.8, alpha=0.7,
            )
        # Lower TF trigger marker
        ltf = overlays.get("lower_tf_trigger")
        if ltf and ltf.get("trigger_ts") and ltf.get("trigger_price"):
            try:
                t_ts = pd.Timestamp(ltf["trigger_ts"])
                ax.scatter(
                    [t_ts], [ltf["trigger_price"]],
                    color="orange", s=60, marker="^",
                    label=ltf.get("trigger_type"),
                )
            except Exception:  # noqa: BLE001
                pass
        # Structure stop / target lines
        sp = overlays.get("structure_stop_plan")
        if sp:
            if sp.get("stop_price") is not None:
                ax.axhline(
                    sp["stop_price"], color="red", linestyle=":",
                    linewidth=1.0, alpha=0.7, label="stop",
                )
            if sp.get("take_profit_price") is not None:
                ax.axhline(
                    sp["take_profit_price"], color="green", linestyle=":",
                    linewidth=1.0, alpha=0.7, label="tp",
                )

        ax.set_title(payload.get("title", "visual_audit"))
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return {"image_status": "rendered", "image_path": str(out_path)}
    except Exception as e:  # noqa: BLE001
        marker = out_path.with_suffix(".image_unavailable")
        marker.write_text(f"render_error: {type(e).__name__}: {e}\n")
        return {
            "image_status": f"render_error:{type(e).__name__}",
            "marker_file": str(marker),
        }


def render_visual_audit(
    *,
    trace: Any,
    df: pd.DataFrame,
    out_dir: Path | str,
    name_prefix: str = "audit",
) -> dict:
    """Render one trace into sidecar JSON (always) and a PNG (best-effort).

    Returns a dict with `status` ∈ {"v2_absent", "rendered", "json_only"},
    `json_path`, and optional `image_path` / `marker_file`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_visual_audit_payload(trace=trace, df=df)
    if payload is None:
        return {
            "status": "v2_absent",
            "reason": (
                "trace.royal_road_decision_v2 is None; default "
                "current_runtime / v1-only profile does not produce "
                "a v2 visual audit"
            ),
        }
    json_path = out_dir / f"{name_prefix}.json"
    json_path.write_text(
        json.dumps(payload, indent=2, default=str)
    )
    image_result = _render_image_optional(
        payload=payload,
        df=df,
        out_path=out_dir / f"{name_prefix}.png",
    )
    out: dict = {
        "status": "rendered" if image_result.get("image_status") == "rendered"
        else "json_only",
        "json_path": str(json_path),
    }
    out.update(image_result)
    return out


__all__ = [
    "SCHEMA_VERSION",
    "build_visual_audit_payload",
    "render_visual_audit",
]
