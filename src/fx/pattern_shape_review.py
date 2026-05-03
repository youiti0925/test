"""Human-readable pattern shape review (observation-only).

Wraps the per-scale shape-matching results from
`pattern_shape_matcher` into a single Japanese-language audit summary
that the visual_audit layer can render directly to the user.

This module is observation-only. It does NOT influence
`royal_road_decision_v2`'s final BUY/SELL/HOLD logic.

Output schema
-------------
PatternShapeReview:
  schema_version       : "pattern_shape_review_v1"
  per_scale            : {scale: {available, trend, best_shape, ...}}
  detected_patterns    : tuple of best ShapeMatch per scale
  best_pattern         : ShapeMatch | None  (overall, ranked by score)
  overall_summary_ja   : str
  entry_interpretation_ja : str
  risk_note_ja         : str
  audit_notes          : tuple[str, ...]  (e.g. heuristic_thresholds_unvalidated)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .pattern_shape_matcher import (
    CANDIDATE_SHAPE_SCORE,
    STRONG_SHAPE_SCORE,
    ShapeMatch,
    match_skeleton,
)
from .pattern_templates import all_templates
from .waveform_encoder import WaveSkeleton


SCHEMA_VERSION: Final[str] = "pattern_shape_review_v1"


@dataclass(frozen=True)
class PatternShapeReview:
    schema_version: str
    per_scale: dict
    detected_patterns: tuple[ShapeMatch, ...]
    best_pattern: ShapeMatch | None
    overall_summary_ja: str
    entry_interpretation_ja: str
    risk_note_ja: str
    audit_notes: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "per_scale": dict(self.per_scale),
            "detected_patterns": [
                m.to_dict() for m in self.detected_patterns
            ],
            "best_pattern": (
                self.best_pattern.to_dict()
                if self.best_pattern is not None else None
            ),
            "overall_summary_ja": str(self.overall_summary_ja),
            "entry_interpretation_ja": str(self.entry_interpretation_ja),
            "risk_note_ja": str(self.risk_note_ja),
            "audit_notes": list(self.audit_notes),
        }


_TREND_LABEL_JA: Final[dict] = {
    "UP": "上昇傾向",
    "DOWN": "下降傾向",
    "RANGE": "レンジ",
    "MIXED": "もみ合い",
    "UNKNOWN": "不明",
}


def _scale_summary_ja(
    scale: str,
    skel: WaveSkeleton | None,
    best: ShapeMatch | None,
) -> dict:
    if skel is None or not skel.pivots:
        return {
            "available": False,
            "reason": "skeleton_unavailable",
            "trend": "UNKNOWN",
            "best_shape": None,
            "shape_score": None,
            "status": None,
            "human_label": "波形情報不足",
            "trade_context": "履歴不足のため形は判定できません。",
        }
    trend_label = _TREND_LABEL_JA.get(skel.trend_hint, skel.trend_hint)
    if best is None or best.shape_score < CANDIDATE_SHAPE_SCORE:
        return {
            "available": True,
            "trend": skel.trend_hint,
            "best_shape": None,
            "shape_score": (
                float(best.shape_score) if best is not None else 0.0
            ),
            "status": "not_matched",
            "human_label": f"波形: {trend_label}",
            "trade_context": "明確なテンプレート形は見えません。",
        }
    score_label = (
        "強い候補" if best.shape_score >= STRONG_SHAPE_SCORE else "候補"
    )
    return {
        "available": True,
        "trend": skel.trend_hint,
        "best_shape": best.kind,
        "shape_score": float(best.shape_score),
        "status": best.status,
        "human_label": f"{best.human_label} ({score_label})",
        "trade_context": best.human_explanation,
    }


def _overall_summary(
    per_scale: dict,
) -> str:
    parts = []
    for scale, info in per_scale.items():
        if info.get("available") and info.get("best_shape"):
            parts.append(
                f"{scale}: {info['human_label']} ({info.get('status')})"
            )
        elif info.get("available"):
            label = info.get("human_label") or "波形情報のみ"
            parts.append(f"{scale}: {label}")
    if not parts:
        return "全スケールで形は不明確です。"
    return " / ".join(parts)


def _entry_interpretation(best: ShapeMatch | None) -> str:
    if best is None:
        return "形からは方向の根拠は出ていません。"
    if best.status == "neckline_broken":
        return (
            f"形だけ見れば {best.human_label} のブレイクが完了しており、"
            f"{best.side_bias} 方向の根拠になります。ただし、王道では"
            "リターンムーブ確認とRR検証が必要です。"
        )
    if best.status == "forming":
        return (
            f"形だけ見れば {best.human_label} ですが、ネックライン (もしくは"
            f"ブレイクライン) を確定させていないため、王道では確定エントリー"
            "にはなりません。"
        )
    if best.status == "retested":
        return (
            f"{best.human_label} のブレイク後リターンムーブも確認できています。"
            "形としては最も整った段階です。"
        )
    return f"{best.human_label} ですが、形は弱いか崩れています。"


def _risk_note(best: ShapeMatch | None) -> str:
    if best is None:
        return "形を理由にした入りは避けてください。"
    if best.status == "forming":
        return (
            "形成中の段階で入る場合は先回りエントリーになります。"
            "王道ではブレイクまたはリターンムーブ確認を待つ判断になります。"
        )
    if best.status == "neckline_broken":
        return (
            "ブレイク直後は騙しもあるため、ATR・RR・上位足整合の確認が必要です。"
        )
    if best.status == "retested":
        return "リターンムーブ後の押し目は王道のエントリー候補ですが、損切り設置を必ず行ってください。"
    return "形が崩れているため、形を根拠としたエントリーは見送る判断が王道です。"


def build_pattern_shape_review(
    skeletons_by_scale: dict[str, WaveSkeleton | None],
    *,
    templates=None,
) -> PatternShapeReview:
    """Build a PatternShapeReview from per-scale skeletons.

    `skeletons_by_scale` keys are scale names (e.g. "micro", "short",
    "medium", "long"). Values can be None (or empty WaveSkeleton) for
    scales where reconstruction is unavailable.
    """
    pool = templates if templates is not None else all_templates()

    per_scale: dict = {}
    detected: list[ShapeMatch] = []
    audit_notes: list[str] = [
        "shape_thresholds_are_heuristic_and_not_validated",
        "observation_only_not_used_in_decision",
    ]

    for scale, skel in skeletons_by_scale.items():
        best = None
        if skel is not None and skel.pivots:
            matches = match_skeleton(skel, templates=pool)
            if matches:
                best = matches[0]
                if best.shape_score >= CANDIDATE_SHAPE_SCORE:
                    detected.append(best)
        per_scale[scale] = _scale_summary_ja(scale, skel, best)

    # Overall best across scales (highest shape_score)
    best_overall: ShapeMatch | None = None
    for m in detected:
        if best_overall is None or m.shape_score > best_overall.shape_score:
            best_overall = m

    return PatternShapeReview(
        schema_version=SCHEMA_VERSION,
        per_scale=per_scale,
        detected_patterns=tuple(detected),
        best_pattern=best_overall,
        overall_summary_ja=_overall_summary(per_scale),
        entry_interpretation_ja=_entry_interpretation(best_overall),
        risk_note_ja=_risk_note(best_overall),
        audit_notes=tuple(audit_notes),
    )


__all__ = [
    "SCHEMA_VERSION",
    "PatternShapeReview",
    "build_pattern_shape_review",
]
