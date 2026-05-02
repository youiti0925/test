"""Tests for src/fx/pattern_shape_review.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.fx.pattern_shape_review import (
    SCHEMA_VERSION,
    PatternShapeReview,
    build_pattern_shape_review,
)
from src.fx.waveform_encoder import encode_wave_skeleton


def _double_bottom_curve(n: int = 200) -> pd.DataFrame:
    x = np.linspace(0.0, 1.0, n)

    def y(t: float) -> float:
        if t < 0.20:
            return 0.70 - (0.70 - 0.05) * (t / 0.20)
        if t < 0.45:
            return 0.05 + (0.55 - 0.05) * ((t - 0.20) / 0.25)
        if t < 0.70:
            return 0.55 - (0.55 - 0.10) * ((t - 0.45) / 0.25)
        return 0.10 + (0.90 - 0.10) * ((t - 0.70) / 0.30)
    base = np.array([y(t) for t in x])
    rng = np.random.default_rng(31)
    close = 1.10 + base * 0.05 + rng.normal(0, 0.0008, n)
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": close - 0.0001, "high": close + 0.0008,
        "low": close - 0.0008, "close": close,
        "volume": [1000] * n,
    }, index=idx)


def test_empty_review_when_no_skeletons():
    review = build_pattern_shape_review({})
    assert isinstance(review, PatternShapeReview)
    assert review.schema_version == SCHEMA_VERSION
    assert review.best_pattern is None
    assert review.detected_patterns == ()


def test_audit_notes_include_unvalidated_warning():
    review = build_pattern_shape_review({"micro": None})
    assert any(
        "heuristic" in n.lower() or "validated" in n.lower()
        for n in review.audit_notes
    )


def test_per_scale_marks_unavailable_for_none_skeleton():
    review = build_pattern_shape_review({"micro": None, "short": None})
    for scale in ("micro", "short"):
        assert review.per_scale[scale]["available"] is False


def test_double_bottom_yields_review_with_best_pattern():
    df = _double_bottom_curve()
    skel = encode_wave_skeleton(df, scale="short")
    review = build_pattern_shape_review({"short": skel})
    assert review.best_pattern is not None
    assert review.best_pattern.kind == "double_bottom"
    info = review.per_scale["short"]
    assert info["available"] is True
    assert info["best_shape"] == "double_bottom"
    assert info["status"] in {"forming", "neckline_broken"}


def test_to_dict_keys():
    review = build_pattern_shape_review({"micro": None})
    d = review.to_dict()
    assert {"schema_version", "per_scale", "detected_patterns",
            "best_pattern", "overall_summary_ja",
            "entry_interpretation_ja", "risk_note_ja",
            "audit_notes"} <= set(d.keys())


def test_japanese_text_fields_are_non_empty_strings():
    review = build_pattern_shape_review({"micro": None})
    assert isinstance(review.overall_summary_ja, str)
    assert isinstance(review.entry_interpretation_ja, str)
    assert isinstance(review.risk_note_ja, str)
    # All scales unavailable → summary mentions inability
    assert review.overall_summary_ja
