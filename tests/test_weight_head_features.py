"""Tests for cattle_phenotyping.pipeline.weight_head_features.

The feature builder is the contract between the pose-prediction stage and
the XGBoost weight head; the schema must stay stable across versions and
the skip-reason taxonomy must catch every degenerate input early.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cattle_phenotyping.data.kaggle import (
    CANONICAL_SIDE_KEYPOINTS,
    FilenameMeta,
    KaggleSample,
)
from cattle_phenotyping.pipeline.weight_head_features import (
    FeatureSkip,
    KNOWN_BATCHES,
    WEIGHT_HEAD_FEATURE_NAMES,
    build_features,
)


# Helpers -------------------------------------------------------------------


def _make_sample(
    *,
    batch: str = "B3",
    image_w: int = 4160,
    image_h: int = 3120,
    weight_kg: float | None = 180.0,
) -> KaggleSample:
    """Construct a minimal KaggleSample for feature-builder tests."""
    return KaggleSample(
        image_path=Path(f"/fake/{batch}/img.jpg"),
        batch=batch,  # type: ignore[arg-type]
        view="side",
        coco_image_id=1,
        coco_category_id=1,
        keypoints={n: None for n in CANONICAL_SIDE_KEYPOINTS},
        bbox=None,
        image_width=image_w,
        image_height=image_h,
        filename_meta=FilenameMeta(
            animal_id="1", view="side", sex="F", weight_kg=weight_kg,
        ),
        mask_path=None,
    )


def _full_predicted_keypoints(
    *,
    chord_px: float = 800.0,
    length_px: float = 1500.0,
    height_px: float = 1200.0,
    conf: float = 0.9,
) -> dict[str, tuple[float, float, float]]:
    """Build a complete predicted-keypoint dict with a consistent geometry.

    All 9 canonical keypoints have a confidence ``conf``. The girth pairs
    span ``chord_px`` vertically; length pair spans ``length_px`` horizontally;
    height pair spans ``height_px`` vertically. Positions don't have to be
    anatomically realistic — only the chord lengths matter for the features.
    """
    return {
        "wither":             (1500.0,  600.0, conf),
        "pinbone":            (1500.0 + length_px, 600.0, conf),  # length chord
        "shoulderbone":       (1500.0, 600.0, conf),
        "front_girth_top":    (1700.0, 600.0, conf),
        "front_girth_bottom": (1700.0, 600.0 + chord_px, conf),
        "rear_girth_top":     (2700.0, 600.0, conf),
        "rear_girth_bottom":  (2700.0, 600.0 + chord_px, conf),
        "height_top":         (2000.0, 300.0, conf),
        "height_bottom":      (2000.0, 300.0 + height_px, conf),
    }


_DEFAULT_STICKER_CM2 = {"B3": 15.27, "B4": 79.33}


# Schema invariants ---------------------------------------------------------


def test_feature_names_unique():
    assert len(set(WEIGHT_HEAD_FEATURE_NAMES)) == len(WEIGHT_HEAD_FEATURE_NAMES)


def test_feature_names_include_schaeffer_and_batch_onehots():
    assert "schaeffer_kg" in WEIGHT_HEAD_FEATURE_NAMES
    # One column per known batch — adding a batch requires updating both.
    for b in KNOWN_BATCHES:
        assert f"batch_{b}" in WEIGHT_HEAD_FEATURE_NAMES


def test_build_features_returns_dict_in_schema_order():
    sample = _make_sample(batch="B3")
    preds = _full_predicted_keypoints()
    out = build_features(sample, preds, sticker_area_px=900, sticker_cm2_by_batch=_DEFAULT_STICKER_CM2)
    assert not isinstance(out, FeatureSkip)
    assert tuple(out.keys()) == WEIGHT_HEAD_FEATURE_NAMES


# Skip taxonomy -------------------------------------------------------------


def test_unknown_batch_yields_skip():
    sample = _make_sample(batch="B7")  # type: ignore[arg-type]  # not in KNOWN_BATCHES
    out = build_features(
        sample, _full_predicted_keypoints(), sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    assert isinstance(out, FeatureSkip)
    assert "unknown batch" in out.reason


def test_missing_sticker_cm2_for_batch_yields_skip():
    sample = _make_sample(batch="B4")
    out = build_features(
        sample, _full_predicted_keypoints(), sticker_area_px=900,
        sticker_cm2_by_batch={"B3": 15.27},  # only B3 calibrated
    )
    assert isinstance(out, FeatureSkip)
    assert "no sticker cm² calibration" in out.reason


def test_zero_sticker_area_yields_skip():
    out = build_features(
        _make_sample(), _full_predicted_keypoints(), sticker_area_px=0,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    assert isinstance(out, FeatureSkip)
    assert "sticker_area_px" in out.reason


def test_missing_front_girth_keypoint_yields_skip():
    preds = _full_predicted_keypoints()
    del preds["front_girth_top"]
    out = build_features(
        _make_sample(), preds, sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    assert isinstance(out, FeatureSkip)
    assert "front girth" in out.reason


def test_missing_length_keypoint_yields_skip():
    preds = _full_predicted_keypoints()
    del preds["shoulderbone"]
    out = build_features(
        _make_sample(), preds, sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    assert isinstance(out, FeatureSkip)
    assert "length" in out.reason


def test_below_conf_threshold_treated_as_missing():
    preds = _full_predicted_keypoints(conf=0.4)
    out = build_features(
        _make_sample(), preds, sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
        conf_threshold=0.5,
    )
    assert isinstance(out, FeatureSkip)
    # Either "all predicted keypoints below conf threshold" or downstream
    # "front girth keypoints missing" — both are valid skip reasons; we
    # care that we don't silently return zeroed features.
    assert out.reason


# Numerical correctness -----------------------------------------------------


def test_sticker_scale_recovers_known_px_per_cm():
    """If sticker area is N px and cm² is M, px_per_cm should be sqrt(N/M)."""
    sample = _make_sample(batch="B3")
    preds = _full_predicted_keypoints()
    out = build_features(
        sample, preds, sticker_area_px=900,
        sticker_cm2_by_batch={"B3": 9.0, "B4": 79.33},
    )
    assert not isinstance(out, FeatureSkip)
    # 900 / 9 = 100, sqrt = 10
    assert abs(out["px_per_cm"] - 10.0) < 1e-9


def test_geometric_features_consistent_with_chord_inputs():
    """Front girth chord px should equal the |girth_top - girth_bottom| we constructed."""
    preds = _full_predicted_keypoints(chord_px=800.0, length_px=1500.0)
    out = build_features(
        _make_sample(), preds, sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    assert not isinstance(out, FeatureSkip)
    assert abs(out["front_girth_chord_px"] - 800.0) < 1e-6
    assert abs(out["body_length_px"] - 1500.0) < 1e-6


def test_schaeffer_kg_matches_pure_schaeffer_call():
    """The schaeffer_kg feature should match models.schaeffer.schaeffer_from_keypoints."""
    from cattle_phenotyping.models.schaeffer import schaeffer_from_keypoints

    preds = _full_predicted_keypoints(chord_px=800.0, length_px=1500.0)
    out = build_features(
        _make_sample(batch="B3"), preds, sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    assert not isinstance(out, FeatureSkip)

    # Compute Schaeffer manually with the same inputs.
    triples = {n: (kp[0], kp[1], 2) for n, kp in preds.items()}
    expected = schaeffer_from_keypoints(triples, out["px_per_cm"])
    assert expected is not None
    assert abs(out["schaeffer_kg"] - expected) < 1e-6


def test_batch_onehot_is_mutually_exclusive():
    out_b3 = build_features(
        _make_sample(batch="B3"), _full_predicted_keypoints(), sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    out_b4 = build_features(
        _make_sample(batch="B4"), _full_predicted_keypoints(), sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    assert not isinstance(out_b3, FeatureSkip)
    assert not isinstance(out_b4, FeatureSkip)
    assert (out_b3["batch_B3"], out_b3["batch_B4"]) == (1.0, 0.0)
    assert (out_b4["batch_B3"], out_b4["batch_B4"]) == (0.0, 1.0)


def test_ratios_are_finite_and_positive():
    out = build_features(
        _make_sample(), _full_predicted_keypoints(), sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    assert not isinstance(out, FeatureSkip)
    for key in ("front_to_rear_girth_ratio_cm", "girth_to_length_ratio_cm",
                "length_to_height_ratio_cm"):
        assert math.isfinite(out[key])
        assert out[key] > 0


def test_kp_conf_summary_min_le_mean():
    preds = _full_predicted_keypoints(conf=0.9)
    preds["wither"] = (preds["wither"][0], preds["wither"][1], 0.1)  # one low-conf kp
    out = build_features(
        _make_sample(), preds, sticker_area_px=900,
        sticker_cm2_by_batch=_DEFAULT_STICKER_CM2,
    )
    assert not isinstance(out, FeatureSkip)
    assert out["kp_conf_min"] <= out["kp_conf_mean"]
    assert abs(out["kp_conf_min"] - 0.1) < 1e-6
