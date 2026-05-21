"""Tests for keypoint_eval (per-keypoint pixel error + forward Schaeffer)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from cattle_phenotyping.data.kaggle import CANONICAL_SIDE_KEYPOINTS, FilenameMeta, KaggleSample
from cattle_phenotyping.eval.keypoint_eval import (
    build_eval_report,
    per_keypoint_pixel_errors,
    predict_weights_via_schaeffer,
    summarize_pixel_errors,
)
from cattle_phenotyping.models.schaeffer import schaeffer_weight_kg


# --------------------------------------------------------------- helpers


def _gtkp(x: float, y: float, v: int = 2) -> tuple[float, float, int]:
    return (x, y, v)


def _predkp(x: float, y: float, conf: float = 0.9) -> tuple[float, float, float]:
    return (x, y, conf)


def _sample(
    *,
    name: str = "img.jpg",
    keypoints: dict | None = None,
    batch: str = "B3",
    weight_kg: float | None = 200.0,
    image_width: int = 1000,
    image_height: int = 800,
) -> KaggleSample:
    return KaggleSample(
        image_path=Path(name),
        batch=batch,  # type: ignore[arg-type]
        view="side",
        coco_image_id=1, coco_category_id=1,
        keypoints=keypoints or {},
        bbox=None,
        image_width=image_width, image_height=image_height,
        filename_meta=FilenameMeta(
            animal_id="1", view="side", weight_kg=weight_kg, sex="F",  # type: ignore[arg-type]
        ),
    )


# ----------------------------------------------- per_keypoint_pixel_errors


def test_per_kp_errors_computes_euclidean_distance():
    """Predicted (103, 104), GT (100, 100) → 5px error."""
    samples = {"x.jpg": _sample(name="x.jpg", keypoints={"wither": _gtkp(100.0, 100.0)})}
    preds = {"x.jpg": {"wither": _predkp(103.0, 104.0)}}
    errs = per_keypoint_pixel_errors(preds, samples)
    assert errs["wither"] == [5.0]
    # Every other keypoint name should have an empty list (no GT visible).
    for kp in CANONICAL_SIDE_KEYPOINTS:
        if kp == "wither":
            continue
        assert errs[kp] == []


def test_per_kp_errors_skips_image_without_prediction():
    samples = {
        "with_pred.jpg": _sample(name="with_pred.jpg", keypoints={"wither": _gtkp(0, 0)}),
        "without_pred.jpg": _sample(name="without_pred.jpg", keypoints={"wither": _gtkp(0, 0)}),
    }
    preds = {"with_pred.jpg": {"wither": _predkp(3, 4)}}  # distance = 5
    errs = per_keypoint_pixel_errors(preds, samples)
    assert errs["wither"] == [5.0]


def test_per_kp_errors_skips_invisible_gt():
    samples = {
        "visible.jpg": _sample(name="visible.jpg", keypoints={"wither": _gtkp(100, 100, v=2)}),
        "occluded.jpg": _sample(name="occluded.jpg", keypoints={"wither": _gtkp(100, 100, v=1)}),
        "missing.jpg": _sample(name="missing.jpg", keypoints={"wither": _gtkp(0, 0, v=0)}),
    }
    preds = {n: {"wither": _predkp(103, 104)} for n in samples}
    errs = per_keypoint_pixel_errors(preds, samples)
    # Visible (v=2) AND occluded (v=1) both contribute. v=0 does not.
    assert sorted(errs["wither"]) == [5.0, 5.0]


def test_per_kp_errors_skips_missing_prediction_for_keypoint():
    """Predictions dict has fewer keypoints than canonical order — partial fine."""
    samples = {"x.jpg": _sample(
        keypoints={"wither": _gtkp(100, 100), "pinbone": _gtkp(200, 200)},
    )}
    preds = {"x.jpg": {"wither": _predkp(103, 104)}}  # no pinbone prediction
    errs = per_keypoint_pixel_errors(preds, samples)
    assert errs["wither"] == [5.0]
    assert errs["pinbone"] == []


def test_per_kp_errors_aggregates_across_samples():
    samples = {
        "a.jpg": _sample(name="a.jpg", keypoints={"wither": _gtkp(0, 0)}),
        "b.jpg": _sample(name="b.jpg", keypoints={"wither": _gtkp(10, 10)}),
        "c.jpg": _sample(name="c.jpg", keypoints={"wither": _gtkp(20, 20)}),
    }
    preds = {
        "a.jpg": {"wither": _predkp(3, 4)},    # 5
        "b.jpg": {"wither": _predkp(13, 14)},  # 5
        "c.jpg": {"wither": _predkp(20, 30)},  # 10
    }
    errs = per_keypoint_pixel_errors(preds, samples)
    assert sorted(errs["wither"]) == [5.0, 5.0, 10.0]


def test_per_kp_errors_respects_custom_keypoint_order():
    """Restricting keypoint_order should only emit those keys."""
    samples = {"x.jpg": _sample(
        keypoints={"wither": _gtkp(0, 0), "pinbone": _gtkp(0, 0)},
    )}
    preds = {"x.jpg": {"wither": _predkp(3, 4), "pinbone": _predkp(5, 12)}}
    errs = per_keypoint_pixel_errors(preds, samples, keypoint_order=("wither",))
    assert set(errs.keys()) == {"wither"}


# ----------------------------------------------- summarize_pixel_errors


def test_summary_basic_stats():
    per_kp = {"wither": [3.0, 5.0, 7.0, 9.0, 11.0]}
    summary = summarize_pixel_errors(per_kp)
    assert summary["wither"]["n"] == 5
    assert summary["wither"]["median_px"] == pytest.approx(7.0)
    assert summary["wither"]["mean_px"] == pytest.approx(7.0)
    # p90 with n=5 floor-indexed → index 3 → 9.0
    assert summary["wither"]["p90_px"] == pytest.approx(9.0)


def test_summary_empty_lists_get_none_placeholders():
    per_kp = {"wither": [], "pinbone": [5.0]}
    summary = summarize_pixel_errors(per_kp)
    assert summary["wither"] == {"n": 0, "median_px": None, "mean_px": None, "p90_px": None}
    assert summary["pinbone"]["n"] == 1
    assert summary["pinbone"]["median_px"] == pytest.approx(5.0)


def test_summary_p90_single_sample():
    per_kp = {"wither": [42.0]}
    summary = summarize_pixel_errors(per_kp)
    assert summary["wither"]["p90_px"] == pytest.approx(42.0)


# --------------------------------- predict_weights_via_schaeffer (records)


def _balanced_geometry_kps():
    """Keypoints that exercise the full Schaeffer formula."""
    return {
        "front_girth_top": _gtkp(0.0, 0.0),
        "front_girth_bottom": _gtkp(0.0, 100.0),  # chord = 100 px
        "shoulderbone": _gtkp(0.0, 0.0),
        "pinbone": _gtkp(200.0, 0.0),  # length = 200 px
    }


def test_predict_via_schaeffer_round_trips_perfect_prediction():
    """If predicted keypoints == GT and calibration is right → residual ≈ 0."""
    gt_kps = _balanced_geometry_kps()
    # Sticker setup: 400 px, cm² = 100 → px_per_cm = 2 → chord_cm = 50, len_cm = 100
    expected_kg = schaeffer_weight_kg(50.0 * math.pi, 100.0)
    sample = _sample(name="a.jpg", keypoints=gt_kps, weight_kg=expected_kg, batch="B3")
    pred = {"a.jpg": {n: _predkp(v[0], v[1], 0.95) for n, v in gt_kps.items()}}

    records = predict_weights_via_schaeffer(
        pred, [sample],
        sticker_area_px_by_filename={"a.jpg": 400},
        sticker_area_cm2_by_batch={"B3": 100.0},
    )
    assert len(records) == 1
    assert records[0].predicted_weight_kg == pytest.approx(expected_kg, rel=1e-9)
    assert records[0].residual_kg == pytest.approx(0.0, abs=1e-9)
    assert records[0].px_per_cm == pytest.approx(2.0)


def test_predict_via_schaeffer_skip_reasons_compose():
    """Each skip path emits a SchaefferRecord with the matching reason."""
    gt_kps = _balanced_geometry_kps()
    samples = [
        _sample(name="no_label.jpg", keypoints=gt_kps, weight_kg=None),
        _sample(name="no_pred.jpg", keypoints=gt_kps, weight_kg=200.0),
        _sample(name="no_mask.jpg", keypoints=gt_kps, weight_kg=200.0),
        _sample(name="zero_mask.jpg", keypoints=gt_kps, weight_kg=200.0),
        _sample(name="bad_batch.jpg", keypoints=gt_kps, weight_kg=200.0, batch="B2"),
    ]
    pred = {
        "no_label.jpg": {n: _predkp(v[0], v[1]) for n, v in gt_kps.items()},
        "no_mask.jpg":  {n: _predkp(v[0], v[1]) for n, v in gt_kps.items()},
        "zero_mask.jpg": {n: _predkp(v[0], v[1]) for n, v in gt_kps.items()},
        "bad_batch.jpg": {n: _predkp(v[0], v[1]) for n, v in gt_kps.items()},
        # 'no_pred.jpg' deliberately absent.
    }
    records = predict_weights_via_schaeffer(
        pred, samples,
        sticker_area_px_by_filename={
            "no_label.jpg": 400, "zero_mask.jpg": 0, "bad_batch.jpg": 400,
            # no_pred.jpg and no_mask.jpg deliberately absent → None lookup
        },
        sticker_area_cm2_by_batch={"B3": 100.0},  # B2 deliberately absent
    )
    by_name = {r.sample.image_path.name: r for r in records}
    assert "no weight label" in by_name["no_label.jpg"].skip_reason
    assert "no model prediction" in by_name["no_pred.jpg"].skip_reason
    assert "mask file missing" in by_name["no_mask.jpg"].skip_reason
    assert "zero sticker pixels" in by_name["zero_mask.jpg"].skip_reason
    assert "B2" in by_name["bad_batch.jpg"].skip_reason


def test_predict_via_schaeffer_low_conf_drops_keypoint():
    """conf_threshold filters predicted keypoints out of the formula."""
    gt_kps = _balanced_geometry_kps()
    sample = _sample(name="a.jpg", keypoints=gt_kps, weight_kg=200.0)
    # All predictions below threshold → schaeffer_from_keypoints returns None
    # → record skipped with 'keypoints insufficient'.
    pred = {"a.jpg": {n: _predkp(v[0], v[1], conf=0.1) for n, v in gt_kps.items()}}
    records = predict_weights_via_schaeffer(
        pred, [sample],
        sticker_area_px_by_filename={"a.jpg": 400},
        sticker_area_cm2_by_batch={"B3": 100.0},
        conf_threshold=0.5,
    )
    assert records[0].predicted_weight_kg is None
    assert "insufficient" in records[0].skip_reason
    # px_per_cm IS still computed (we know the scale even when keypoints fail).
    assert records[0].px_per_cm == pytest.approx(2.0)


def test_predict_via_schaeffer_uses_per_batch_sticker():
    """Same GT, same predicted keypoints, different sticker cm² → different weights."""
    gt_kps = _balanced_geometry_kps()
    b3_sample = _sample(name="b3.jpg", keypoints=gt_kps, weight_kg=200.0, batch="B3")
    b4_sample = _sample(name="b4.jpg", keypoints=gt_kps, weight_kg=200.0, batch="B4")
    pred = {n: {kp: _predkp(v[0], v[1]) for kp, v in gt_kps.items()} for n in ("b3.jpg", "b4.jpg")}
    records = predict_weights_via_schaeffer(
        pred, [b3_sample, b4_sample],
        sticker_area_px_by_filename={"b3.jpg": 400, "b4.jpg": 400},
        sticker_area_cm2_by_batch={"B3": 100.0, "B4": 400.0},  # 4× larger sticker for B4
    )
    by_name = {r.sample.image_path.name: r for r in records}
    # Sticker cm² ×4 → px_per_cm halved → cm distances doubled → weight ×8.
    assert by_name["b4.jpg"].predicted_weight_kg == pytest.approx(
        8.0 * by_name["b3.jpg"].predicted_weight_kg, rel=1e-9,
    )


# ------------------------------------------------ build_eval_report


def test_build_eval_report_combines_both_metric_layers():
    """End-to-end: report contains pixel errors + weight metrics + config echo."""
    gt_kps = _balanced_geometry_kps()
    expected_kg = schaeffer_weight_kg(50.0 * math.pi, 100.0)
    sample = _sample(name="x.jpg", keypoints=gt_kps, weight_kg=expected_kg, batch="B3")
    pred = {"x.jpg": {n: _predkp(v[0] + 1.0, v[1]) for n, v in gt_kps.items()}}  # 1-px shift

    report = build_eval_report(
        predictions=pred, samples=[sample],
        sticker_area_px_by_filename={"x.jpg": 400},
        sticker_area_cm2_by_batch={"B3": 100.0},
    )

    assert report["n_samples"] == 1
    assert report["n_with_prediction"] == 1
    # Pixel-error block has one entry per canonical keypoint.
    for kp in CANONICAL_SIDE_KEYPOINTS:
        assert kp in report["pixel_errors_by_keypoint"]
    # Predicted weight should be near GT (small pixel shift = small kg shift).
    assert report["weight_overall"]["n"] == 1
    assert abs(report["weight_overall"]["bias_kg"]) < 30.0
    assert "B3" in report["weight_by_batch"]
    # Config block echoes the inputs.
    assert report["config"]["sticker_area_cm2_by_batch"] == {"B3": 100.0}
    assert report["config"]["girth_multiplier"] == pytest.approx(math.pi)


def test_build_eval_report_aggregates_skip_reasons():
    gt_kps = _balanced_geometry_kps()
    samples = [
        _sample(name="ok.jpg", keypoints=gt_kps, weight_kg=200.0),
        _sample(name="no_label.jpg", keypoints=gt_kps, weight_kg=None),
    ]
    pred = {n: {kp: _predkp(v[0], v[1]) for kp, v in gt_kps.items()} for n in ("ok.jpg", "no_label.jpg")}
    report = build_eval_report(
        predictions=pred, samples=samples,
        sticker_area_px_by_filename={"ok.jpg": 400, "no_label.jpg": 400},
        sticker_area_cm2_by_batch={"B3": 100.0},
    )
    assert report["skip_reasons"].get("no weight label") == 1
    assert report["n_with_prediction"] == 1
