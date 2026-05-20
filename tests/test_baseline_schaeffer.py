"""Tests for the forward Schaeffer baseline evaluator."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from cattle_phenotyping.data.kaggle import FilenameMeta, KaggleSample
from cattle_phenotyping.eval.baseline_schaeffer import (
    DEFAULT_STICKER_AREA_CM2,
    SchaefferRecord,
    compute_metrics,
    evaluate_sample,
    flag_outliers,
    group_metrics,
    load_split_filenames,
)
from cattle_phenotyping.models.schaeffer import schaeffer_weight_kg


# --------------------------------------------------------- helpers


def _kp(x, y, v=2):
    return (x, y, v)


def _make_sample(
    *,
    kps: dict | None = None,
    weight_kg: float | None = 200.0,
    batch: str = "B3",
    sex: str = "F",
    animal_id: str = "1",
    image_name: str = "img.jpg",
) -> KaggleSample:
    return KaggleSample(
        image_path=Path(image_name),
        batch=batch,  # type: ignore[arg-type]
        view="side",
        coco_image_id=1,
        coco_category_id=1,
        keypoints=kps or {
            "front_girth_top": _kp(0, 0),
            "front_girth_bottom": _kp(0, 200),
            "shoulderbone": _kp(0, 0),
            "pinbone": _kp(400, 0),
        },
        bbox=None,
        image_width=1900,
        image_height=1425,
        filename_meta=FilenameMeta(
            animal_id=animal_id, view="side",
            weight_kg=weight_kg, sex=sex,  # type: ignore[arg-type]
        ),
    )


# ------------------------------------------------ split CSV loading


def test_load_split_filenames_extracts_image_filename_column(tmp_path):
    csv_path = tmp_path / "test.csv"
    csv_path.write_text(
        "image_filename,batch,animal_id\n"
        "9_s_144_M.jpg,B3,9\n"
        "12_s_200_F.jpg,B3,12\n"
        "100_b4-1_s_124_F.jpg,B4,100\n",
        encoding="utf-8",
    )
    names = load_split_filenames(csv_path)
    assert names == {"9_s_144_M.jpg", "12_s_200_F.jpg", "100_b4-1_s_124_F.jpg"}


def test_load_split_filenames_missing_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="image_filename"):
        load_split_filenames(csv_path)


# ----------------------------------------------- evaluate_sample


def test_evaluate_sample_round_trip_known_geometry():
    """Forward-Schaeffer on synthetic data should recover the cm geometry exactly."""
    # Set up: chord 100 px, length 200 px, with sticker area 400 px and
    # sticker_area_cm² = 100. Then px_per_cm = sqrt(400/100) = 2.
    # In cm: chord = 50, length = 100. HG = 50π ≈ 157.08. weight_kg by Schaeffer.
    kps = {
        "front_girth_top": _kp(0, 0),
        "front_girth_bottom": _kp(0, 100),
        "shoulderbone": _kp(0, 0),
        "pinbone": _kp(200, 0),
    }
    expected_weight = schaeffer_weight_kg(50 * math.pi, 100)
    sample = _make_sample(kps=kps, weight_kg=expected_weight)

    record = evaluate_sample(
        sample,
        sticker_area_cm2=100.0,
        sticker_area_px=400,
        girth_multiplier=math.pi,
    )

    assert record.predicted_weight_kg == pytest.approx(expected_weight, rel=1e-9)
    assert record.px_per_cm == pytest.approx(2.0, rel=1e-9)
    assert record.residual_kg == pytest.approx(0.0, abs=1e-9)
    assert record.skip_reason == ""


def test_evaluate_sample_missing_mask_records_skip_reason():
    sample = _make_sample()
    rec = evaluate_sample(sample, sticker_area_cm2=18.21, sticker_area_px=None)
    assert rec.predicted_weight_kg is None
    assert rec.residual_kg is None
    assert "mask" in rec.skip_reason


def test_evaluate_sample_zero_sticker_pixels_skips():
    sample = _make_sample()
    rec = evaluate_sample(sample, sticker_area_cm2=18.21, sticker_area_px=0)
    assert rec.predicted_weight_kg is None
    assert "zero sticker pixels" in rec.skip_reason


def test_evaluate_sample_invisible_keypoints_skips():
    kps = {
        "front_girth_top": _kp(0, 0, v=0),
        "front_girth_bottom": _kp(0, 100, v=0),
        "rear_girth_top": _kp(0, 0, v=0),
        "rear_girth_bottom": _kp(0, 100, v=0),
        "shoulderbone": _kp(0, 0),
        "pinbone": _kp(200, 0),
    }
    sample = _make_sample(kps=kps)
    rec = evaluate_sample(sample, sticker_area_cm2=18.21, sticker_area_px=400)
    assert rec.predicted_weight_kg is None
    assert "keypoints" in rec.skip_reason
    # Even when prediction fails, px_per_cm is still computed and recorded.
    assert rec.px_per_cm == pytest.approx(math.sqrt(400 / 18.21), rel=1e-9)


def test_evaluate_sample_no_weight_label_skips():
    sample = _make_sample(weight_kg=None)
    rec = evaluate_sample(sample, sticker_area_cm2=18.21, sticker_area_px=400)
    assert rec.predicted_weight_kg is None
    assert "weight" in rec.skip_reason


# --------------------------------------------------- aggregation


def _record_with_residual(residual_kg: float, *, batch: str = "B3", sex: str = "F") -> SchaefferRecord:
    """Build a record with a known residual + true weight 200 kg."""
    sample = _make_sample(batch=batch, sex=sex)
    true_w = 200.0
    pred = true_w + residual_kg
    return SchaefferRecord(
        sample=sample,
        labelled_weight_kg=true_w,
        predicted_weight_kg=pred,
        sticker_area_px=400,
        px_per_cm=2.0,
        residual_kg=residual_kg,
    )


def test_compute_metrics_basic_arithmetic():
    records = [
        _record_with_residual(10.0),   # |err|=10, sq=100
        _record_with_residual(-20.0),  # |err|=20, sq=400
        _record_with_residual(0.0),    # |err|=0,  sq=0
    ]
    m = compute_metrics(records)
    assert m is not None
    assert m.n == 3
    assert m.mae_kg == pytest.approx(30.0 / 3)
    assert m.rmse_kg == pytest.approx(math.sqrt(500.0 / 3))
    # bias is mean(pred - true) = mean(10, -20, 0) = -10/3
    assert m.bias_kg == pytest.approx(-10.0 / 3)
    assert m.mean_true_kg == pytest.approx(200.0)


def test_compute_metrics_returns_none_on_empty():
    assert compute_metrics([]) is None
    # Only-skipped also gives None.
    skipped = SchaefferRecord(
        sample=_make_sample(), labelled_weight_kg=200,
        predicted_weight_kg=None, sticker_area_px=None,
        px_per_cm=None, residual_kg=None, skip_reason="x",
    )
    assert compute_metrics([skipped]) is None


def test_compute_metrics_r2_perfect_predictions():
    # All predictions exactly match labels — but labels need variance for R² to be defined.
    rec_perfect = [
        SchaefferRecord(
            sample=_make_sample(weight_kg=w),
            labelled_weight_kg=w,
            predicted_weight_kg=w,
            sticker_area_px=400,
            px_per_cm=2.0,
            residual_kg=0.0,
        )
        for w in (150.0, 200.0, 250.0)
    ]
    m = compute_metrics(rec_perfect)
    assert m is not None
    assert m.r2 == pytest.approx(1.0)


def test_group_metrics_splits_by_batch():
    records = [
        _record_with_residual(5.0, batch="B3"),
        _record_with_residual(10.0, batch="B3"),
        _record_with_residual(-15.0, batch="B4"),
    ]
    by_batch = group_metrics(records, key="batch")
    assert set(by_batch.keys()) == {"B3", "B4"}
    assert by_batch["B3"].n == 2
    assert by_batch["B4"].n == 1
    assert by_batch["B3"].mae_kg == pytest.approx(7.5)


def test_group_metrics_by_sex():
    records = [
        _record_with_residual(5.0, sex="F"),
        _record_with_residual(-5.0, sex="F"),
        _record_with_residual(20.0, sex="M"),
    ]
    by_sex = group_metrics(records, key="sex")
    assert by_sex["F"].n == 2
    assert by_sex["M"].n == 1


def test_group_metrics_unknown_key_raises():
    with pytest.raises(ValueError, match="Unknown group key"):
        group_metrics([], key="not_a_key")


# ---------------------------------------------------------- outliers


def test_flag_outliers_thresholds_at_n_sigma():
    # 9 small residuals + 1 huge one — the huge one must come out as outlier.
    records = [_record_with_residual(2.0) for _ in range(9)]
    records.append(_record_with_residual(50.0))
    outliers = flag_outliers(records, n_sigma=2.0)
    assert len(outliers) == 1
    assert outliers[0].residual_kg == pytest.approx(50.0)


def test_flag_outliers_top_k_caps_count():
    records = [_record_with_residual(float(r)) for r in [-100, -50, 5, 10, 50, 200]]
    out = flag_outliers(records, n_sigma=0.0, top_k=3)
    assert len(out) == 3
    # Sorted by absolute residual, largest first.
    assert abs(out[0].residual_kg) >= abs(out[-1].residual_kg)


def test_flag_outliers_empty_input():
    assert flag_outliers([]) == []
    assert flag_outliers([SchaefferRecord(
        sample=_make_sample(),
        labelled_weight_kg=200,
        predicted_weight_kg=None,
        sticker_area_px=None,
        px_per_cm=None,
        residual_kg=None,
    )]) == []


def test_default_sticker_constant_matches_calibration():
    """Pin the calibration value so a future drift causes a test failure."""
    assert DEFAULT_STICKER_AREA_CM2 == pytest.approx(18.21)


# --------------------------------------------- per-batch sticker calibration


def test_evaluate_sample_uses_per_batch_mapping():
    """A {batch: cm²} mapping should produce different scales per batch.

    Geometry held fixed: chord 100 px, length 200 px, sticker area 400 px.
    With cm²=100 → px_per_cm=2; with cm²=400 → px_per_cm=1. The latter
    doubles cm distances, which raises Schaeffer's HG²×BL output by 8×.
    """
    kps = {
        "front_girth_top": _kp(0, 0),
        "front_girth_bottom": _kp(0, 100),
        "shoulderbone": _kp(0, 0),
        "pinbone": _kp(200, 0),
    }
    b3 = _make_sample(kps=kps, weight_kg=100.0, batch="B3")
    b4 = _make_sample(kps=kps, weight_kg=100.0, batch="B4")

    by_batch = {"B3": 100.0, "B4": 400.0}
    rec_b3 = evaluate_sample(b3, sticker_area_cm2=by_batch, sticker_area_px=400)
    rec_b4 = evaluate_sample(b4, sticker_area_cm2=by_batch, sticker_area_px=400)

    assert rec_b3.predicted_weight_kg is not None
    assert rec_b4.predicted_weight_kg is not None
    # cm²×4 → px_per_cm halved → cm distances doubled → weight × 2³ = 8.
    assert rec_b4.predicted_weight_kg == pytest.approx(8.0 * rec_b3.predicted_weight_kg, rel=1e-9)


def test_evaluate_sample_missing_batch_in_mapping_skips():
    sample = _make_sample(batch="B2")
    rec = evaluate_sample(
        sample, sticker_area_cm2={"B3": 15.0, "B4": 80.0}, sticker_area_px=400,
    )
    assert rec.predicted_weight_kg is None
    assert "B2" in rec.skip_reason
    assert "no sticker cm" in rec.skip_reason.lower()


def test_evaluate_sample_scalar_still_works():
    """Scalar path is unchanged for backwards compat."""
    sample = _make_sample(weight_kg=200.0)
    rec = evaluate_sample(sample, sticker_area_cm2=18.21, sticker_area_px=400)
    assert rec.predicted_weight_kg is not None
    assert rec.px_per_cm == pytest.approx(math.sqrt(400 / 18.21), rel=1e-9)
