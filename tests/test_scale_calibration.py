"""Tests for the sticker / scale back-derivation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cattle_phenotyping.data.kaggle import FilenameMeta, KaggleSample
from cattle_phenotyping.models.schaeffer import (
    schaeffer_from_keypoints,
    schaeffer_weight_kg,
)
from cattle_phenotyping.pipeline.scale_calibration import (
    DEFAULT_COLOR_TOLERANCE,
    STICKER_RGB_BY_BATCH,
    aggregate_sticker_size,
    aggregate_sticker_size_by_batch,
    invert_schaeffer_for_px_per_cm,
    iter_sample_scales,
    per_sample_scale,
    sticker_area_px_in_mask,
)


# ----------------------------------------------------- math: inversion round-trip


@pytest.mark.parametrize(
    "chord_cm,length_cm,multiplier",
    [
        (50.0, 100.0, math.pi),
        (60.0, 130.0, math.pi),
        (45.0, 120.0, 2.0),  # elliptical-body multiplier
        (52.5, 145.0, 2.5),
    ],
)
def test_inversion_round_trips(chord_cm, length_cm, multiplier):
    """If we forward-compute weight from cm and then invert, px_per_cm = 1.0."""
    hg_cm = chord_cm * multiplier
    weight = schaeffer_weight_kg(hg_cm, length_cm)
    # chord and length in "pixels" = cm × 1.0 (scale 1) — invert should give 1.0.
    pxcm = invert_schaeffer_for_px_per_cm(
        chord_cm, length_cm, weight,
        girth_chord_to_circumference=multiplier,
    )
    assert pxcm == pytest.approx(1.0, rel=1e-9)


def test_inversion_scales_correctly():
    """Doubling all pixel distances must double the recovered px_per_cm."""
    chord_cm = 50.0
    length_cm = 100.0
    weight = schaeffer_weight_kg(chord_cm * math.pi, length_cm)

    pxcm_unit = invert_schaeffer_for_px_per_cm(chord_cm, length_cm, weight)
    pxcm_double = invert_schaeffer_for_px_per_cm(chord_cm * 2, length_cm * 2, weight)
    assert pxcm_double == pytest.approx(2 * pxcm_unit, rel=1e-9)


def test_inversion_rejects_bad_inputs():
    with pytest.raises(ValueError, match="positive"):
        invert_schaeffer_for_px_per_cm(0.0, 100.0, 200.0)
    with pytest.raises(ValueError, match="positive"):
        invert_schaeffer_for_px_per_cm(50.0, 100.0, -5.0)


# ----------------------------------------------- per-sample scale via Schaeffer


def _kp(x, y, v=2):
    return (x, y, v)


def _make_sample(
    keypoints: dict,
    weight_kg: float | None = 150.0,
    batch: str = "B3",
    view: str = "side",
) -> KaggleSample:
    return KaggleSample(
        image_path=Path("dummy.jpg"),
        batch=batch,  # type: ignore[arg-type]
        view=view,  # type: ignore[arg-type]
        coco_image_id=1, coco_category_id=1,
        keypoints=keypoints, bbox=None,
        image_width=2000, image_height=1500,
        filename_meta=FilenameMeta(animal_id="1", view=view, weight_kg=weight_kg, sex="F"),  # type: ignore[arg-type]
    )


def test_per_sample_consistent_with_schaeffer_forward():
    """Round-trip: invert to get px_per_cm, then forward-Schaeffer must hit weight."""
    kps = {
        "front_girth_top": _kp(100.0, 100.0),
        "front_girth_bottom": _kp(100.0, 300.0),  # chord 200 px
        "shoulderbone": _kp(0.0, 500.0),
        "pinbone": _kp(400.0, 500.0),  # length 400 px
    }
    target_weight = 180.0
    sample = _make_sample(kps, weight_kg=target_weight)
    result = per_sample_scale(sample)
    assert result.px_per_cm is not None
    forward = schaeffer_from_keypoints(kps, result.px_per_cm)
    assert forward == pytest.approx(target_weight, rel=1e-9)


def test_per_sample_falls_back_to_rear_girth():
    kps = {
        "rear_girth_top": _kp(100.0, 100.0),
        "rear_girth_bottom": _kp(100.0, 250.0),
        "shoulderbone": _kp(0.0, 500.0),
        "pinbone": _kp(400.0, 500.0),
        # Front girth deliberately invisible.
        "front_girth_top": _kp(0.0, 0.0, v=0),
        "front_girth_bottom": _kp(0.0, 0.0, v=0),
    }
    result = per_sample_scale(_make_sample(kps, weight_kg=150.0))
    assert result.px_per_cm is not None
    assert result.chord_px == pytest.approx(150.0)


def test_per_sample_skipped_when_no_weight():
    kps = {
        "front_girth_top": _kp(0, 0),
        "front_girth_bottom": _kp(0, 100),
        "shoulderbone": _kp(0, 0),
        "pinbone": _kp(200, 0),
    }
    result = per_sample_scale(_make_sample(kps, weight_kg=None))
    assert result.px_per_cm is None
    assert "weight" in result.reason


def test_per_sample_skipped_when_keypoints_invisible():
    kps = {
        "front_girth_top": _kp(0, 0, v=0),
        "front_girth_bottom": _kp(0, 100, v=0),
        "rear_girth_top": _kp(0, 0, v=0),
        "rear_girth_bottom": _kp(0, 100, v=0),
        "shoulderbone": _kp(0, 0),
        "pinbone": _kp(200, 0),
    }
    result = per_sample_scale(_make_sample(kps))
    assert result.px_per_cm is None
    assert "girth" in result.reason


def test_iter_sample_scales_yields_all_input_samples():
    s1 = _make_sample({
        "front_girth_top": _kp(0, 0),
        "front_girth_bottom": _kp(0, 100),
        "shoulderbone": _kp(0, 0),
        "pinbone": _kp(200, 0),
    })
    s2 = _make_sample({}, weight_kg=120.0)  # no keypoints → unresolved
    results = list(iter_sample_scales([s1, s2]))
    assert len(results) == 2
    assert results[0].px_per_cm is not None
    assert results[1].px_per_cm is None


# --------------------------------------------------- mask color thresholding


@pytest.fixture
def mask_array_b3_with_sticker():
    """200×200 RGB mask: a 40×40 blue sticker patch on a uniform background."""
    arr = np.full((200, 200, 3), fill_value=128, dtype=np.uint8)  # neutral background
    arr[80:120, 80:120] = STICKER_RGB_BY_BATCH["B3"]  # blue sticker block 40×40 = 1600 px
    return arr


def test_sticker_area_counts_matching_pixels(mask_array_b3_with_sticker):
    area = sticker_area_px_in_mask(mask_array_b3_with_sticker, batch="B3")
    assert area == 40 * 40


def test_sticker_area_tolerance_handles_anti_aliasing():
    arr = np.full((10, 10, 3), fill_value=128, dtype=np.uint8)
    # Place a sticker block whose colour drifts by exactly the tolerance.
    drift = DEFAULT_COLOR_TOLERANCE
    blue = STICKER_RGB_BY_BATCH["B3"]
    arr[2:7, 2:7] = (blue[0] + drift, blue[1] - drift, min(255, blue[2]))
    area = sticker_area_px_in_mask(arr, batch="B3")
    assert area == 25  # all 5×5 pixels still within tolerance


def test_sticker_area_zero_when_color_absent():
    arr = np.full((50, 50, 3), fill_value=0, dtype=np.uint8)  # all black
    assert sticker_area_px_in_mask(arr, batch="B3") == 0


def test_sticker_area_unknown_batch_raises():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="No sticker color"):
        sticker_area_px_in_mask(arr, batch="X1")


def test_sticker_area_wrong_shape_raises():
    arr = np.zeros((10, 10), dtype=np.uint8)  # grayscale, not RGB
    with pytest.raises(ValueError, match="HxWx3"):
        sticker_area_px_in_mask(arr, batch="B3")


def test_b2_uses_yellow_color():
    arr = np.full((10, 10, 3), fill_value=0, dtype=np.uint8)
    arr[2:5, 2:5] = STICKER_RGB_BY_BATCH["B2"]  # 3×3 yellow
    assert sticker_area_px_in_mask(arr, batch="B2") == 9
    # The same array should NOT match B3 (blue).
    assert sticker_area_px_in_mask(arr, batch="B3") == 0


# ---------------------------------------------------- Stage B aggregation


class _StubResult:
    """Lightweight stand-in so the aggregator test doesn't need full samples."""
    def __init__(self, px_per_cm):
        self.px_per_cm = px_per_cm


def test_aggregate_sticker_size_uses_median():
    # Construct paired (px_per_cm, sticker_px) that all imply a 100 cm² sticker.
    # area_cm² = sticker_px / px_per_cm²  ⇒  sticker_px = px_per_cm² × 100
    pairs = []
    for pxcm in (1.0, 2.0, 3.0, 4.0, 5.0):
        pairs.append((_StubResult(pxcm), int(round(pxcm * pxcm * 100))))
    est = aggregate_sticker_size(pairs)
    assert est is not None
    assert est.n_samples == 5
    assert est.median_area_cm2 == pytest.approx(100.0, abs=0.5)
    diameter = 2.0 * math.sqrt(100.0 / math.pi)
    assert est.median_diameter_cm == pytest.approx(diameter, abs=0.05)


def test_aggregate_sticker_size_drops_invalid_rows():
    pairs = [
        (_StubResult(None), 1600),    # px_per_cm missing
        (_StubResult(2.0), 0),        # zero-area sticker
        (_StubResult(2.0), 400),      # valid: 100 cm²
    ]
    est = aggregate_sticker_size(pairs)
    assert est is not None
    assert est.n_samples == 1
    assert est.median_area_cm2 == pytest.approx(100.0)


def test_aggregate_sticker_size_returns_none_when_empty():
    assert aggregate_sticker_size([]) is None
    # Only-invalid input also returns None.
    assert aggregate_sticker_size([(_StubResult(None), 100)]) is None


# ------------------------------------- Stage B aggregation grouped by batch


class _BatchedStubResult:
    """Stub with the bare attributes the per-batch aggregator inspects."""
    def __init__(self, px_per_cm, batch):
        self.px_per_cm = px_per_cm

        class _S:
            pass
        self.sample = _S()
        self.sample.batch = batch


def test_aggregate_by_batch_groups_independently():
    """B3 cm² ≈ 15 and B4 cm² ≈ 80 should not contaminate each other.

    Mirrors the 2026-05-20 baseline finding where the global median
    18.21 cm² was wrong for both batches because the dataset is bimodal.
    """
    pairs = []
    # 5 B3 samples implying a 15 cm² sticker.
    for pxcm in (1.0, 2.0, 3.0, 4.0, 5.0):
        pairs.append((_BatchedStubResult(pxcm, "B3"), int(round(pxcm * pxcm * 15))))
    # 5 B4 samples implying an 80 cm² sticker.
    for pxcm in (1.0, 2.0, 3.0, 4.0, 5.0):
        pairs.append((_BatchedStubResult(pxcm, "B4"), int(round(pxcm * pxcm * 80))))

    by_batch = aggregate_sticker_size_by_batch(pairs)
    assert set(by_batch.keys()) == {"B3", "B4"}
    assert by_batch["B3"].median_area_cm2 == pytest.approx(15.0, abs=0.5)
    assert by_batch["B4"].median_area_cm2 == pytest.approx(80.0, abs=0.5)
    assert by_batch["B3"].n_samples == 5
    assert by_batch["B4"].n_samples == 5


def test_aggregate_by_batch_skips_invalid_rows_per_bucket():
    pairs = [
        (_BatchedStubResult(None, "B3"), 100),   # invalid → dropped
        (_BatchedStubResult(2.0, "B3"), 60),     # 15 cm²
        (_BatchedStubResult(2.0, "B4"), 0),      # zero area → dropped
        (_BatchedStubResult(2.0, "B4"), 320),    # 80 cm²
    ]
    by_batch = aggregate_sticker_size_by_batch(pairs)
    assert by_batch["B3"].median_area_cm2 == pytest.approx(15.0)
    assert by_batch["B4"].median_area_cm2 == pytest.approx(80.0)


def test_aggregate_by_batch_empty_returns_empty_dict():
    assert aggregate_sticker_size_by_batch([]) == {}
    # Only-invalid rows produce no batches.
    assert aggregate_sticker_size_by_batch(
        [(_BatchedStubResult(None, "B3"), 100)]
    ) == {}


# ------------------------------------------------------ outlier identifiability


def test_outlier_sample_emerges_from_per_sample_pxcm():
    """If one sample has obviously wrong labelled weight, its px_per_cm is an outlier."""
    # Build 5 consistent samples + 1 with a 10× wrong weight label.
    good_kps = {
        "front_girth_top": _kp(0, 0),
        "front_girth_bottom": _kp(0, 200),
        "shoulderbone": _kp(0, 0),
        "pinbone": _kp(400, 0),
    }
    pxcm_consistent = invert_schaeffer_for_px_per_cm(200.0, 400.0, 200.0)

    samples = [_make_sample(good_kps, weight_kg=200.0) for _ in range(5)]
    samples.append(_make_sample(good_kps, weight_kg=20.0))  # 10× lighter than its geometry suggests
    results = list(iter_sample_scales(samples))
    pxcms = [r.px_per_cm for r in results if r.px_per_cm is not None]

    # The outlier should be ~10^(1/3) ≈ 2.15× above the others.
    consistent = pxcms[:5]
    outlier = pxcms[5]
    assert all(abs(p - pxcm_consistent) < 1e-9 for p in consistent)
    assert outlier > 2.0 * pxcm_consistent
