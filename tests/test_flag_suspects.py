"""Tests for the suspect-sample flagger."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from cattle_phenotyping.data.kaggle import FilenameMeta, KaggleSample
from cattle_phenotyping.eval.baseline_schaeffer import SchaefferRecord
from cattle_phenotyping.eval.flag_suspects import (
    FLAG_CROSS_BATCH_ID_COLLISION,
    FLAG_IMPLAUSIBLE_LOW_WEIGHT,
    FLAG_LARGE_RESIDUAL,
    SUSPECT_CSV_COLUMNS,
    SuspectRow,
    build_suspect_rows,
    detect_cross_batch_id_collisions,
    detect_implausible_low_weight,
    detect_large_residual,
    residual_stdev,
    write_suspect_csv,
)


# -------------------------------------------------------------- helpers


def _sample(*, animal_id="1", batch="B3", weight_kg=200.0, name="img.jpg") -> KaggleSample:
    return KaggleSample(
        image_path=Path(name),
        batch=batch,  # type: ignore[arg-type]
        view="side",
        coco_image_id=1, coco_category_id=1,
        keypoints={}, bbox=None,
        image_width=1900, image_height=1425,
        filename_meta=FilenameMeta(
            animal_id=animal_id, view="side",
            weight_kg=weight_kg, sex="F",  # type: ignore[arg-type]
        ),
    )


def _record(
    *,
    residual_kg: float | None = None,
    predicted_kg: float | None = None,
    animal_id: str = "1",
    batch: str = "B3",
    name: str = "img.jpg",
    weight_kg: float = 200.0,
) -> SchaefferRecord:
    sample = _sample(animal_id=animal_id, batch=batch, weight_kg=weight_kg, name=name)
    return SchaefferRecord(
        sample=sample,
        labelled_weight_kg=weight_kg,
        predicted_weight_kg=predicted_kg,
        sticker_area_px=400,
        px_per_cm=2.0,
        residual_kg=residual_kg,
    )


# -------------------------------------------------------------- residual_stdev


def test_residual_stdev_requires_two_predictions():
    assert residual_stdev([]) is None
    assert residual_stdev([_record(residual_kg=10.0, predicted_kg=210.0)]) is None


def test_residual_stdev_skips_unresolved_records():
    records = [
        _record(residual_kg=10.0, predicted_kg=210.0),
        _record(residual_kg=-10.0, predicted_kg=190.0),
        _record(residual_kg=None, predicted_kg=None),  # skipped
    ]
    sd = residual_stdev(records)
    assert sd is not None
    assert sd == pytest.approx(10.0 * (2 ** 0.5), rel=1e-9)


# --------------------------------------------------- per-record flag detectors


def test_detect_large_residual_thresholds_on_n_sigma():
    rec = _record(residual_kg=30.0, predicted_kg=230.0)
    assert detect_large_residual(rec, stdev_kg=10.0, n_sigma=2.5) is True
    assert detect_large_residual(rec, stdev_kg=10.0, n_sigma=3.5) is False


def test_detect_large_residual_handles_missing_residual():
    rec = _record(residual_kg=None, predicted_kg=None)
    assert detect_large_residual(rec, stdev_kg=10.0, n_sigma=2.5) is False


def test_detect_large_residual_handles_zero_stdev():
    rec = _record(residual_kg=30.0, predicted_kg=230.0)
    assert detect_large_residual(rec, stdev_kg=0.0, n_sigma=2.5) is False


def test_detect_implausible_low_weight():
    rec_low = _record(residual_kg=-195.0, predicted_kg=5.0)
    rec_ok = _record(residual_kg=20.0, predicted_kg=220.0)
    rec_none = _record(residual_kg=None, predicted_kg=None)
    assert detect_implausible_low_weight(rec_low, threshold_kg=50.0) is True
    assert detect_implausible_low_weight(rec_ok, threshold_kg=50.0) is False
    assert detect_implausible_low_weight(rec_none, threshold_kg=50.0) is False


# ---------------------------------------------------- cross-batch collisions


def test_cross_batch_collision_flagged_when_weights_disagree():
    # Animal "314" in B3 = 325 kg, in B4 = 181 kg → ~80% disagreement.
    train = [_sample(animal_id="314", batch="B3", weight_kg=325.0, name="314_b3.jpg")]
    test = [_sample(animal_id="314", batch="B4", weight_kg=181.0, name="314_b4.jpg")]
    suspects = detect_cross_batch_id_collisions(
        {"train": train, "test": test}, disagreement_pct=20.0,
    )
    assert suspects == {"314"}


def test_cross_batch_within_threshold_not_flagged():
    # Same animal_id across batches but labels agree within 5%.
    train = [_sample(animal_id="5", batch="B3", weight_kg=200.0, name="5_b3.jpg")]
    val = [_sample(animal_id="5", batch="B4", weight_kg=205.0, name="5_b4.jpg")]
    assert detect_cross_batch_id_collisions(
        {"train": train, "val": val}, disagreement_pct=20.0,
    ) == set()


def test_cross_batch_same_batch_only_not_flagged():
    # Animal "9" appears 3× in B3 but never in another batch.
    train = [
        _sample(animal_id="9", batch="B3", weight_kg=200.0, name="9_a.jpg"),
        _sample(animal_id="9", batch="B3", weight_kg=210.0, name="9_b.jpg"),
        _sample(animal_id="9", batch="B3", weight_kg=190.0, name="9_c.jpg"),
    ]
    assert detect_cross_batch_id_collisions(
        {"train": train}, disagreement_pct=20.0,
    ) == set()


def test_cross_batch_uses_median_per_batch():
    # B3 median 200, B4 median 250 → 25% disagreement → flagged at 20%, not at 30%.
    train = [
        _sample(animal_id="42", batch="B3", weight_kg=180.0, name="42_a.jpg"),
        _sample(animal_id="42", batch="B3", weight_kg=200.0, name="42_b.jpg"),
        _sample(animal_id="42", batch="B3", weight_kg=220.0, name="42_c.jpg"),
        _sample(animal_id="42", batch="B4", weight_kg=250.0, name="42_d.jpg"),
    ]
    assert detect_cross_batch_id_collisions(
        {"train": train}, disagreement_pct=20.0,
    ) == {"42"}
    assert detect_cross_batch_id_collisions(
        {"train": train}, disagreement_pct=30.0,
    ) == set()


def test_cross_batch_ignores_samples_with_no_weight():
    train = [_sample(animal_id="100", batch="B3", weight_kg=200.0, name="a.jpg")]
    train[0] = KaggleSample(  # rebuild with None weight via a fresh sample
        image_path=Path("a.jpg"),
        batch="B3", view="side",
        coco_image_id=1, coco_category_id=1,
        keypoints={}, bbox=None,
        image_width=10, image_height=10,
        filename_meta=FilenameMeta(animal_id="100", view="side", weight_kg=None, sex=None),
    )
    val = [_sample(animal_id="100", batch="B4", weight_kg=300.0, name="b.jpg")]
    # Only one batch has a usable weight, so no cross-batch comparison happens.
    assert detect_cross_batch_id_collisions(
        {"train": train, "val": val}, disagreement_pct=20.0,
    ) == set()


# -------------------------------------------------------- build_suspect_rows


def test_build_suspect_rows_emits_only_flagged():
    records = {
        "train": [
            # clean: small residual, plausible prediction → no flag
            _record(residual_kg=2.0, predicted_kg=202.0, name="clean.jpg", animal_id="A"),
            # large residual
            _record(residual_kg=50.0, predicted_kg=250.0, name="huge_residual.jpg", animal_id="B"),
            # implausible low weight
            _record(residual_kg=-190.0, predicted_kg=10.0, name="ghost.jpg", animal_id="C"),
        ],
        "val": [],
    }
    rows = build_suspect_rows(
        records_by_split=records, stdev_kg=10.0, n_sigma=2.5,
        min_plausible_kg=50.0, cross_batch_suspect_ids=set(),
    )
    names = {row.image_filename for row in rows}
    assert names == {"huge_residual.jpg", "ghost.jpg"}


def test_build_suspect_rows_composes_multiple_flags():
    # One sample that's both a large residual AND has a cross-batch collision.
    records = {"train": [_record(
        residual_kg=80.0, predicted_kg=280.0, animal_id="314", name="x.jpg",
    )]}
    rows = build_suspect_rows(
        records_by_split=records, stdev_kg=10.0, n_sigma=2.5,
        min_plausible_kg=50.0, cross_batch_suspect_ids={"314"},
    )
    assert len(rows) == 1
    assert set(rows[0].flags) == {FLAG_LARGE_RESIDUAL, FLAG_CROSS_BATCH_ID_COLLISION}


def test_build_suspect_rows_records_sigma_z_with_sign():
    records = {"train": [
        _record(residual_kg=-30.0, predicted_kg=170.0, name="under.jpg"),
        _record(residual_kg=30.0, predicted_kg=230.0, name="over.jpg"),
    ]}
    rows = build_suspect_rows(
        records_by_split=records, stdev_kg=10.0, n_sigma=2.5,
        min_plausible_kg=50.0, cross_batch_suspect_ids=set(),
    )
    by_name = {r.image_filename: r for r in rows}
    assert by_name["under.jpg"].residual_sigma_z == pytest.approx(-3.0)
    assert by_name["over.jpg"].residual_sigma_z == pytest.approx(3.0)


def test_build_suspect_rows_skips_when_residual_missing_but_collision_hits():
    """Sample with no Schaeffer prediction still emitted if it has a collision flag."""
    records = {"train": [_record(
        residual_kg=None, predicted_kg=None, animal_id="314", name="masked_out.jpg",
    )]}
    rows = build_suspect_rows(
        records_by_split=records, stdev_kg=10.0, n_sigma=2.5,
        min_plausible_kg=50.0, cross_batch_suspect_ids={"314"},
    )
    assert len(rows) == 1
    assert rows[0].flags == [FLAG_CROSS_BATCH_ID_COLLISION]
    assert rows[0].residual_sigma_z is None


def test_build_suspect_rows_zero_stdev_disables_residual_flag():
    """If train residuals are all identical, large-residual flag never fires."""
    records = {"train": [_record(residual_kg=100.0, predicted_kg=300.0, name="x.jpg")]}
    rows = build_suspect_rows(
        records_by_split=records, stdev_kg=0.0, n_sigma=2.5,
        min_plausible_kg=50.0, cross_batch_suspect_ids=set(),
    )
    # No flags trigger → no rows.
    assert rows == []


# -------------------------------------------------------------- CSV writing


def test_write_suspect_csv_round_trip(tmp_path):
    rows = [
        SuspectRow(
            image_filename="20_s_104_M.jpg",
            split="train", batch="B3", animal_id="20",
            labelled_weight_kg=104.0, predicted_weight_kg=1.41,
            residual_kg=-102.59, residual_sigma_z=-3.4,
            sticker_area_px=10132, px_per_cm=25.76,
            flags=[FLAG_LARGE_RESIDUAL, FLAG_IMPLAUSIBLE_LOW_WEIGHT],
        ),
        SuspectRow(
            image_filename="314_b4-3_s_181_F.jpg",
            split="val", batch="B4", animal_id="314",
            labelled_weight_kg=181.0, predicted_weight_kg=None,
            residual_kg=None, residual_sigma_z=None,
            sticker_area_px=None, px_per_cm=None,
            flags=[FLAG_CROSS_BATCH_ID_COLLISION],
        ),
    ]
    out_path = tmp_path / "suspect.csv"
    n = write_suspect_csv(rows, out_path)
    assert n == 2

    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert tuple(reader.fieldnames or ()) == SUSPECT_CSV_COLUMNS
        loaded = list(reader)
    assert len(loaded) == 2
    assert loaded[0]["image_filename"] == "20_s_104_M.jpg"
    assert loaded[0]["flags"] == "large_residual|implausible_low_weight"
    assert loaded[1]["predicted_weight_kg"] == ""  # None → empty cell
    assert loaded[1]["flags"] == "cross_batch_id_collision"


def test_write_suspect_csv_creates_parent_dirs(tmp_path):
    out_path = tmp_path / "nested" / "deeper" / "suspect.csv"
    write_suspect_csv([], out_path)
    assert out_path.exists()
