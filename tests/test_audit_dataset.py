"""Tests for the dataset auditor / animal_id proposer."""

import csv

from cattle_phenotyping.training import audit_dataset


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_animal_id_assignment_clusters_duplicates(tmp_path):
    labels = tmp_path / "labels.csv"
    out = tmp_path / "labels_with_id.csv"

    fieldnames = [
        "image_name", "weight", "body_length_cm", "withers_height_cm",
        "heart_girth_cm", "hip_length_cm", "bcs",
    ]
    rows = [
        # animal A — same measurements, two photos
        {"image_name": "1.png", "weight": "450", "body_length_cm": "150",
         "withers_height_cm": "120", "heart_girth_cm": "180", "hip_length_cm": "45", "bcs": "3.0"},
        {"image_name": "2.png", "weight": "450", "body_length_cm": "150",
         "withers_height_cm": "120", "heart_girth_cm": "180", "hip_length_cm": "45", "bcs": "3.0"},
        # animal B — different
        {"image_name": "3.png", "weight": "500", "body_length_cm": "155",
         "withers_height_cm": "122", "heart_girth_cm": "185", "hip_length_cm": "46", "bcs": "3.5"},
    ]
    _write_csv(labels, rows, fieldnames)

    rc = audit_dataset.main([
        "--labels", str(labels),
        "--propose-animal-id", str(out),
    ])
    assert rc == 0
    assert out.exists()

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        produced = list(reader)

    by_image = {row["image_name"]: row["animal_id"] for row in produced}
    assert by_image["1.png"] == by_image["2.png"], "duplicates should share an animal_id"
    assert by_image["3.png"] != by_image["1.png"], "non-duplicate should be a new animal_id"
    # Exactly two unique animals.
    assert len({row["animal_id"] for row in produced}) == 2


def test_handles_missing_data_dir_gracefully(tmp_path):
    labels = tmp_path / "labels.csv"
    fieldnames = [
        "image_name", "weight", "body_length_cm", "withers_height_cm",
        "heart_girth_cm", "hip_length_cm", "bcs",
    ]
    _write_csv(labels, [
        {"image_name": "1.png", "weight": "450", "body_length_cm": "150",
         "withers_height_cm": "120", "heart_girth_cm": "180", "hip_length_cm": "45", "bcs": "3.0"},
    ], fieldnames)

    rc = audit_dataset.main(["--labels", str(labels)])
    assert rc == 0
