"""Tests for the Kaggle train/val/test split builder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cattle_phenotyping.data.kaggle import KaggleSample
from cattle_phenotyping.training.build_splits import (
    _assign_strata,
    animal_weights,
    build_splits,
    filter_weight_samples,
    stratified_animal_split,
)


# ------------------------------------------------------------------ fixture helpers


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    path.write_text(_json.dumps(payload), encoding="utf-8")
    return path


def _b3_side_payload(triplets: list[tuple[int, str, float, str]]) -> dict:
    """Helper that builds a B3 Side COCO doc with N animals.

    Each ``(image_id, animal_id, weight, sex)`` becomes one image + one
    annotation with deterministic dummy keypoints.
    """
    return {
        "categories": [{
            "id": 1,
            "name": "b3_side",
            "supercategory": "cow",
            "keypoints": [
                "1_wither", "2_pinbone", "3_shoulderbone",
                "5_front_girth_bottom", "4_front_girth_top",
                "9_Height_bottom", "8_Height_top",
                "7_rear_girth_bottom", "6_rear_girth_top",
            ],
            "skeleton": [],
        }],
        "images": [
            {"id": img_id, "file_name": f"{aid}_s_{int(w)}_{sx}.jpg", "width": 4160, "height": 3120}
            for img_id, aid, w, sx in triplets
        ],
        "annotations": [
            {
                "id": img_id,
                "image_id": img_id,
                "category_id": 1,
                "num_keypoints": 9,
                "bbox": [100, 100, 800, 600],
                "keypoints": [float(img_id)] * 27,  # 9 kpts × 3 (x,y,v)
            }
            for img_id, *_ in triplets
        ],
    }


def _make_dataset(tmp_path: Path, n_animals: int = 40) -> Path:
    """Create a synthetic dataset root with B3 Side covering ``n_animals``.

    Weights are drawn deterministically across a 60-450 kg range so the
    quartile stratification has something to bite on.
    """
    triplets = []
    for i in range(1, n_animals + 1):
        # Sweep 60..450 kg roughly uniformly.
        weight = 60 + (390 * (i - 1) / max(n_animals - 1, 1))
        sex = "F" if i % 3 != 0 else "M"
        triplets.append((i, str(i), weight, sex))
    _write_json(
        tmp_path / "Vector/B3/Side/data/COCO_Side.json",
        _b3_side_payload(triplets),
    )
    return tmp_path


# ----------------------------------------------------------- filter / aggregate


def test_filter_excludes_b2_seq_only():
    from cattle_phenotyping.data.kaggle import FilenameMeta

    seq_sample = KaggleSample(
        image_path=Path("1_r.jpg"),
        batch="B2", view="rear",
        coco_image_id=1, coco_category_id=1,
        keypoints={}, bbox=None,
        image_width=100, image_height=100,
        filename_meta=FilenameMeta(animal_id=None, view="rear", is_b2_seq_only=True),
    )
    real_sample = KaggleSample(
        image_path=Path("9_s_144_M.jpg"),
        batch="B3", view="side",
        coco_image_id=2, coco_category_id=1,
        keypoints={}, bbox=None,
        image_width=100, image_height=100,
        filename_meta=FilenameMeta(animal_id="9", view="side", weight_kg=144.0, sex="M"),
    )
    kept = filter_weight_samples([seq_sample, real_sample])
    assert len(kept) == 1
    assert kept[0] is real_sample


def test_animal_weights_averages_per_animal():
    from cattle_phenotyping.data.kaggle import FilenameMeta

    def _sample(batch, aid, w):
        return KaggleSample(
            image_path=Path(f"{aid}_s_{int(w)}_F.jpg"),
            batch=batch, view="side",
            coco_image_id=0, coco_category_id=1,
            keypoints={}, bbox=None, image_width=10, image_height=10,
            filename_meta=FilenameMeta(animal_id=aid, view="side", weight_kg=w, sex="F"),
        )
    samples = [
        _sample("B3", "1", 100.0),
        _sample("B3", "1", 102.0),  # same animal, slight noise
        _sample("B3", "2", 200.0),
        _sample("B4", "1", 300.0),  # different batch -> different animal_key
    ]
    w = animal_weights(samples)
    assert w[("B3", "1")] == pytest.approx(101.0)
    assert w[("B3", "2")] == pytest.approx(200.0)
    assert w[("B4", "1")] == pytest.approx(300.0)


# ----------------------------------------------------------- stratification


def test_assign_strata_buckets_balanced():
    weights = {("B3", str(i)): float(i) for i in range(1, 21)}  # 20 animals
    strata = _assign_strata(weights)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for s in strata.values():
        counts[s] += 1
    # 20 / 4 strata = 5 each.
    assert counts == {0: 5, 1: 5, 2: 5, 3: 5}


def test_stratified_split_no_animal_crossover():
    weights = {("B3", str(i)): float(i) * 5 for i in range(1, 41)}  # 40 animals, 5–200 kg
    splits = stratified_animal_split(weights, seed=42)
    assert not (splits["train"] & splits["val"])
    assert not (splits["train"] & splits["test"])
    assert not (splits["val"] & splits["test"])
    assert splits["train"] | splits["val"] | splits["test"] == set(weights)


def test_stratified_split_is_deterministic():
    weights = {("B3", str(i)): float(i) * 3 for i in range(1, 51)}
    s1 = stratified_animal_split(weights, seed=42)
    s2 = stratified_animal_split(weights, seed=42)
    assert s1 == s2

    s3 = stratified_animal_split(weights, seed=123)
    # Different seed should give a different partition for a 50-animal set.
    assert s1 != s3


def test_stratified_split_rejects_bad_ratios():
    weights = {("B3", "1"): 100.0}
    with pytest.raises(ValueError, match="sum to 1"):
        stratified_animal_split(weights, ratios=(0.5, 0.3, 0.3))
    with pytest.raises(ValueError, match="Negative ratio"):
        stratified_animal_split(weights, ratios=(0.7, -0.1, 0.4))


def test_stratified_split_weight_range_covered_per_split():
    """Each split should span a wide weight range, not cluster at one end."""
    weights = {("B3", str(i)): 60.0 + 10.0 * i for i in range(1, 41)}  # 70..460 kg
    splits = stratified_animal_split(weights, ratios=(0.7, 0.15, 0.15), seed=42)
    for split_name, members in splits.items():
        if not members:
            continue
        member_weights = [weights[k] for k in members]
        spread = max(member_weights) - min(member_weights)
        # With quartile stratification, each split must span >50% of the range.
        assert spread > 200, (
            f"{split_name} weight spread too narrow: {spread} kg over "
            f"{len(members)} animals"
        )


# --------------------------------------------------------------- end-to-end


def test_build_splits_writes_csvs_and_manifest(tmp_path):
    dataset_root = _make_dataset(tmp_path, n_animals=20)
    output_dir = tmp_path / "splits"
    samples_by_split = build_splits(
        dataset_root,
        output_dir=output_dir,
        batches=("B3",),
        views=("side",),
        seed=42,
    )
    # All three CSVs exist with at least one row (small N might leave val/test empty
    # under specific seeds, but with 20 animals over 4 strata we expect non-empty).
    for split in ("train", "val", "test"):
        csv_path = output_dir / f"{split}.csv"
        assert csv_path.exists(), f"Missing {csv_path}"
        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows, f"{csv_path} has no rows"

    # Manifest sums match.
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    total_samples = sum(m["n_samples"] for m in manifest["splits"].values())
    total_animals = sum(m["n_animals"] for m in manifest["splits"].values())
    assert total_samples == sum(len(v) for v in samples_by_split.values())
    assert total_animals == 20

    # No animal_id appears in more than one split CSV.
    seen: dict[str, str] = {}
    for split in ("train", "val", "test"):
        with (output_dir / f"{split}.csv").open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = f"{row['batch']}:{row['animal_id']}"
                if key in seen:
                    assert seen[key] == split, (
                        f"Animal {key} appears in both {seen[key]} and {split}"
                    )
                seen[key] = split


def test_build_splits_ratios_approximately_held(tmp_path):
    dataset_root = _make_dataset(tmp_path, n_animals=100)
    samples_by_split = build_splits(
        dataset_root,
        output_dir=tmp_path / "splits",
        batches=("B3",),
        views=("side",),
        seed=42,
    )
    total = sum(len(v) for v in samples_by_split.values())
    train_frac = len(samples_by_split["train"]) / total
    val_frac = len(samples_by_split["val"]) / total
    test_frac = len(samples_by_split["test"]) / total
    # Allow ±5% slack from quartile stratification + integer rounding.
    assert 0.65 < train_frac < 0.75
    assert 0.10 < val_frac < 0.20
    assert 0.10 < test_frac < 0.20
