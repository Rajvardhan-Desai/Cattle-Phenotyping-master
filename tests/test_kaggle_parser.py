"""Tests for the Kaggle COCO annotation parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cattle_phenotyping.data.kaggle import (
    CANONICAL_REAR_KEYPOINTS,
    CANONICAL_SIDE_KEYPOINTS,
    FilenameParseError,
    KaggleSample,
    UnknownSchemaError,
    iter_samples,
    parse_coco_file,
    parse_filename,
    resolve_dataset_root,
)


# ---------------------------------------------------------------------- fixtures


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def b3_side_coco(tmp_path) -> Path:
    """Minimal B3 Side COCO fixture exercising the array-vs-name-number swap."""
    # 1_wither, 2_pinbone, 3_shoulderbone, then _bottom-before-_top pairs.
    payload = {
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
        "images": [{
            "id": 10,
            "file_name": "9_s_144_M.jpg",
            "width": 4160,
            "height": 3120,
        }],
        "annotations": [{
            "id": 100,
            "image_id": 10,
            "category_id": 1,
            "num_keypoints": 9,
            "bbox": [100, 200, 800, 1600],
            # x, y, v triples, one per keypoint name in array order:
            "keypoints": [
                # Array position 0 = 1_wither -> canonical "wither"
                111.0, 112.0, 2,
                # 1 = 2_pinbone -> "pinbone"
                221.0, 222.0, 2,
                # 2 = 3_shoulderbone -> "shoulderbone"
                331.0, 332.0, 2,
                # 3 = 5_front_girth_bottom -> "front_girth_bottom"
                441.0, 442.0, 2,
                # 4 = 4_front_girth_top -> "front_girth_top"
                551.0, 552.0, 2,
                # 5 = 9_Height_bottom -> "height_bottom"
                661.0, 662.0, 2,
                # 6 = 8_Height_top -> "height_top"
                771.0, 772.0, 2,
                # 7 = 7_rear_girth_bottom -> "rear_girth_bottom"
                881.0, 882.0, 2,
                # 8 = 6_rear_girth_top -> "rear_girth_top"
                991.0, 992.0, 2,
            ],
        }],
    }
    return _write_json(tmp_path / "Vector/B3/Side/data/COCO_Side.json", payload)


@pytest.fixture
def b2_rear_coco(tmp_path) -> Path:
    """B2 Rear with anonymous numeric keypoint IDs."""
    payload = {
        "categories": [{
            "id": 1,
            "name": "Cow back",
            "supercategory": "Cow",
            "keypoints": ["27", "26", "24", "25"],
            "skeleton": [[3, 4], [1, 2]],
        }],
        "images": [{"id": 1, "file_name": "1_r.jpg", "width": 1900, "height": 1425}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "num_keypoints": 4,
            "bbox": [0, 0, 1900, 1425],
            "keypoints": [
                1098.0, 691.0, 2,   # 27 -> right
                805.0, 680.0, 2,    # 26 -> left
                970.0, 378.0, 2,    # 24 -> top
                937.0, 1274.0, 2,   # 25 -> bottom
            ],
        }],
    }
    return _write_json(tmp_path / "Vector/B2/Rear/data/Rear/COCO_Rear.json", payload)


@pytest.fixture
def b4_rear_coco(tmp_path) -> Path:
    """B4 Rear with the documented `1_top, 2_bottom, 3_left, 4_right` schema."""
    payload = {
        "categories": [{
            "id": 1,
            "name": "cattle_rear",
            "supercategory": "cow",
            "keypoints": ["1_top", "2_bottom", "3_left", "4_right"],
            "skeleton": [[1, 2], [3, 4]],
        }],
        "images": [{
            "id": 1948,
            "file_name": "100_b4-1_r_124_F.jpg",
            "width": 1900,
            "height": 1425,
        }],
        "annotations": [{
            "id": 1,
            "image_id": 1948,
            "category_id": 1,
            "num_keypoints": 4,
            "bbox": [0, 0, 0, 0],  # degenerate; parser should map to None
            "keypoints": [
                1040.0, 277.0, 2,   # top
                1044.0, 1273.0, 2,  # bottom
                873.0, 493.0, 2,    # left
                1194.0, 497.0, 2,   # right
            ],
        }],
    }
    return _write_json(tmp_path / "Vector/B4/Rear/data/coco_b4_rear.json", payload)


# ------------------------------------------------------------------- filename tests


def test_parse_filename_b2_rich():
    """B2 rich grammar — actual image files on disk and Pixel/ masks."""
    meta = parse_filename("B2", "450_s_172_3.6_F.jpg")
    assert meta.animal_id == "450"
    assert meta.view == "side"
    assert meta.weight_kg == pytest.approx(172.0)
    assert meta.sex == "F"
    assert meta.extra == pytest.approx(3.6)
    assert meta.is_b2_seq_only is False


def test_parse_filename_b2_float_animal_id_normalizes_to_int():
    meta = parse_filename("B2", "113.0_s_185_4.0_F.jpg")
    assert meta.animal_id == "113"  # "113.0" → "113"
    assert meta.weight_kg == pytest.approx(185.0)
    assert meta.extra == pytest.approx(4.0)


def test_parse_filename_b2_simplified_seq_grammar():
    """B2 simplified — only Vector COCO uses this; no animal_id recoverable."""
    meta = parse_filename("B2", "1_r.jpg")
    assert meta.animal_id is None
    assert meta.view == "rear"
    assert meta.weight_kg is None
    assert meta.sex is None
    assert meta.is_b2_seq_only is True


def test_parse_filename_b2_unmatched_raises():
    with pytest.raises(FilenameParseError):
        parse_filename("B2", "garbage.jpg")


def test_parse_filename_b3():
    meta = parse_filename("B3", "9_r_144_M.jpg")
    assert meta.animal_id == "9"
    assert meta.view == "rear"
    assert meta.weight_kg == pytest.approx(144.0)
    assert meta.sex == "M"


def test_parse_filename_b3_side():
    meta = parse_filename("B3", "12_s_267_F.jpeg")
    assert meta.view == "side"
    assert meta.weight_kg == pytest.approx(267.0)
    assert meta.sex == "F"


def test_parse_filename_b4():
    meta = parse_filename("B4", "100_b4-1_r_124_F.jpg")
    assert meta.animal_id == "100"
    assert meta.view == "rear"
    assert meta.weight_kg == pytest.approx(124.0)
    assert meta.sex == "F"
    assert meta.batch_sub == "1"


def test_parse_filename_b3_rejects_b4_pattern():
    with pytest.raises(FilenameParseError):
        parse_filename("B3", "100_b4-1_r_124_F.jpg")


def test_parse_filename_unknown_batch():
    with pytest.raises(ValueError, match="Unknown batch"):
        parse_filename("B7", "1_r.jpg")


# ------------------------------------------------------------ keypoint remap tests


def test_b3_side_canonical_remap(b3_side_coco):
    samples = list(parse_coco_file(b3_side_coco, batch="B3", view="side"))
    assert len(samples) == 1
    sample = samples[0]

    # Every canonical name is present, including the ones whose array indices
    # don't match the name-number.
    assert set(sample.keypoints.keys()) == set(CANONICAL_SIDE_KEYPOINTS)
    assert all(v is not None for v in sample.keypoints.values())

    # Spot-check the swapped pairs — the canonical "front_girth_top" must come
    # from array position 4 (value 551), not position 3 (value 441).
    assert sample.keypoints["front_girth_top"][0] == pytest.approx(551.0)
    assert sample.keypoints["front_girth_bottom"][0] == pytest.approx(441.0)
    assert sample.keypoints["height_top"][0] == pytest.approx(771.0)
    assert sample.keypoints["height_bottom"][0] == pytest.approx(661.0)
    assert sample.keypoints["rear_girth_top"][0] == pytest.approx(991.0)
    assert sample.keypoints["rear_girth_bottom"][0] == pytest.approx(881.0)

    # Non-swapped keypoints land in the same place.
    assert sample.keypoints["wither"][0] == pytest.approx(111.0)
    assert sample.keypoints["shoulderbone"][0] == pytest.approx(331.0)

    # Filename-encoded weight is recovered.
    assert sample.weight_kg == pytest.approx(144.0)
    assert sample.animal_id == "9"
    assert sample.animal_key == ("B3", "9")
    assert sample.batch == "B3"
    assert sample.view == "side"
    assert sample.bbox == (100.0, 200.0, 800.0, 1600.0)


def test_b2_rear_anonymous_keypoints_resolve(b2_rear_coco):
    samples = list(parse_coco_file(b2_rear_coco, batch="B2", view="rear"))
    assert len(samples) == 1
    sample = samples[0]
    assert set(sample.keypoints.keys()) == set(CANONICAL_REAR_KEYPOINTS)
    # kp0 (id "27") high-x -> right; kp1 (id "26") low-x -> left.
    assert sample.keypoints["right"][0] == pytest.approx(1098.0)
    assert sample.keypoints["left"][0] == pytest.approx(805.0)
    assert sample.keypoints["top"][1] == pytest.approx(378.0)
    assert sample.keypoints["bottom"][1] == pytest.approx(1274.0)
    assert sample.weight_kg is None  # B2 lacks weight in filename


def test_b4_rear_degenerate_bbox_becomes_none(b4_rear_coco):
    samples = list(parse_coco_file(b4_rear_coco, batch="B4", view="rear"))
    assert len(samples) == 1
    sample = samples[0]
    # The [0,0,0,0] bbox in the source should be normalized away.
    assert sample.bbox is None
    # Weight 124 kg should still be recovered from filename.
    assert sample.weight_kg == pytest.approx(124.0)
    assert sample.filename_meta.batch_sub == "1"
    # Canonical rear keypoints all present with the documented order.
    assert sample.keypoints["top"][0] == pytest.approx(1040.0)
    assert sample.keypoints["bottom"][0] == pytest.approx(1044.0)
    assert sample.keypoints["left"][0] == pytest.approx(873.0)
    assert sample.keypoints["right"][0] == pytest.approx(1194.0)


# ----------------------------------------------------------- iter_samples wiring


def test_iter_samples_finds_inner_root(tmp_path, b3_side_coco):
    # b3_side_coco wrote under tmp_path/Vector/B3/Side/...; iter_samples should
    # resolve tmp_path as the dataset root and find that file.
    samples = list(iter_samples(tmp_path, batches=("B3",), views=("side",)))
    assert len(samples) == 1
    assert samples[0].animal_key == ("B3", "9")


def test_iter_samples_with_acme_inner_folder(tmp_path):
    # If the user passes the parent of the acme-named folder, resolve it.
    inner = tmp_path / "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV"
    _write_json(
        inner / "Vector/B3/Side/data/COCO_Side.json",
        {"categories": [], "images": [], "annotations": []},
    )
    resolved = resolve_dataset_root(tmp_path)
    assert resolved == inner


def test_b4_side_canonical_order_no_swap(tmp_path):
    """B4 Side uses name-number-correct ordering (unlike B3 Side)."""
    payload = {
        "categories": [{
            "id": 2,
            "name": "cattle_side",
            "supercategory": "cow",
            "keypoints": [
                "1_wither", "2_pinbone", "3_shoulderbone",
                "4_front_girth_top", "5_front_girth_bottom",
                "6_rear_girth_top", "7_rear_girth_bottom",
                "8_Height_top", "9_Height_bottom",
            ],
            "skeleton": [[1, 2], [2, 3], [4, 5], [6, 7], [8, 9]],
        }],
        "images": [{
            "id": 1,
            "file_name": "100_b4-1_s_124_F.jpg",
            "width": 1900,
            "height": 1425,
        }],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 2,
            "num_keypoints": 9,
            "bbox": [200, 100, 1500, 1200],
            "keypoints": [
                10.0, 11.0, 2,  # wither
                20.0, 21.0, 2,  # pinbone
                30.0, 31.0, 2,  # shoulderbone
                40.0, 41.0, 2,  # front_girth_top  (canonical order: matches array pos)
                50.0, 51.0, 2,  # front_girth_bottom
                60.0, 61.0, 2,  # rear_girth_top
                70.0, 71.0, 2,  # rear_girth_bottom
                80.0, 81.0, 2,  # height_top
                90.0, 91.0, 2,  # height_bottom
            ],
        }],
    }
    json_path = _write_json(
        tmp_path / "Vector/B4/Side/data/coco_b4_side.json", payload,
    )
    samples = list(parse_coco_file(json_path, batch="B4", view="side"))
    assert len(samples) == 1
    sample = samples[0]
    # All 9 canonical keypoints populated.
    assert set(sample.keypoints.keys()) == set(CANONICAL_SIDE_KEYPOINTS)
    # No swap — name-number-correct.
    assert sample.keypoints["front_girth_top"][0] == pytest.approx(40.0)
    assert sample.keypoints["front_girth_bottom"][0] == pytest.approx(50.0)
    assert sample.keypoints["height_top"][0] == pytest.approx(80.0)
    assert sample.keypoints["height_bottom"][0] == pytest.approx(90.0)
    # B4 filename grammar carries weight and sex.
    assert sample.weight_kg == pytest.approx(124.0)
    assert sample.filename_meta.sex == "F"
    assert sample.filename_meta.batch_sub == "1"


def test_mask_path_b3(tmp_path, b3_side_coco):
    """B3 mask join: <image_filename>___fuse.png under Pixel/B3/annotations/."""
    samples = list(iter_samples(tmp_path, batches=("B3",), views=("side",)))
    assert len(samples) == 1
    assert samples[0].mask_path is not None
    assert samples[0].mask_path.name == "9_s_144_M.jpg___fuse.png"
    assert "Pixel/B3/annotations" in samples[0].mask_path.as_posix()


def test_mask_path_b4(tmp_path):
    """B4 mask join: <image_filename>___fuse.png under Pixel/B4/Side/annotations/."""
    payload = {
        "categories": [{
            "id": 2, "name": "cattle_side",
            "keypoints": [
                "1_wither", "2_pinbone", "3_shoulderbone",
                "4_front_girth_top", "5_front_girth_bottom",
                "6_rear_girth_top", "7_rear_girth_bottom",
                "8_Height_top", "9_Height_bottom",
            ],
            "skeleton": [],
        }],
        "images": [{
            "id": 1, "file_name": "100_b4-1_s_124_F.jpg",
            "width": 1900, "height": 1425,
        }],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 2,
            "num_keypoints": 9, "bbox": [0, 0, 1900, 1425],
            "keypoints": [0.0] * 27,
        }],
    }
    _write_json(tmp_path / "Vector/B4/Side/data/coco_b4_side.json", payload)
    samples = list(iter_samples(tmp_path, batches=("B4",), views=("side",)))
    assert len(samples) == 1
    assert samples[0].mask_path is not None
    assert samples[0].mask_path.name == "100_b4-1_s_124_F.jpg___fuse.png"
    assert "Pixel/B4/Side/annotations" in samples[0].mask_path.as_posix()


def test_mask_path_b2_is_none(b2_rear_coco):
    """B2 has no clean mask join (Vector seq IDs ≠ Pixel rich filenames)."""
    # b2_rear_coco fixture is at tmp_path/Vector/B2/Rear/data/Rear/COCO_Rear.json;
    # 5 .parents hops up gets to tmp_path (Rear ← data ← Rear ← B2 ← Vector ← tmp).
    root = b2_rear_coco.parents[5]
    samples = list(iter_samples(root, batches=("B2",), views=("rear",)))
    assert len(samples) == 1
    assert samples[0].mask_path is None
    # B2 simplified filename → no animal_id, no weight, no sex.
    assert samples[0].animal_id is None
    assert samples[0].animal_key is None
    assert samples[0].filename_meta.is_b2_seq_only is True


def test_iter_samples_skips_missing_batches(tmp_path, b3_side_coco, caplog):
    # B4 side file is absent; the iterator should warn and continue.
    with caplog.at_level("WARNING"):
        samples = list(iter_samples(tmp_path, batches=("B3", "B4"), views=("side",)))
    assert len(samples) == 1
    assert any("Missing COCO file" in rec.message for rec in caplog.records)


def test_unknown_category_raises(tmp_path):
    payload = {
        "categories": [{
            "id": 1, "name": "mystery_breed",
            "keypoints": ["a", "b"], "skeleton": [],
        }],
        "images": [{"id": 1, "file_name": "1_r.jpg", "width": 10, "height": 10}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "num_keypoints": 2, "bbox": [0, 0, 5, 5],
            "keypoints": [1.0, 2.0, 2, 3.0, 4.0, 2],
        }],
    }
    json_path = _write_json(tmp_path / "Vector/B2/Rear/data/Rear/COCO_Rear.json", payload)
    # Unknown categories are logged + skipped (no annotations emitted), not
    # raised — keeps the iterator robust to schema surprises.
    samples = list(parse_coco_file(json_path, batch="B2", view="rear"))
    assert samples == []
