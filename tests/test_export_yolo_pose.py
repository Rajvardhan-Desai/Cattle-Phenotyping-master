"""Tests for the YOLOv8-pose dataset export utility."""

from __future__ import annotations

from pathlib import Path

import pytest

from cattle_phenotyping.data.kaggle import FilenameMeta, KaggleSample
from cattle_phenotyping.training.export_yolo_pose import (
    DEFAULT_BBOX_MARGIN,
    DEFAULT_CLASS_ID,
    SIDE_KEYPOINT_NAMES,
    export_split,
    keypoint_hull_bbox,
    sample_to_yolo_label_string,
    write_data_yaml,
)


# ----------------------------------------------------------- helpers


def _kp(x: float, y: float, v: int = 2) -> tuple[float, float, int]:
    return (x, y, v)


def _make_sample(
    keypoints: dict,
    *,
    image_path: Path | str = "img.jpg",
    image_width: int = 1000,
    image_height: int = 800,
    batch: str = "B3",
    weight_kg: float = 200.0,
    animal_id: str = "1",
) -> KaggleSample:
    return KaggleSample(
        image_path=Path(image_path),
        batch=batch,  # type: ignore[arg-type]
        view="side",
        coco_image_id=1, coco_category_id=1,
        keypoints=keypoints, bbox=None,
        image_width=image_width, image_height=image_height,
        filename_meta=FilenameMeta(
            animal_id=animal_id, view="side",
            weight_kg=weight_kg, sex="F",  # type: ignore[arg-type]
        ),
    )


# -------------------------------------------------- keypoint_hull_bbox


def test_keypoint_hull_bbox_returns_none_when_no_visible_keypoints():
    kps = {
        "wither": _kp(100, 100, v=0),
        "pinbone": _kp(200, 100, v=0),
    }
    assert keypoint_hull_bbox(kps, image_w=1000, image_h=800) is None


def test_keypoint_hull_bbox_normalizes_against_image():
    kps = {
        "wither": _kp(200, 200),
        "pinbone": _kp(800, 600),
    }
    bbox = keypoint_hull_bbox(kps, image_w=1000, image_h=800, margin=0.0)
    assert bbox is not None
    cx, cy, w, h = bbox
    # Hull spans x=[200,800], y=[200,600]. With margin=0:
    # cx = 500/1000 = 0.5, cy = 400/800 = 0.5, w = 600/1000 = 0.6, h = 400/800 = 0.5
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(0.6)
    assert h == pytest.approx(0.5)


def test_keypoint_hull_bbox_applies_margin_symmetrically():
    kps = {"wither": _kp(400, 300), "pinbone": _kp(600, 500)}
    # Hull 200x200. With margin=0.5, each side gains 100px → bbox 400x400 from (300,200) to (700,600).
    bbox = keypoint_hull_bbox(kps, image_w=1000, image_h=800, margin=0.5)
    assert bbox is not None
    cx, cy, w, h = bbox
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(0.4)
    assert h == pytest.approx(0.5)


def test_keypoint_hull_bbox_clips_to_image_boundaries():
    """A keypoint near the corner with large margin must not produce >1.0 coords."""
    kps = {"wither": _kp(50, 50), "pinbone": _kp(100, 100)}
    bbox = keypoint_hull_bbox(kps, image_w=1000, image_h=800, margin=5.0)
    assert bbox is not None
    cx, cy, w, h = bbox
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    # All margin would push x_max past the image; should clip at the image edge.
    assert w <= 1.0
    assert h <= 1.0


def test_keypoint_hull_bbox_handles_single_visible_keypoint():
    """Collapsed hull (one keypoint) should still produce a non-zero bbox."""
    kps = {"wither": _kp(500, 400)}
    bbox = keypoint_hull_bbox(kps, image_w=1000, image_h=800, margin=0.15)
    assert bbox is not None
    cx, cy, w, h = bbox
    # Center at the keypoint, w/h > 0 thanks to the min-1px floor.
    assert cx == pytest.approx(0.5, abs=0.001)
    assert cy == pytest.approx(0.5, abs=0.001)
    assert w > 0
    assert h > 0


def test_keypoint_hull_bbox_skips_invisible_keypoints():
    kps = {
        "wither": _kp(100, 100, v=0),      # invisible — excluded
        "pinbone": _kp(900, 700, v=0),     # invisible — excluded
        "shoulderbone": _kp(400, 300),     # visible
        "front_girth_top": _kp(600, 500),  # visible
    }
    bbox = keypoint_hull_bbox(kps, image_w=1000, image_h=800, margin=0.0)
    assert bbox is not None
    cx, cy, w, h = bbox
    # Hull is only over the visible pair (400,300)-(600,500).
    assert cx == pytest.approx(0.5)
    assert cy == pytest.approx(0.5)
    assert w == pytest.approx(0.2)
    assert h == pytest.approx(0.25)


def test_keypoint_hull_bbox_rejects_bad_image_dims():
    with pytest.raises(ValueError, match="positive"):
        keypoint_hull_bbox({"wither": _kp(0, 0)}, image_w=0, image_h=100)


# ---------------------------------------------- sample_to_yolo_label_string


def test_label_string_full_format():
    """Verify the entire label-line shape end-to-end."""
    kps = {name: _kp(100.0 * (i + 1), 50.0 * (i + 1)) for i, name in enumerate(SIDE_KEYPOINT_NAMES)}
    sample = _make_sample(kps, image_width=1000, image_height=500)
    line = sample_to_yolo_label_string(sample, margin=0.0)
    assert line is not None

    parts = line.split()
    # class_id + 4 bbox + 9 × 3 = 5 + 27 = 32 tokens
    assert len(parts) == 1 + 4 + 9 * 3
    assert parts[0] == str(DEFAULT_CLASS_ID)
    # Each of the 9 visibility codes should be "2" (all _kp default v=2).
    visibility_tokens = parts[5 + 2::3]  # offsets 7, 10, 13, ...
    assert all(v == "2" for v in visibility_tokens)
    assert len(visibility_tokens) == 9


def test_label_string_encodes_missing_keypoints_as_000():
    kps = {
        "wither": _kp(500, 250),  # only one keypoint provided
    }
    sample = _make_sample(kps, image_width=1000, image_height=500)
    line = sample_to_yolo_label_string(sample, margin=0.0)
    assert line is not None
    parts = line.split()
    # First keypoint = wither, visibility=2; remaining 8 keypoints = "0 0 0".
    kp_tokens = parts[5:]
    assert kp_tokens[0:3] == ["0.500000", "0.500000", "2"]
    for i in range(1, 9):
        assert kp_tokens[i * 3 : i * 3 + 3] == ["0.000000", "0.000000", "0"]


def test_label_string_returns_none_for_zero_visible_keypoints():
    kps = {name: _kp(100, 100, v=0) for name in SIDE_KEYPOINT_NAMES}
    sample = _make_sample(kps)
    assert sample_to_yolo_label_string(sample) is None


def test_label_string_preserves_visibility_code_one():
    """v=1 (labelled but occluded) must round-trip as int '1', not be silently dropped."""
    kps = {
        "wither": _kp(200, 100, v=2),
        "pinbone": _kp(600, 300, v=1),
    }
    sample = _make_sample(kps, image_width=1000, image_height=400)
    line = sample_to_yolo_label_string(sample, margin=0.0)
    assert line is not None
    parts = line.split()
    # Order of canonical keypoints: wither (0), pinbone (1)
    assert parts[5 + 2] == "2"   # wither
    assert parts[8 + 2] == "1"   # pinbone


def test_label_string_uses_custom_class_id():
    kps = {"wither": _kp(500, 250)}
    sample = _make_sample(kps, image_width=1000, image_height=500)
    line = sample_to_yolo_label_string(sample, class_id=7)
    assert line is not None
    assert line.split()[0] == "7"


# ------------------------------------------------------------- export_split


def _seed_image(tmp_path: Path, name: str) -> Path:
    """Create a tiny placeholder image file so symlink/copy has something to point at."""
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    return p


def test_export_split_writes_label_and_image(tmp_path):
    img_src = _seed_image(tmp_path, "src.jpg")
    kps = {name: _kp(100, 100) for name in SIDE_KEYPOINT_NAMES}
    sample = _make_sample(kps, image_path=img_src, image_width=1000, image_height=800)

    images_dir = tmp_path / "yolo" / "images" / "train"
    labels_dir = tmp_path / "yolo" / "labels" / "train"
    counts = export_split([sample], images_dir=images_dir, labels_dir=labels_dir, symlink=False)

    assert counts["n_input"] == 1
    assert counts["n_written"] == 1
    assert counts["n_no_bbox"] == 0
    assert counts.get("n_dropped_by_filter", 0) == 0
    # Image was copied (symlink=False forces copy on every platform).
    assert (images_dir / "src.jpg").exists()
    # Label file written with the .txt extension at the same stem.
    label_path = labels_dir / "src.txt"
    assert label_path.exists()
    assert label_path.read_text(encoding="utf-8").startswith("0 ")


def test_export_split_respects_drop_set(tmp_path):
    img_keep = _seed_image(tmp_path, "keep.jpg")
    img_drop = _seed_image(tmp_path, "drop.jpg")
    kps = {name: _kp(100, 100) for name in SIDE_KEYPOINT_NAMES}
    samples = [
        _make_sample(kps, image_path=img_keep),
        _make_sample(kps, image_path=img_drop),
    ]
    counts = export_split(
        samples,
        images_dir=tmp_path / "images",
        labels_dir=tmp_path / "labels",
        drop_set={"drop.jpg"},
        symlink=False,
    )
    assert counts["n_input"] == 2
    assert counts["n_dropped_by_filter"] == 1
    assert counts["n_written"] == 1
    assert (tmp_path / "images" / "keep.jpg").exists()
    assert not (tmp_path / "images" / "drop.jpg").exists()


def test_export_split_counts_no_bbox_when_keypoints_invisible(tmp_path):
    img_src = _seed_image(tmp_path, "ghost.jpg")
    kps = {name: _kp(0, 0, v=0) for name in SIDE_KEYPOINT_NAMES}
    sample = _make_sample(kps, image_path=img_src)
    counts = export_split(
        [sample],
        images_dir=tmp_path / "images",
        labels_dir=tmp_path / "labels",
        symlink=False,
    )
    assert counts["n_no_bbox"] == 1
    assert counts["n_written"] == 0
    # Neither image nor label written.
    assert not (tmp_path / "images" / "ghost.jpg").exists()
    assert not (tmp_path / "labels" / "ghost.txt").exists()


def test_export_split_replaces_existing_image(tmp_path):
    """Re-running export on the same target should overwrite cleanly."""
    img_src = _seed_image(tmp_path, "x.jpg")
    kps = {name: _kp(100, 100) for name in SIDE_KEYPOINT_NAMES}
    sample = _make_sample(kps, image_path=img_src)
    images_dir = tmp_path / "out" / "images"
    labels_dir = tmp_path / "out" / "labels"

    counts1 = export_split([sample], images_dir=images_dir, labels_dir=labels_dir, symlink=False)
    counts2 = export_split([sample], images_dir=images_dir, labels_dir=labels_dir, symlink=False)
    assert counts1["n_written"] == 1
    assert counts2["n_written"] == 1
    assert (images_dir / "x.jpg").exists()


# ---------------------------------------------------------- write_data_yaml


def test_write_data_yaml_emits_expected_keys(tmp_path):
    out_dir = tmp_path / "ds"
    path = write_data_yaml(out_dir)
    content = path.read_text(encoding="utf-8")
    assert "path:" in content
    assert "train: images/train" in content
    assert "val: images/val" in content
    assert "kpt_shape: [9, 3]" in content
    assert "flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8]" in content
    assert "0: cattle" in content


def test_write_data_yaml_rejects_mismatched_flip_idx(tmp_path):
    with pytest.raises(ValueError, match="flip_idx length"):
        write_data_yaml(tmp_path, flip_idx=[0, 1])  # only 2 indices for 9 keypoints


def test_write_data_yaml_preserves_keypoint_names_order(tmp_path):
    path = write_data_yaml(tmp_path)
    content = path.read_text(encoding="utf-8")
    # All 9 canonical names should appear in order in the keypoint_names line.
    kp_line = next(line for line in content.splitlines() if line.startswith("keypoint_names"))
    for name in SIDE_KEYPOINT_NAMES:
        assert name in kp_line


def test_write_data_yaml_creates_output_dir(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    path = write_data_yaml(deep)
    assert path.exists()
    assert path.parent == deep
