"""Export Kaggle B3/B4 side-view samples to Ultralytics YOLOv8-pose format.

YOLOv8-pose training expects the following on-disk layout::

    <root>/
        data.yaml
        images/train/<stem>.jpg
        images/val/<stem>.jpg
        labels/train/<stem>.txt
        labels/val/<stem>.txt

Each ``<stem>.txt`` is a single line per instance (we have one cattle per
image)::

    class_id cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ... kp9_x kp9_y kp9_v

All coordinates are normalized to ``[0, 1]`` against ``image_width`` and
``image_height``. Keypoint visibility encoding follows the Ultralytics /
COCO convention: ``0`` = not labelled, ``1`` = labelled but occluded,
``2`` = labelled and visible. Missing keypoints (no entry in
``sample.keypoints``) are written as ``0 0 0``.

The training bbox is derived from the **convex hull of visible keypoints**
expanded by a 15% margin per side (decision documented in
[[progress-2026-05-21]]). This matches standard pose-training practice and
avoids the coupling and I/O of using cow-mask bboxes during dataset export.

The :func:`export_split` function symlinks images by default so the YOLO
dataset doesn't duplicate the ~3 GB of side-view JPEGs sitting in
``/kaggle/input/``. Falls back to copy on filesystems that don't support
symlinks (Windows-non-admin / some Kaggle write paths). Set
``symlink=False`` to force copying.

CLI::

    python -m cattle_phenotyping.training.export_yolo_pose \\
        --dataset-root /kaggle/input/.../cattle-weight-detection-model-dataset-12k \\
        --train-csv data/splits/train.csv \\
        --val-csv data/splits/val.csv \\
        --suspect-csv data/calibration/suspect_samples.csv \\
        --output-dir /kaggle/working/yolo_dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

from cattle_phenotyping.data.kaggle import (
    CANONICAL_SIDE_KEYPOINTS,
    KaggleSample,
    iter_samples,
    resolve_dataset_root,
)
from cattle_phenotyping.eval.baseline_schaeffer import load_split_filenames
from cattle_phenotyping.eval.flag_suspects import load_filter_set
from cattle_phenotyping.utils.log import get_logger, setup_logging
from cattle_phenotyping.utils.seed import seed_everything

log = get_logger(__name__)


# ---------------------------------------------------------------- constants

DEFAULT_BBOX_MARGIN = 0.15           # fraction added on each side of keypoint hull
DEFAULT_CLASS_ID = 0                  # single "cattle" class
SIDE_KEYPOINT_NAMES: tuple[str, ...] = CANONICAL_SIDE_KEYPOINTS  # 9 keypoints
_VISIBLE_CODES = {1, 2}


# ----------------------------------------------------------- bbox derivation


def keypoint_hull_bbox(
    keypoints: dict,
    image_w: int,
    image_h: int,
    *,
    margin: float = DEFAULT_BBOX_MARGIN,
    keypoint_order: Sequence[str] = SIDE_KEYPOINT_NAMES,
) -> tuple[float, float, float, float] | None:
    """Compute YOLO-normalized bbox ``(cx, cy, w, h)`` from the keypoint hull.

    Considers only keypoints whose visibility code is ``1`` or ``2``. Expands
    the axis-aligned bounding box by ``margin`` on each side, then clips to
    the image bounds. Returns ``None`` when zero keypoints are visible.

    All four return values are normalized to ``[0, 1]``.
    """
    if image_w <= 0 or image_h <= 0:
        raise ValueError(
            f"image dimensions must be positive; got w={image_w}, h={image_h}"
        )

    xs: list[float] = []
    ys: list[float] = []
    for name in keypoint_order:
        kp = keypoints.get(name)
        if kp is None:
            continue
        if kp[2] not in _VISIBLE_CODES:
            continue
        xs.append(float(kp[0]))
        ys.append(float(kp[1]))
    if not xs:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w_pix = x_max - x_min
    h_pix = y_max - y_min

    # Expand by margin on each side. When the hull collapses to a single point
    # (w_pix or h_pix == 0), apply a 1px floor so YOLO doesn't choke on a
    # zero-area box. Otherwise honour the requested margin exactly — including
    # margin=0.0, which preserves the bare hull.
    pad_x = margin * w_pix if w_pix > 0 else 1.0
    pad_y = margin * h_pix if h_pix > 0 else 1.0
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    # Clip to image.
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    x_max = min(float(image_w), x_max)
    y_max = min(float(image_h), y_max)

    cx = (x_min + x_max) / 2.0 / image_w
    cy = (y_min + y_max) / 2.0 / image_h
    bw = (x_max - x_min) / image_w
    bh = (y_max - y_min) / image_h
    return (cx, cy, bw, bh)


# ----------------------------------------------------------- label encoding


def sample_to_yolo_label_string(
    sample: KaggleSample,
    *,
    class_id: int = DEFAULT_CLASS_ID,
    margin: float = DEFAULT_BBOX_MARGIN,
    keypoint_order: Sequence[str] = SIDE_KEYPOINT_NAMES,
) -> str | None:
    """Single-line YOLO-pose label for one sample, or ``None`` if no bbox.

    Returns ``None`` exactly when :func:`keypoint_hull_bbox` returns ``None``
    (no visible keypoints). Missing keypoints in the canonical order are
    encoded as ``0 0 0``.
    """
    bbox = keypoint_hull_bbox(
        sample.keypoints, sample.image_width, sample.image_height,
        margin=margin, keypoint_order=keypoint_order,
    )
    if bbox is None:
        return None
    cx, cy, bw, bh = bbox

    parts: list[str] = [str(class_id), f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
    for name in keypoint_order:
        kp = sample.keypoints.get(name)
        if kp is None or kp[2] == 0:
            parts.extend(["0.000000", "0.000000", "0"])
            continue
        nx = float(kp[0]) / sample.image_width
        ny = float(kp[1]) / sample.image_height
        parts.extend([f"{nx:.6f}", f"{ny:.6f}", str(int(kp[2]))])
    return " ".join(parts)


# ----------------------------------------------------------- file emission


def _link_or_copy_image(src: Path, dst: Path, *, symlink: bool) -> str:
    """Place ``src`` at ``dst`` via symlink (preferred) or copy fallback.

    Returns the strategy used: ``"symlink"`` or ``"copy"``.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError as exc:
            log.debug("symlink failed (%s) — falling back to copy for %s", exc, src.name)
    shutil.copy(src, dst)
    return "copy"


def export_split(
    samples: Iterable[KaggleSample],
    *,
    images_dir: Path,
    labels_dir: Path,
    drop_set: set[str] | None = None,
    class_id: int = DEFAULT_CLASS_ID,
    margin: float = DEFAULT_BBOX_MARGIN,
    keypoint_order: Sequence[str] = SIDE_KEYPOINT_NAMES,
    symlink: bool = True,
) -> dict[str, int]:
    """Materialize one split as YOLO-pose files in ``images_dir`` and ``labels_dir``.

    Returns a count dict::

        {
            "n_input":             total samples iterated,
            "n_dropped_by_filter": removed by the suspect-flag drop set,
            "n_no_bbox":           skipped because no keypoints visible,
            "n_written":           images + labels successfully emitted,
            "n_symlinked":         subset of n_written that used symlinks,
            "n_copied":            subset of n_written that used file copy,
        }
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    drop_set = drop_set or set()
    # Pre-populate the counter so callers can index any expected key without KeyError.
    counts: dict[str, int] = {
        "n_input": 0,
        "n_dropped_by_filter": 0,
        "n_no_bbox": 0,
        "n_written": 0,
        "n_symlinked": 0,
        "n_copied": 0,
    }

    for sample in samples:
        counts["n_input"] += 1
        if sample.image_path.name in drop_set:
            counts["n_dropped_by_filter"] += 1
            continue

        label_str = sample_to_yolo_label_string(
            sample, class_id=class_id, margin=margin, keypoint_order=keypoint_order,
        )
        if label_str is None:
            counts["n_no_bbox"] += 1
            continue

        label_path = labels_dir / f"{sample.image_path.stem}.txt"
        label_path.write_text(label_str + "\n", encoding="utf-8")

        image_target = images_dir / sample.image_path.name
        strategy = _link_or_copy_image(sample.image_path, image_target, symlink=symlink)
        if strategy == "symlink":
            counts["n_symlinked"] += 1
        else:
            counts["n_copied"] += 1
        counts["n_written"] += 1

    return counts


# ----------------------------------------------------------- data.yaml


def write_data_yaml(
    output_dir: Path,
    *,
    train_subdir: str = "images/train",
    val_subdir: str = "images/val",
    keypoint_names: Sequence[str] = SIDE_KEYPOINT_NAMES,
    class_names: Sequence[str] = ("cattle",),
    flip_idx: Sequence[int] | None = None,
) -> Path:
    """Emit the Ultralytics ``data.yaml`` describing this dataset.

    ``flip_idx`` defaults to the identity permutation, disabling symmetric-
    keypoint swap during any horizontal flip. Combined with ``fliplr=0.0``
    in the training args, horizontal flips are fully off — the chosen
    augmentation policy for cattle side-view photos.
    """
    n_kp = len(keypoint_names)
    if flip_idx is None:
        flip_idx = list(range(n_kp))
    if len(flip_idx) != n_kp:
        raise ValueError(
            f"flip_idx length {len(flip_idx)} != keypoint count {n_kp}"
        )

    lines: list[str] = [
        f"# Generated by cattle_phenotyping.training.export_yolo_pose",
        f"path: {Path(output_dir).resolve()}",
        f"train: {train_subdir}",
        f"val: {val_subdir}",
        "",
        f"# Keypoint metadata ({n_kp} keypoints, [x, y, visibility] per keypoint)",
        f"kpt_shape: [{n_kp}, 3]",
        f"flip_idx: {list(flip_idx)}",
        f"keypoint_names: {list(keypoint_names)}",
        "",
        "names:",
    ]
    for i, name in enumerate(class_names):
        lines.append(f"  {i}: {name}")
    lines.append("")

    out_path = output_dir / "data.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------- CLI driver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export Kaggle B3/B4 side-view samples to Ultralytics YOLOv8-pose format."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the Kaggle dataset root")
    parser.add_argument("--train-csv", required=True, help="Path to data/splits/train.csv")
    parser.add_argument("--val-csv", required=True, help="Path to data/splits/val.csv")
    parser.add_argument(
        "--suspect-csv", default="data/calibration/suspect_samples.csv",
        help="Path to suspect_samples.csv; used via load_filter_set(). Empty string disables filtering.",
    )
    parser.add_argument(
        "--output-dir", default="/kaggle/working/yolo_dataset",
        help="Where to materialize images/, labels/, data.yaml.",
    )
    parser.add_argument("--batches", nargs="+", default=["B3", "B4"])
    parser.add_argument("--views", nargs="+", default=["side"], choices=["side", "rear"])
    parser.add_argument(
        "--margin", type=float, default=DEFAULT_BBOX_MARGIN,
        help=f"Keypoint-hull bbox margin (each side); default {DEFAULT_BBOX_MARGIN}.",
    )
    parser.add_argument(
        "--no-symlink", action="store_true",
        help="Force file copy instead of symlinking images.",
    )
    parser.add_argument(
        "--summary-output", default=None,
        help="Optional JSON path to write the export summary.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)
    seed_everything()

    root = resolve_dataset_root(args.dataset_root)
    log.info("Resolved dataset root: %s", root)

    if args.suspect_csv:
        drop_set = load_filter_set(Path(args.suspect_csv))
        log.info("Loaded %d filenames to drop from %s", len(drop_set), args.suspect_csv)
    else:
        drop_set = set()
        log.info("No suspect-CSV filter applied (--suspect-csv was empty).")

    output_dir = Path(args.output_dir)

    def _samples_in(split_csv: Path) -> list[KaggleSample]:
        wanted = load_split_filenames(split_csv)
        return [
            s for s in iter_samples(root, batches=tuple(args.batches), views=tuple(args.views))  # type: ignore[arg-type]
            if s.image_path.name in wanted
        ]

    summary: dict[str, dict[str, int]] = {}
    for split_name, csv_path in (("train", args.train_csv), ("val", args.val_csv)):
        log.info("[%s] loading samples from %s", split_name, csv_path)
        samples = _samples_in(Path(csv_path))
        log.info("[%s] %d samples matched the split CSV", split_name, len(samples))
        counts = export_split(
            samples,
            images_dir=output_dir / "images" / split_name,
            labels_dir=output_dir / "labels" / split_name,
            drop_set=drop_set,
            margin=args.margin,
            symlink=not args.no_symlink,
        )
        log.info("[%s] export: %s", split_name, counts)
        summary[split_name] = counts

    yaml_path = write_data_yaml(output_dir)
    log.info("Wrote data.yaml -> %s", yaml_path)

    if args.summary_output:
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({
            "splits": summary,
            "data_yaml": str(yaml_path),
            "n_dropped_from_filter": len(drop_set),
            "config": {
                "margin": args.margin,
                "batches": list(args.batches),
                "views": list(args.views),
                "symlink": not args.no_symlink,
            },
        }, indent=2, sort_keys=True), encoding="utf-8")
        log.info("Wrote summary JSON -> %s", sp)

    return 0


if __name__ == "__main__":
    sys.exit(main())
