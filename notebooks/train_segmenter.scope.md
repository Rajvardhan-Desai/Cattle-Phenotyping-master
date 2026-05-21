# `train_segmenter.ipynb` — scope

Train YOLOv8s-seg on **cow + sticker** masks from B3/B4 side-view. Replaces SAM at inference and recovers per-image sticker pixel area without depending on the dataset's hand-labelled `Pixel/.../annotations/*___fuse.png` files.

Bar to beat:
- **Per-class IoU** ≥ 0.85 (cattle), ≥ 0.60 (sticker) on val.
- **Forward-Schaeffer MAE** on val using **predicted sticker area + GT keypoints** ≤ 35 kg. Compare to:
  - 30.25 kg (GT keypoints + GT sticker mask, test) — pure baseline ceiling.
  - The MAE from `notebooks/train_keypoints.scope.md` (predicted keypoints + GT sticker mask) — keypoint-only contribution.
  - The MAE from this notebook (GT keypoints + **predicted** sticker mask) — segmenter-only contribution.

Run on Kaggle (T4×2 GPU). Same harness as the keypoint scope.

## Decisions locked in

| Choice | Value | Why |
|---|---|---|
| Model | `yolov8s-seg` | 11.8M params, parallels keypoint scope's compute footprint |
| Image size | 640 px | Same as keypoint scope; sticker still readable |
| Classes | `0=cattle, 1=sticker` | Two-class seg; ground/background folded into bg |
| Horizontal flip | **Enabled** (`fliplr=0.5`, default) | Unlike pose, masks don't care about L/R anatomy |
| Mask source | `Pixel/<batch>/[<Side>/]annotations/<file>___fuse.png` | Documented in `docs/kaggle_dataset_notes.md` |
| Sticker colors | B3/B4 = `(0,117,255)` (blue), tolerance ±15 | From `pipeline/scale_calibration.STICKER_RGB_BY_BATCH` |
| Cattle color | **TODO before run — verify on Kaggle, see Cell 0 below** | Not documented in repo yet |
| Filter | `load_filter_set("data/calibration/suspect_samples.csv")` (~110 drops) | Same as keypoint scope |
| Eval | Per-class IoU + per-image sticker-area MAE (px²) + forward Schaeffer with predicted sticker | Three layers; the third is the deliverable metric |

## Pre-work required before this notebook runs

Two new modules need to land first. Both parallel the keypoint workstream:

1. **`cattle_phenotyping/training/export_yolo_seg.py`** — analog of `export_yolo_pose.py`. For each sample, read its `mask_path` PNG, threshold for each class color (cow, sticker), extract polygon contours via OpenCV `findContours`, and emit YOLO-seg labels:
   ```
   class_id x1 y1 x2 y2 ... xN yN   # normalized to [0,1]; one line per instance
   ```
   Same symlink-or-copy image strategy, same suspect-CSV filter, same train/val split CSVs, same `data.yaml` writer (just with `names: {0: cattle, 1: sticker}` and no `kpt_shape`/`flip_idx`). Tests mirror `tests/test_export_yolo_pose.py` against a tiny synthetic mask fixture.
2. **`cattle_phenotyping/eval/segmenter_eval.py`** — analog of `keypoint_eval.py`. Given predicted masks per filename (Ultralytics output) + `KaggleSample` list + per-batch sticker cm², compute:
   - per-class IoU (against GT masks loaded the same way as the export step),
   - per-image predicted sticker pixel area,
   - forward Schaeffer with `(GT keypoints, predicted sticker area, per-batch cm²)`,
   - skip reasons (no cow detected, no sticker detected, both, etc.).
   Single `build_eval_report(...)` entry point so the notebook is one call.

**Open question both modules need answered first:** what RGB color encodes the cow class in `Pixel/.../annotations/*___fuse.png`? Only the sticker color is documented. Add Cell 0 (below) to determine it from a sample mask before exporting.

These are not gated on the keypoint training run — they can be written in parallel.

## Cell-by-cell

### 0. Identify the cattle mask color (one-time, before export)

```python
# Cell 0a: pick any B3 mask file and inspect its unique RGB values
from PIL import Image
import numpy as np
from pathlib import Path

DATASET_ROOT = Path("/kaggle/input/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
mask_dir = DATASET_ROOT / "www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/Pixel/B3/annotations"
sample_mask = next(mask_dir.glob("*___fuse.png"))
arr = np.asarray(Image.open(sample_mask).convert("RGB"), dtype=np.uint8)
unique_colors, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
order = np.argsort(-counts)
for c, n in zip(unique_colors[order][:8], counts[order][:8]):
    print(f"  RGB={tuple(int(x) for x in c)}  count={n}  ({100*n/arr.size*3:.2f}%)")
# Expected output:
#   - Background (largest count) — black or white?
#   - Sticker (0, 117, 255) — already known
#   - Cattle (TBD) — the third dominant cluster
# Record the cattle RGB and add it to STICKER_RGB_BY_BATCH-style constants
# in cattle_phenotyping/training/export_yolo_seg.py before training.
```

### 1. Install + clone + sanity (cells 1-5) — same as keypoint scope

```python
# Cell 1: install Ultralytics
!pip -q install "ultralytics>=8.1.0"

# Cell 2: clone the repo
!cd /kaggle/working && [ -d Cattle-Phenotyping-master ] || git clone https://github.com/Rajvardhan-Desai/Cattle-Phenotyping-master.git

# Cell 3: install the package editable
%cd /kaggle/working/Cattle-Phenotyping-master
!pip -q install -e .

# Cell 4: import + setup
import json
from pathlib import Path
from cattle_phenotyping.utils.log import setup_logging
from cattle_phenotyping.utils.seed import seed_everything
setup_logging(level="INFO")
seed_everything(seed=42)

DATASET_ROOT = Path("/kaggle/input/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
SPLITS_DIR   = Path("data/splits")
CALIB_DIR    = Path("data/calibration")
RESULTS_DIR  = Path("data/results")
YOLO_DS_DIR  = Path("/kaggle/working/yolo_seg_dataset")
RUNS_DIR     = Path("/kaggle/working/runs/seg")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Cell 5: verify required inputs exist
required = [
    SPLITS_DIR / "train.csv",
    SPLITS_DIR / "val.csv",
    SPLITS_DIR / "test.csv",
    CALIB_DIR / "sticker_area_cm2_by_batch.json",
    CALIB_DIR / "suspect_samples.csv",
]
missing = [p for p in required if not p.exists()]
assert not missing, f"Missing required input files: {missing}"
sticker_by_batch = json.loads((CALIB_DIR / "sticker_area_cm2_by_batch.json").read_text())
print(f"Per-batch sticker cm²: {sticker_by_batch}")
```

### 2. Dataset export (cells 6-9)

```python
# Cell 6: load the filter set
from cattle_phenotyping.eval.flag_suspects import load_filter_set
drop_set = load_filter_set(CALIB_DIR / "suspect_samples.csv")
print(f"Filter drops: {len(drop_set)} samples")

# Cell 7: export train + val to YOLO-seg format
# (PRE-WORK: cattle_phenotyping.training.export_yolo_seg must exist by this point.)
!python -m cattle_phenotyping.training.export_yolo_seg \
    --dataset-root "{DATASET_ROOT}" \
    --train-csv data/splits/train.csv \
    --val-csv data/splits/val.csv \
    --suspect-csv data/calibration/suspect_samples.csv \
    --output-dir /kaggle/working/yolo_seg_dataset \
    --summary-output /kaggle/working/yolo_seg_dataset/export_summary.json

# Cell 8: inspect the exported counts
print(json.loads((YOLO_DS_DIR / "export_summary.json").read_text()))
# Expected (approx): train ~3080 written, val ~660 written, ~110-115 dropped.
# Each label file should contain 2 lines (cow polygon + sticker polygon).

# Cell 9: spot-check a label file
sample_label = next((YOLO_DS_DIR / "labels" / "train").glob("*.txt"))
print(sample_label.name)
for line in sample_label.read_text().splitlines():
    head = line.split(maxsplit=1)
    print(f"  class={head[0]}  n_points={(len(line.split()) - 1) // 2}")
# Expect: "class=0 n_points=NN" (cattle) and "class=1 n_points=NN" (sticker).
```

### 3. Train (cells 10-12)

```python
# Cell 10: launch training
from ultralytics import YOLO
model = YOLO("yolov8s-seg.pt")
results = model.train(
    data=str(YOLO_DS_DIR / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=20,           # seg is slightly heavier than pose; smaller batch for safety
    device="0,1",
    workers=4,
    project=str(RUNS_DIR),
    name="seg_001",
    exist_ok=True,
    # NOTE: fliplr=0.5 (default) is fine — segmentation masks are L/R symmetric.
    mosaic=1.0,
    mixup=0.0,
    cos_lr=True,
    patience=15,
    seed=42,
)
print(f"Best weights at: {results.save_dir}/weights/best.pt")

# Cell 11: Ultralytics native eval
metrics = model.val()
print(f"Box mAP50: {metrics.box.map50:.3f}   Mask mAP50: {metrics.seg.map50:.3f}")
print(f"Per-class mask IoU (TP-only): {metrics.seg.maps}")

# Cell 12: persist weights to /kaggle/working for download
weights_dst = RUNS_DIR / "seg_001" / "weights" / "best.pt"
print(f"Weights size: {weights_dst.stat().st_size / 1e6:.1f} MB")
```

### 4. Eval setup — predict val masks + run scoring (cells 13-16)

```python
# Cell 13: re-load val samples (need GT keypoints + GT masks for scoring)
from cattle_phenotyping.data.kaggle import iter_samples, resolve_dataset_root
from cattle_phenotyping.eval.baseline_schaeffer import load_split_filenames

root = resolve_dataset_root(DATASET_ROOT)
val_names = load_split_filenames(SPLITS_DIR / "val.csv")
val_samples = [
    s for s in iter_samples(root, batches=("B3", "B4"), views=("side",))
    if s.image_path.name in val_names
]
print(f"Val samples loaded: {len(val_samples)}")

# Cell 14: predict on every val image
predictor = YOLO(str(RUNS_DIR / "seg_001" / "weights" / "best.pt"))
preds = predictor.predict(
    source=str(YOLO_DS_DIR / "images" / "val"),
    imgsz=640, conf=0.25, device="0",
    save=False, save_txt=False, verbose=False, stream=True,
    retina_masks=True,   # full-resolution masks instead of mask-proto-sized
)

# Build per-filename mapping: {filename: {"cattle": mask_HxW_bool, "sticker": mask_HxW_bool}}
import numpy as np
predicted = {}
for r in preds:
    name = Path(r.path).name
    if r.masks is None or r.boxes is None or len(r.boxes) == 0:
        continue
    classes = r.boxes.cls.cpu().numpy().astype(int)
    masks = r.masks.data.cpu().numpy().astype(bool)  # (N, H, W)
    confs = r.boxes.conf.cpu().numpy()
    # For each class, pick the highest-confidence instance.
    per_class = {}
    for cls_id, cls_name in [(0, "cattle"), (1, "sticker")]:
        cls_mask = classes == cls_id
        if not cls_mask.any():
            continue
        idxs = np.where(cls_mask)[0]
        best = idxs[int(confs[cls_mask].argmax())]
        per_class[cls_name] = masks[best]
    predicted[name] = per_class
print(f"Got predictions for {len(predicted)}/{len(val_samples)} val images")

# Cell 15: pre-compute predicted sticker pixel areas (one int per filename)
predicted_sticker_area_px = {
    name: int(per_class["sticker"].sum())
    for name, per_class in predicted.items()
    if "sticker" in per_class
}
n_no_sticker = len(predicted) - len(predicted_sticker_area_px)
print(f"Predicted sticker present: {len(predicted_sticker_area_px)} / {len(predicted)}")
print(f"No-sticker-detected count: {n_no_sticker}")

# Cell 16: also load GT sticker areas for direct comparison (sanity)
from cattle_phenotyping.data.mask_io import load_sticker_areas
mask_jobs = [(i, s.mask_path, s.batch) for i, s in enumerate(val_samples) if s.mask_path is not None]
gt_sticker_areas_by_idx = load_sticker_areas(mask_jobs, workers=8)
gt_sticker_area_px = {
    val_samples[i].image_path.name: a
    for i, a in gt_sticker_areas_by_idx.items() if a is not None
}
```

### 5. Eval — three metric layers (cells 17-19)

```python
# Cell 17: per-class IoU (predicted vs GT mask)
# (PRE-WORK: cattle_phenotyping.eval.segmenter_eval.build_eval_report must exist.)
from cattle_phenotyping.eval.segmenter_eval import build_eval_report
report = build_eval_report(
    predictions=predicted,
    samples=val_samples,
    gt_sticker_area_px_by_filename=gt_sticker_area_px,
    sticker_area_cm2_by_batch=sticker_by_batch,
    # GT keypoints are pulled from `samples`; predicted sticker area is pulled
    # from `predictions["<file>"]["sticker"].sum()` internally.
)
print(json.dumps({
    "n_samples": report["n_samples"],
    "n_with_cow_pred": report["n_with_cow_pred"],
    "n_with_sticker_pred": report["n_with_sticker_pred"],
    "skip_reasons": report["skip_reasons"],
}, indent=2))

# Cell 18: render the three metric layers
print("Per-class mask IoU (val):")
for cls_name, m in report["iou_by_class"].items():
    print(f"  {cls_name:<10} n={m['n']}  median_iou={m['median']:.3f}  mean_iou={m['mean']:.3f}  p10={m['p10']:.3f}")

print()
print("Sticker area error (predicted vs GT mask, px²):")
sa = report["sticker_area_error"]
print(f"  n={sa['n']}  median_abs_pct_err={sa['median_abs_pct_err']:.2f}%  bias_pct={sa['bias_pct']:+.2f}%")

print()
print("Forward-Schaeffer with GT keypoints + PREDICTED sticker (val):")
overall = report["weight_overall"]
print(f"  OVERALL  n={overall['n']}  MAE={overall['mae_kg']:.2f}  RMSE={overall['rmse_kg']:.2f}  MAPE={overall['mape_pct']:.2f}%  bias={overall['bias_kg']:+.2f}  R²={overall['r2']:.4f}")
for batch_name, m in report["weight_by_batch"].items():
    print(f"  [{batch_name}]   n={m['n']}  MAE={m['mae_kg']:.2f}  RMSE={m['rmse_kg']:.2f}  MAPE={m['mape_pct']:.2f}%  bias={m['bias_kg']:+.2f}")

print()
print("Baseline (GT kpts + GT sticker, TEST):  MAE=30.25 kg  MAPE=19.69%")
print(f"This model (GT kpts + PRED sticker, VAL):  MAE={overall['mae_kg']:.2f} kg  MAPE={overall['mape_pct']:.2f}%")

# Cell 19: persist the full report
report_path = RESULTS_DIR / "yolov8s_seg_val_eval.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
print(f"Wrote {report_path}")
```

### 6. Persist artifacts to commit back (cells 20-21)

```python
# Cell 20: copy best.pt to a downloadable location
import shutil
weights_out = Path("/kaggle/working/yolov8s_seg_seg_001_best.pt")
shutil.copy(RUNS_DIR / "seg_001" / "weights" / "best.pt", weights_out)
print(f"Download from: {weights_out}  ({weights_out.stat().st_size/1e6:.1f} MB)")
# Workflow: download this file and upload to GitHub Releases as
# `yolov8s_seg_seg_001.pt`. Register hash in configs/default.yaml.

# Cell 21: what to commit back to repo
artifacts_to_commit = [
    "data/results/yolov8s_seg_val_eval.json",
    f"{RUNS_DIR}/seg_001/results.csv",
    f"{RUNS_DIR}/seg_001/args.yaml",
]
for p in artifacts_to_commit:
    if Path(p).exists():
        print(f"  ✓ {p}")
    else:
        print(f"  ✗ MISSING: {p}")
```

## Success criteria

- **Pipeline correctness:** export → train → eval runs end-to-end without exceptions.
- **Mask mAP50 ≥ 0.85** on val (Ultralytics' native metric — sanity that the seg head trained).
- **Per-class IoU:** cattle ≥ 0.85, sticker ≥ 0.60 (sticker is small; lower bar is realistic).
- **Predicted sticker pixel area** within ±15% of GT mask area, median.
- **Forward-Schaeffer MAE on val (GT kpts + PRED sticker) ≤ 35 kg.** If the gap to the 30.25 baseline (GT kpts + GT sticker, test) is <5 kg, segmentation is not the bottleneck. If the gap is >15 kg, debug sticker recall before moving on.

## Open questions to revisit after the first run

- Does the sticker recall drop on B4 (larger sticker, easier) or B3 (smaller sticker, harder)? Per-batch IoU break-down should answer this.
- Are mis-detections clustered on certain backgrounds (mud, hay, dark coat)? Spot-check failure cases.
- Should we add `retina_masks=True` to training (not just inference) for sharper sticker boundaries at low resolution?
- Is `mosaic=1.0` actually helping for a 2-class dataset with one cattle per image? Consider a no-mosaic ablation.

## Follow-up plan after a successful train

1. Wire the segmenter into the inference pipeline alongside the trained keypoint model — replace SAM in `pipeline/phenotyping_pipeline.py`.
2. Run combined-prediction eval: forward Schaeffer with **predicted keypoints + predicted sticker**. This is the true end-to-end ceiling for Schaeffer; the gap to the all-GT 30.25 kg baseline is the total noise budget for the learned weight head to beat.
3. Build the learned weight head on top of `(predicted_keypoints + predicted_sticker_area + per-batch cm²)`.
