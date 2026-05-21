# `train_keypoints.ipynb` — scope

Train YOLOv8s-pose on the 9 canonical side-view keypoints. Bar to beat:
**30.25 kg MAE / 19.69% MAPE** on test (forward Schaeffer with per-batch
sticker constants; see `data/results/baseline_schaeffer_test_perbatch.json`).

Run on Kaggle (T4×2 GPU). Cells in execution order. Every snippet below is
copy-paste-ready into a Kaggle code cell.

## Decisions locked in

| Choice | Value | Why |
|---|---|---|
| Model | `yolov8s-pose` | 11.6M params, T4×2 fits ~50 epochs in 2-3 hr |
| Image size | 640 px | Default; sticker ≈ 50-100 px at 640, readable |
| Horizontal flip | **Disabled** | Side-view photos break L/R if flipped |
| Bbox source | Keypoint hull + 15% margin | No mask I/O, deterministic, standard |
| OKS sigmas | `apply_cattle_pose_sigmas()` (0.025-0.05 per kpt) | **Critical** — default σ=1/N=0.111 lets the metric report 0.95 while the model places `_top` keypoints at the bbox edge. See [[progress-2026-05-21]] post-mortem. |
| Eval | Forward-Schaeffer MAE/MAPE + per-keypoint pixel error | Direct comparison to baseline |
| Filter | `load_filter_set("data/calibration/suspect_samples.csv")` (~110-115 drops) | `large_residual` + `implausible_low_weight` only |

## Cell-by-cell

### 1. Install + clone repo + sanity (cells 1-5)

```python
# Cell 1: install Ultralytics (Kaggle has older default)
!pip -q install "ultralytics>=8.1.0"

# Cell 2: clone the repo into /kaggle/working
!cd /kaggle/working && [ -d Cattle-Phenotyping-master ] || git clone https://github.com/Rajvardhan-Desai/Cattle-Phenotyping-master.git

# Cell 3: install the package (editable, so notebook can call into it)
%cd /kaggle/working/Cattle-Phenotyping-master
!pip -q install -e .

# Cell 4: import + setup
import json, math
from pathlib import Path
from cattle_phenotyping.utils.log import setup_logging
from cattle_phenotyping.utils.seed import seed_everything
setup_logging(level="INFO")
seed_everything(seed=42)

DATASET_ROOT = Path("/kaggle/input/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k")
SPLITS_DIR   = Path("data/splits")
CALIB_DIR    = Path("data/calibration")
RESULTS_DIR  = Path("data/results")
YOLO_DS_DIR  = Path("/kaggle/working/yolo_dataset")
RUNS_DIR     = Path("/kaggle/working/runs/pose")
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Cell 5: verify required inputs exist (fail fast if a previous step wasn't committed)
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

# Cell 7: export train + val to YOLO format
# (Symlink fallback to copy on Kaggle since /kaggle/working supports symlinks.)
!python -m cattle_phenotyping.training.export_yolo_pose \
    --dataset-root "{DATASET_ROOT}" \
    --train-csv data/splits/train.csv \
    --val-csv data/splits/val.csv \
    --suspect-csv data/calibration/suspect_samples.csv \
    --output-dir /kaggle/working/yolo_dataset \
    --summary-output /kaggle/working/yolo_dataset/export_summary.json

# Cell 8: inspect the exported counts
print(json.loads((YOLO_DS_DIR / "export_summary.json").read_text()))
# Expected (approx): train ~3080 written, val ~660 written, ~110-115 dropped.

# Cell 9: spot-check a label file
sample_label = next((YOLO_DS_DIR / "labels" / "train").glob("*.txt"))
print(sample_label.name, "->", sample_label.read_text().strip())
# Should be: "0 0.xxx 0.xxx 0.xxx 0.xxx" + 9 × "kpx kpy v"  (32 tokens)
```

### 3. Train (cells 10-12)

```python
# Cell 10: launch training
from ultralytics import YOLO
from cattle_phenotyping.training.kpt_sigmas import apply_cattle_pose_sigmas

# CRITICAL: install cattle-specific OKS sigmas BEFORE model.train() runs.
# Without this, Ultralytics 8.4.x falls back to sigma = 1/N = 0.111 for every
# keypoint, which is so loose that the pose loss gradient can't push the head
# off the "predict every _top keypoint at the bbox top edge" local minimum.
# The 2026-05-21 run hit mAP50-95(P) = 0.953 with that bug and produced
# forward-Schaeffer val MAE = 166 kg. Patch first, train after.
apply_cattle_pose_sigmas()

model = YOLO("yolov8s-pose.pt")
results = model.train(
    data=str(YOLO_DS_DIR / "data.yaml"),
    epochs=50,
    imgsz=640,
    batch=24,           # T4 has 16GB; 24 is safe for yolov8s at 640.
    device="0,1",       # both T4s
    workers=4,
    project=str(RUNS_DIR),
    name="kpt_001",
    exist_ok=True,
    fliplr=0.0,         # CRITICAL: disable horizontal flip
    mosaic=1.0,
    mixup=0.0,
    cos_lr=True,
    patience=15,        # early-stop if val pose loss plateaus 15 epochs
    pose=12.0,          # pose-loss weight; default is fine for canonical anatomy
    seed=42,
)
print(f"Best weights at: {results.save_dir}/weights/best.pt")

# Cell 11: quick sanity — Ultralytics' own metrics
metrics = model.val()  # uses the data.yaml's val: split
print(f"OKS-50: {metrics.pose.map50:.3f}   OKS-95: {metrics.pose.map:.3f}")

# Cell 12: persist weights to /kaggle/working for download
import shutil
weights_dst = RUNS_DIR / "kpt_001" / "weights" / "best.pt"
print(f"Weights size: {weights_dst.stat().st_size / 1e6:.1f} MB")
```

### 4. Eval setup — load val + run predictor (cells 13-15)

```python
# Cell 13: re-load samples for val, aligned to image filenames
from cattle_phenotyping.data.kaggle import iter_samples, resolve_dataset_root, CANONICAL_SIDE_KEYPOINTS
from cattle_phenotyping.eval.baseline_schaeffer import load_split_filenames

root = resolve_dataset_root(DATASET_ROOT)
val_names = load_split_filenames(SPLITS_DIR / "val.csv")
val_samples = [s for s in iter_samples(root, batches=("B3", "B4"), views=("side",)) if s.image_path.name in val_names]
print(f"Val samples loaded: {len(val_samples)}")

# Cell 14: predict on every val image — Ultralytics output adapter
# (this is the only YOLO-API-coupled cell; everything downstream is plain dicts)
predictor = YOLO(str(RUNS_DIR / "kpt_001" / "weights" / "best.pt"))
preds = predictor.predict(
    source=str(YOLO_DS_DIR / "images" / "val"),
    imgsz=640, conf=0.25, device="0",
    save=False, save_txt=False, verbose=False, stream=True,
)

# Map filename -> {keypoint_name: (x_px, y_px, conf)} for the top-confidence instance.
predicted = {}
for r in preds:
    name = Path(r.path).name
    if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
        continue
    best_idx = int(r.boxes.conf.argmax())
    kxy = r.keypoints.xy[best_idx].cpu().numpy()        # (K, 2)
    kconf = r.keypoints.conf[best_idx].cpu().numpy()     # (K,)
    predicted[name] = {
        kp_name: (float(kxy[i, 0]), float(kxy[i, 1]), float(kconf[i]))
        for i, kp_name in enumerate(CANONICAL_SIDE_KEYPOINTS)
    }
print(f"Got predictions for {len(predicted)}/{len(val_samples)} val images")

# Cell 15: load sticker mask areas for val (reused from baseline path)
from cattle_phenotyping.data.mask_io import load_sticker_areas
mask_jobs = [(i, s.mask_path, s.batch) for i, s in enumerate(val_samples) if s.mask_path is not None]
sticker_areas_by_idx = load_sticker_areas(mask_jobs, workers=8)
# Re-key by filename for build_eval_report.
sticker_areas_by_name = {
    val_samples[i].image_path.name: a
    for i, a in sticker_areas_by_idx.items() if a is not None
}
```

### 5. Eval — one call into the package (cells 16-18)

```python
# Cell 16: full eval report (per-keypoint pixel error + forward Schaeffer)
from cattle_phenotyping.eval.keypoint_eval import build_eval_report
report = build_eval_report(
    predictions=predicted,
    samples=val_samples,
    sticker_area_px_by_filename=sticker_areas_by_name,
    sticker_area_cm2_by_batch=sticker_by_batch,
    conf_threshold=0.0,   # accept every predicted keypoint by default
)
print(json.dumps({
    "n_samples": report["n_samples"],
    "n_with_prediction": report["n_with_prediction"],
    "skip_reasons": report["skip_reasons"],
}, indent=2))

# Cell 17: render the two metric layers
print("Per-keypoint pixel error (val):")
print(f"  {'keypoint':<22} {'n':>4}  {'median_px':>10}  {'mean_px':>10}  {'p90_px':>8}")
for kp_name in CANONICAL_SIDE_KEYPOINTS:
    s = report["pixel_errors_by_keypoint"][kp_name]
    if s["n"] == 0:
        print(f"  {kp_name:<22} {0:>4}")
        continue
    print(f"  {kp_name:<22} {s['n']:>4}  {s['median_px']:>10.2f}  {s['mean_px']:>10.2f}  {s['p90_px']:>8.2f}")

print()
print("Forward-Schaeffer with predicted keypoints (val):")
overall = report["weight_overall"]
print(f"  OVERALL  n={overall['n']}  MAE={overall['mae_kg']:.2f}  RMSE={overall['rmse_kg']:.2f}  MAPE={overall['mape_pct']:.2f}%  bias={overall['bias_kg']:+.2f}  R²={overall['r2']:.4f}")
for batch_name, m in report["weight_by_batch"].items():
    print(f"  [{batch_name}]   n={m['n']}  MAE={m['mae_kg']:.2f}  RMSE={m['rmse_kg']:.2f}  MAPE={m['mape_pct']:.2f}%  bias={m['bias_kg']:+.2f}")

print()
print("Baseline (GT keypoints, TEST split):  MAE=30.25 kg  MAPE=19.69%")
print(f"This model (PREDICTED, VAL split):    MAE={overall['mae_kg']:.2f} kg  MAPE={overall['mape_pct']:.2f}%")

# Cell 18: persist the full report
report_path = RESULTS_DIR / "yolov8s_pose_val_eval.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
print(f"Wrote {report_path}")
```

### 6. Persist artifacts to commit back (cells 23-24)

```python
# Cell 23: copy best.pt to a downloadable location with a stable name
import shutil
weights_out = Path("/kaggle/working/yolov8s_pose_kpt_001_best.pt")
shutil.copy(RUNS_DIR / "kpt_001" / "weights" / "best.pt", weights_out)
print(f"Download from: {weights_out}  ({weights_out.stat().st_size/1e6:.1f} MB)")
# Workflow: download this file and upload to GitHub Releases as
# `yolov8s_pose_kpt_001.pt`. Register hash in configs/default.yaml.

# Cell 24: what to commit back to repo
artifacts_to_commit = [
    "data/results/yolov8s_pose_val_eval.json",
    f"{RUNS_DIR}/kpt_001/results.csv",  # Ultralytics' per-epoch log
    f"{RUNS_DIR}/kpt_001/args.yaml",    # the resolved hyperparams
]
for p in artifacts_to_commit:
    if Path(p).exists():
        print(f"  ✓ {p}")
    else:
        print(f"  ✗ MISSING: {p}")
```

## Success criteria

- **Pipeline correctness:** export → train → eval runs end-to-end without exceptions.
- **Forward-Schaeffer MAE on val with predicted keypoints ≤ 40 kg** — this is the *binding* criterion. With tight cattle sigmas applied, Ultralytics' OKS now reflects real anatomy, but the forward-Schaeffer number is still the deliverable metric.
- **OKS-50 ≥ 0.85** on val *under the cattle sigma override*. With the tighter sigmas, an OKS-50 of 0.85 is now meaningful (≈median pixel error ≲ 0.03 × bbox_side on the skeletal keypoints) — unlike the 2026-05-21 run where 0.99 OKS-50 was achievable with 400 px errors.
- **Per-keypoint p90 pixel error at 4160×3120:**
  - Skeletal anchors (`wither`, `pinbone`, `shoulderbone`): ≤ 80 px.
  - Girth top/bottom keypoints: ≤ 120 px (live on body outline, fuzzier).
  - Height top/bottom: ≤ 160 px (fuzziest — top of back curve, ground line).
- **Per-batch forward-Schaeffer:** B3 and B4 should each land within 5 kg of overall MAE. A large per-batch split signals the model learned one batch's framing and not the other's.

## Open questions to revisit after the first run

- Does B4 have a systematically larger keypoint error than B3? (B4 has the larger sticker — animals are framed differently, may impact keypoint precision.)
- Are the height keypoints (mid-torso) the worst? They have the weakest visual anchor and matter least for weight via Schaeffer.
- Is 50 epochs enough, or should we promote to 80 with cosine LR? Decide from the `results.csv` curve.

## Follow-up plan after a successful train

1. Train yolov8s-seg in parallel (cattle + sticker masks) — `train_segmenter.ipynb`. Replaces SAM at inference.
2. Build the learned weight head (XGBoost or small MLP) on `(predicted_keypoints + sticker_features)` from train. Train on Kaggle train.csv, eval on test. **Target: beat 30.25 kg MAE.**
3. README + CC BY 4.0 attribution.
4. App rewrite to use the new pipeline.
