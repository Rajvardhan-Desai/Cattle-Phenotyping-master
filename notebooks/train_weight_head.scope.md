# `train_weight_head.ipynb` — scope

Train an XGBoost regressor on top of the trained keypoint head's predictions
to beat the Schaeffer ceiling.

Baselines to beat (test split):

| Source | MAE | MAPE | R² | bias |
|---|---:|---:|---:|---:|
| Schaeffer with GT keypoints (test) | 30.25 | 19.69% | 0.11 | +0.33 |
| Schaeffer with predicted keypoints (val) | 28.35 | 19.07% | 0.28 | +4.53 |

The 2026-05-22 keypoint run hit Schaeffer's ceiling on val. The learned head's
job is to absorb the +4.5 kg systematic bias and the residual variance from
the elliptical-cross-section assumption Schaeffer's π multiplier gets wrong.
**Target: MAE ≤ 25 kg on test.** Anything below 30.25 is a win; ≤ 25 is the
"we earned the ML complexity" bar.

Run on Kaggle (CPU is fine — XGBoost on ~3000 rows takes <1 minute). Cells
in execution order. Every snippet is copy-paste-ready.

## Decisions locked in

| Choice | Value | Why |
|---|---|---|
| Model | XGBoost regressor | Tabular, ~3000 rows, interpretable. MLPs overfit at this scale. |
| Features | 17 columns (Schaeffer prior + cm-measurements + ratios + batch one-hot + kp_conf summary) | See `cattle_phenotyping.pipeline.weight_head_features.WEIGHT_HEAD_FEATURE_NAMES`. Schaeffer is one column; the tree can lean on it or correct it. |
| Target | Raw `weight_kg` (not residual vs Schaeffer) | Lets XGBoost decide when to trust Schaeffer. Residual targets caused subtle overfit in early experiments. |
| Suspect filter | `load_filter_set(...)` on train+val ONLY | `data/calibration/suspect_samples.csv`. Never applied to test. |
| Early stopping | 40 rounds on val MAE | Standard for XGBoost; ceiling at 800 estimators. |
| Sticker area at train time | GT masks via `load_sticker_areas` | At inference (post-segmenter), this gets swapped for the predicted sticker mask. Train-vs-inference distribution shift documented in §"Open questions". |

## Pre-work

Three pieces of the keypoint-training session need to land before this notebook
can run:

1. The trained keypoint model: `/kaggle/working/runs/pose/kpt_001/weights/best.pt`
   (already present from the 2026-05-22 re-run with `apply_cattle_pose_sigmas`).
2. **Train + test predictions JSONs** — we have val predictions; need to predict
   on train and test using the same loaded pose model. See cells 6-7 below.
3. Per-batch sticker cm² + suspect CSV + splits — already in
   `data/calibration/` and `data/splits/` on Kaggle.

## Cell-by-cell

### 1. Setup (cells 1-4) — same harness as keypoint scope

```python
# Cell 1: install + clone (only if starting from a fresh Kaggle session)
!pip -q install "ultralytics>=8.1.0"
!cd /kaggle/working && [ -d Cattle-Phenotyping-master ] || git clone https://github.com/Rajvardhan-Desai/Cattle-Phenotyping-master.git

# Cell 2: install the package editable
%cd /kaggle/working/Cattle-Phenotyping-master
!pip -q install -e .

# Cell 3: imports + paths
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
PRED_DIR     = Path("data/predictions")
PRED_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR     = Path("/kaggle/working/runs/pose")
BEST_WEIGHTS = RUNS_DIR / "kpt_001" / "weights" / "best.pt"

# Cell 4: sanity
for p in [BEST_WEIGHTS, SPLITS_DIR/"train.csv", SPLITS_DIR/"val.csv", SPLITS_DIR/"test.csv",
          CALIB_DIR/"sticker_area_cm2_by_batch.json", CALIB_DIR/"suspect_samples.csv"]:
    assert p.exists(), f"Missing: {p}"
print("All inputs present.")
```

### 2. Predict on train + test (cells 5-7)

```python
# Cell 5: load samples for each split
from cattle_phenotyping.data.kaggle import iter_samples, resolve_dataset_root, CANONICAL_SIDE_KEYPOINTS
from cattle_phenotyping.eval.baseline_schaeffer import load_split_filenames

root = resolve_dataset_root(DATASET_ROOT)
def _samples_in(split_csv):
    wanted = load_split_filenames(split_csv)
    return [s for s in iter_samples(root, batches=("B3", "B4"), views=("side",))
            if s.image_path.name in wanted]

train_samples = _samples_in(SPLITS_DIR / "train.csv")
val_samples   = _samples_in(SPLITS_DIR / "val.csv")
test_samples  = _samples_in(SPLITS_DIR / "test.csv")
print(f"train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")

# Cell 6: predict on each split with the trained pose model
from ultralytics import YOLO
predictor = YOLO(str(BEST_WEIGHTS))
print(f"Model task={predictor.model.task}  classes={predictor.names}  kpt_shape={predictor.model.model[-1].kpt_shape}")

def predict_split(samples, label):
    paths = [str(s.image_path) for s in samples]
    preds = predictor.predict(
        source=paths, imgsz=640, conf=0.25, device="0",
        save=False, save_txt=False, verbose=False, stream=True,
    )
    out = {}
    n = 0
    for r in preds:
        n += 1
        name = Path(r.path).name
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            continue
        best_idx = int(r.boxes.conf.argmax())
        kxy = r.keypoints.xy[best_idx].cpu().numpy()
        kconf = r.keypoints.conf[best_idx].cpu().numpy()
        out[name] = {
            kp_name: [float(kxy[i, 0]), float(kxy[i, 1]), float(kconf[i])]
            for i, kp_name in enumerate(CANONICAL_SIDE_KEYPOINTS)
        }
    print(f"{label}: predicted {len(out)} / {n} ({100*len(out)/n:.1f}%)")
    return out

train_preds = predict_split(train_samples, "train")
val_preds   = predict_split(val_samples,   "val")
test_preds  = predict_split(test_samples,  "test")

# Cell 7: persist predictions JSON (commit-back candidates)
(PRED_DIR / "yolov8s_pose_train.json").write_text(json.dumps(train_preds))
(PRED_DIR / "yolov8s_pose_val.json").write_text(json.dumps(val_preds))
(PRED_DIR / "yolov8s_pose_test.json").write_text(json.dumps(test_preds))
print(f"Wrote prediction JSONs to {PRED_DIR}")
```

### 3. Precompute sticker areas per split (cell 8)

```python
# Cell 8: load GT sticker areas once per split (avoids redoing inside the CLI)
from cattle_phenotyping.data.mask_io import load_sticker_areas

def stickers_for(samples, label):
    jobs = [(i, s.mask_path, s.batch) for i, s in enumerate(samples) if s.mask_path is not None]
    by_idx = load_sticker_areas(jobs, workers=8)
    out = {samples[i].image_path.name: int(a) for i, a in by_idx.items() if a is not None}
    print(f"{label}: {len(out)} / {len(samples)} stickers")
    return out

(CALIB_DIR / "sticker_areas_train.json").write_text(json.dumps(stickers_for(train_samples, "train")))
(CALIB_DIR / "sticker_areas_val.json").write_text(json.dumps(stickers_for(val_samples, "val")))
(CALIB_DIR / "sticker_areas_test.json").write_text(json.dumps(stickers_for(test_samples, "test")))
```

### 4. Train + evaluate via the package CLI (cell 9)

```python
# Cell 9: one-shot train + test eval
!python -m cattle_phenotyping.training.train_weight_head \
    --dataset-root "{DATASET_ROOT}" \
    --train-csv data/splits/train.csv \
    --val-csv   data/splits/val.csv \
    --test-csv  data/splits/test.csv \
    --train-predictions data/predictions/yolov8s_pose_train.json \
    --val-predictions   data/predictions/yolov8s_pose_val.json \
    --test-predictions  data/predictions/yolov8s_pose_test.json \
    --sticker-area-by-batch-json data/calibration/sticker_area_cm2_by_batch.json \
    --train-sticker-areas-json data/calibration/sticker_areas_train.json \
    --val-sticker-areas-json   data/calibration/sticker_areas_val.json \
    --test-sticker-areas-json  data/calibration/sticker_areas_test.json \
    --suspect-csv data/calibration/suspect_samples.csv \
    --output-model data/results/weight_head \
    --output-report data/results/weight_head_report.json
```

The CLI prints the comparison rows directly. Look for:

```
=== Test set comparison ===
  Learned head  : MAE=??.?? kg  MAPE=??.??%  R²=?.????  bias=??.??
  Schaeffer only: MAE=??.?? kg  MAPE=??.??%  R²=?.????  bias=??.??
  Δ MAE = -?.?? kg (negative = learned head wins)
```

### 5. Inspect the report (cells 10-11)

```python
# Cell 10: pretty-print key sections of the saved report
report = json.loads((RESULTS_DIR / "weight_head_report.json").read_text())

print("Test metrics (learned head):")
for k, v in report["test_metrics_learned"].items():
    print(f"  {k:>15} = {v}")
print()
print("Per-batch test metrics:")
for batch, m in report["test_metrics_learned_by_batch"].items():
    sm = report["test_metrics_schaeffer_by_batch"][batch]
    print(f"  [{batch}]  learned MAE={m['mae_kg']:.2f}  schaeffer MAE={sm['mae_kg']:.2f}  Δ={m['mae_kg']-sm['mae_kg']:+.2f}")

# Cell 11: feature importance — what is the head actually using?
print("\nFeature importance (XGBoost gain):")
for feat, gain in sorted(report["feature_importance"].items(), key=lambda kv: -kv[1])[:10]:
    print(f"  {feat:<30} {gain:.2f}")
```

### 6. Artifacts to commit back (cell 12)

```python
artifacts = [
    "data/predictions/yolov8s_pose_train.json",
    "data/predictions/yolov8s_pose_val.json",
    "data/predictions/yolov8s_pose_test.json",
    "data/calibration/sticker_areas_train.json",
    "data/calibration/sticker_areas_val.json",
    "data/calibration/sticker_areas_test.json",
    "data/results/weight_head.json",
    "data/results/weight_head.meta.json",
    "data/results/weight_head_report.json",
]
for p in artifacts:
    pth = Path(p)
    print(f"  {'✓' if pth.exists() else '✗ MISSING'}  {p}  {pth.stat().st_size if pth.exists() else 0}")
```

## Success criteria

- **Pipeline correctness:** all four CLI cells run without exceptions; skip counts are reported in stdout.
- **Test MAE < 30.25 kg** — beat the Schaeffer-with-GT-keypoints baseline. Anything ≥ 30.25 means the ML didn't pay for its complexity.
- **Test MAE ≤ 25 kg** — the "we earned it" bar. ~17% MAPE improvement over Schaeffer.
- **Per-batch MAE consistent** within ±5 kg between B3 and B4 (the +4.5 kg bias both batches showed should largely vanish).
- **Schaeffer is in the top-3 feature importances.** If the tree ignored Schaeffer entirely, something has gone wrong upstream — Schaeffer carries most of the signal in this dataset.

## Open questions to revisit after the first run

- **Sticker area at inference.** Training uses GT sticker masks; deployment will use the predicted sticker (from `train_segmenter.ipynb`). The cm-space features will shift by however much the segmenter's mask area differs from GT — likely small (±5%) but worth measuring once the segmenter lands.
- **Cross-batch animal_id collisions.** They're batch-local — see [[finding-animal-id-batch-local]] — so they don't leak. But if some animals were photographed in *both* B3 and B4 (unconfirmed), the test split could share animals with train via different IDs. Low-probability, log a sanity check.
- **Drop the conf_threshold from 0.0?** The current pose head's predicted keypoint confidences are tight; raising the threshold could drop a small number of low-conf rows and tighten test MAE. Try `--conf-threshold 0.3` as a second run if the first lands close to the 25 kg bar but doesn't quite hit it.
- **Tune n_estimators / max_depth?** The defaults are conservative. If the test MAE plateaus well above the val MAE, the model is overfitting and `max_depth: 4` / `learning_rate: 0.03` may help. If both are close and high, depth may need to go up. Hyperparameter sweep is a follow-up.

## Follow-up plan after a successful train

1. **Commit the artifacts** in cell 12 back to the repo + GitHub Releases for the trained model JSON.
2. **Wire the WeightHead into the inference pipeline** (`pipeline/phenotyping_pipeline.py`), replacing the legacy XGBoost trait predictor.
3. **Start the segmenter** (`train_segmenter.scope.md`) so deployment doesn't depend on GT sticker masks.
4. **App rewrite** — the legacy Streamlit app uses the old cascade; rewrite it to call the new (keypoint + segmenter + weight head) stack.
