# Cattle Phenotyping

Estimate cattle body weight (kg) from a single side-view image, using a sticker reference object for absolute scale calibration. Research prototype targeting smallholder livestock contexts.

**Status (2026-05-21):** Phase 0 complete. Forward-Schaeffer baseline with per-batch sticker calibration achieves **MAE 30.25 kg / MAPE 19.69% / R² 0.11** on the held-out test split. Keypoint-detection training is the active workstream. Legacy SAM+XGBoost cascade is still in the tree but no longer the recommended path — see [Roadmap](#roadmap).

---

## Dataset and attribution

This project uses the [Kaggle `sadhliroomyprime/cattle-weight-detection-model-dataset-12k`](https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k) dataset as its sole training and evaluation source.

### Attribution (required by CC BY 4.0)

The dataset is produced by **Acme AI Ltd.** (`www.acmeai.tech`) and funded by the **Bill & Melinda Gates Foundation (BMGF)** under the *BMGF — LivestockWeight — CV* program. The Kaggle release is published by Sadhli Roomy / Acme AI. Licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

When you reuse this code, dataset, or any derived metrics, you must:
- Credit Acme AI Ltd. as the dataset creator and the Bill & Melinda Gates Foundation as the funder.
- Cite the Kaggle release: `Sadhli Roomy, Cattle Weight Detection Model + Dataset 12k, Kaggle, sadhliroomyprime/cattle-weight-detection-model-dataset-12k.`
- Preserve the CC BY 4.0 license on any redistribution of the dataset or derivative datasets.

The trained model weights produced by this repository (when released) inherit the same CC BY 4.0 obligations because they derive from a CC BY 4.0 corpus.

### What we actually use

Of the ~27k images in the full dataset, the project trains and evaluates on **B3 + B4 side-view** only — **4,549 images / 1,014 distinct `(batch, animal_id)` pairs** after filtering out ~110 suspect samples (large-residual or implausibly low weight). B2 keypoint annotations cannot be joined to weight labels and are excluded; rear-view is out of scope. See [docs/kaggle_dataset_notes.md](docs/kaggle_dataset_notes.md) for the dataset audit.

---

## How the baseline works

The dataset includes a custom **sticker** applied to each cow's body as a scale reference. Acme does **not** publish the sticker's physical size, so we back-derive it from the dataset itself using the Schaeffer weight formula (`weight_lb = (HG_in² × BL_in) / 300`):

1. **Per-batch sticker calibration** ([`pipeline/scale_calibration.py`](cattle_phenotyping/pipeline/scale_calibration.py)) — invert Schaeffer over annotated keypoints (heart-girth chord and body-length chord) on the train split only. Each image gives one equation in one unknown (px-per-cm). The sticker's pixel area then maps to a single physical cm² per batch. Result:

   | Batch | Sticker cm² | Equivalent diameter |
   |---|---:|---:|
   | B3 | 15.27 | ~4.41 cm |
   | B4 | 79.33 | ~10.05 cm |

   The two batches use **physically different stickers**. A single global constant produces MAE ≈ 96 kg with opposite-sign per-batch bias; per-batch constants drop test MAE to 30.25 kg. **Never use one global sticker size for this dataset.**

2. **Forward Schaeffer** ([`eval/baseline_schaeffer.py`](cattle_phenotyping/eval/baseline_schaeffer.py)) — apply the same formula in the forward direction with the per-batch sticker constant, using ground-truth keypoints + ground-truth sticker mask. This is the **published smallholder formula Acme themselves teach** and is the bar every learned model must beat.

3. **Suspect filtering** ([`eval/flag_suspects.py`](cattle_phenotyping/eval/flag_suspects.py)) — drop train/val rows where forward Schaeffer disagrees with the label by ≥2.5σ (`large_residual`) or predicts < 50 kg (`implausible_low_weight`). About 110–115 samples (~3%) drop. The `cross_batch_id_collision` flag is **diagnostic only** — `animal_id` is batch-local, so apparent collisions across B3/B4 are coincidental numeric ID matches between unrelated animals.

---

## Repo layout

```
cattle_phenotyping/
├── data/
│   ├── kaggle.py              # COCO parser + KaggleSample iterator (B3/B4 side-view)
│   └── mask_io.py             # Sticker / cow mask pixel-area loader (parallel)
├── pipeline/
│   ├── scale_calibration.py   # Back-derive sticker cm² per batch (CLI)
│   ├── keypoint_scale_features.py
│   ├── feature_extractor.py   # legacy
│   └── phenotyping_pipeline.py # legacy cascade
├── eval/
│   ├── baseline_schaeffer.py  # Forward-Schaeffer eval on a split (CLI)
│   ├── flag_suspects.py       # Suspect-sample flagger (CLI)
│   └── keypoint_eval.py       # build_eval_report() — pixel error + forward Schaeffer
├── training/
│   ├── build_splits.py        # Animal-grouped, weight-stratified splits (CLI)
│   ├── export_yolo_pose.py    # Export Kaggle → Ultralytics YOLOv8-pose format (CLI)
│   ├── audit_dataset.py
│   └── train_trait_model.py   # legacy
├── models/
│   ├── schaeffer.py           # Pure-function Schaeffer formula
│   ├── detector_yolov8.py     # legacy
│   ├── segmenter_sam.py       # legacy
│   ├── trait_model_xgboost.py # legacy
│   └── keypoint_scale_weight_model.py
├── utils/                     # config, logging, seeding, run-dir, visualization
├── app/streamlit_app.py       # legacy — slated for rewrite
└── cli.py                     # `python -m cattle_phenotyping.cli` (legacy pipeline)

notebooks/
├── inspect_kaggle.ipynb           # Phase 0 dataset audit (run on Kaggle)
└── train_keypoints.scope.md       # YOLOv8s-pose training scope (paste-ready into Kaggle)

data/                              # gitignored at runtime; produced on Kaggle
├── splits/{train,val,test}.csv
├── calibration/
│   ├── sticker_area_cm2_by_batch.json     # source of truth: {B3: 15.27, B4: 79.33}
│   ├── scale_calibration_train.json
│   ├── suspect_samples.csv
│   └── suspect_samples_summary.json
└── results/
    └── baseline_schaeffer_test_perbatch.json

docs/
├── kaggle_dataset_notes.md     # dataset audit (filename grammars, keypoint schemas, etc.)
└── kaggle_model_integration.md

tests/                          # pytest; 159 tests passing on CPU
```

`main.py` at the repo root is a thin shim around `cattle_phenotyping.cli` (the legacy single-image cascade) — kept for backward compatibility, not the recommended entry point.

---

## Setup

```bash
pip install -e .
python -m pytest tests/   # should report 159 passing
```

The Kaggle dataset (~48 GB) is not downloaded locally; all dataset-touching commands run on Kaggle/Colab and consume `/kaggle/input/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k/`. See [docs/kaggle_dataset_notes.md](docs/kaggle_dataset_notes.md) for the Kaggle CLI download workflow if you do want it locally.

---

## Reproducing the baseline

The four CLIs below reproduce the 30.25 kg / 19.69% MAPE test baseline from scratch. Run on Kaggle (the dataset is too large for most local machines).

```bash
# 1. Build animal-grouped, weight-stratified splits (B3+B4 side, ~4,549 samples)
python -m cattle_phenotyping.training.build_splits \
  --dataset-root "$DATASET_ROOT" \
  --output-dir data/splits \
  --seed 42

# 2. Back-derive per-batch sticker cm² on TRAIN ONLY (no test leakage)
python -m cattle_phenotyping.pipeline.scale_calibration \
  --dataset-root "$DATASET_ROOT" \
  --split-csv data/splits/train.csv \
  --load-masks --workers 8 \
  --output data/calibration/scale_calibration_train.json \
  --by-batch-output data/calibration/sticker_area_cm2_by_batch.json

# 3. Flag suspect train/val samples (large residual, implausible weight)
python -m cattle_phenotyping.eval.flag_suspects \
  --dataset-root "$DATASET_ROOT" \
  --train-csv data/splits/train.csv \
  --val-csv data/splits/val.csv \
  --test-csv data/splits/test.csv \
  --sticker-area-by-batch-json data/calibration/sticker_area_cm2_by_batch.json \
  --workers 8 \
  --output data/calibration/suspect_samples.csv \
  --summary-output data/calibration/suspect_samples_summary.json

# 4. Forward-Schaeffer baseline on the held-out test split
python -m cattle_phenotyping.eval.baseline_schaeffer \
  --dataset-root "$DATASET_ROOT" \
  --split-csv data/splits/test.csv \
  --sticker-area-by-batch-json data/calibration/sticker_area_cm2_by_batch.json \
  --workers 8 \
  --output data/results/baseline_schaeffer_test_perbatch.json
```

Expected test-split output:

```
OVERALL  n=≈900  MAE=30.25  RMSE=≈45  MAPE=19.69%  bias=+0.33  R²=0.11
[B3]     MAE≈30  bias≈+0.3
[B4]     MAE≈30  bias≈+0.3
```

---

## Known limitations

- **Single-domain.** All training data is Bangladeshi zebu (*Bos indicus*) from one collection program. Performance on other breeds, regions, ages, or camera setups is **undefined** and the model should not be deployed outside that domain without further validation.
- **Sticker is a hard dependency.** The pipeline assumes the same fiducial sticker is in frame at inference as at training time. Without it, scale calibration fails and there is no quiet fallback — by design.
- **Per-batch sticker constants.** Any new batch (e.g. a hypothetical B5) needs its own sticker cm² re-derivation. Do not reuse B3's or B4's constant.
- **Animal IDs are batch-local.** Use the tuple key `(batch, animal_id)` for any join or split key. Raw `animal_id` collisions across batches are coincidental and must not be treated as the same animal.
- **No BCS.** The Kaggle dataset has no Body Condition Score labels, so BCS is dropped from the project deliverable. Heuristic BCS predictions in `models/trait_model_xgboost.py` are legacy and should not be shipped.
- **Legacy SAM cascade.** The original YOLOv8 → SAM → OpenCV → XGBoost cascade in `pipeline/phenotyping_pipeline.py` predates the Kaggle pivot. It is kept in the tree for reference but is not the recommended inference path.

---

## Roadmap

1. **Keypoint detector — YOLOv8s-pose on B3+B4 side, 9 canonical keypoints.** Training scope is paste-ready in [notebooks/train_keypoints.scope.md](notebooks/train_keypoints.scope.md); target is forward-Schaeffer MAE ≤ 40 kg on val with predicted keypoints. Currently *in progress* on Kaggle.
2. **Segmenter — YOLOv8s-seg on cow + sticker masks.** Replaces SAM at inference. Scope doc next.
3. **Learned weight head.** XGBoost or small MLP on `(predicted keypoints + sticker features)`. Trained on train.csv, evaluated against the 30.25 kg / 19.69% MAPE test baseline.
4. **App rewrite.** Replace the legacy Streamlit app with one that calls the new keypoint + segmenter + scale-calibration + learned-weight stack.

Trained model weights will be published as artifacts via GitHub Releases (registered in `configs/default.yaml` by SHA-256) rather than committed into the repo.

---

## References

- **Dataset:** Sadhli Roomy / Acme AI Ltd., *Cattle Weight Detection Model + Dataset 12k*, Kaggle, [sadhliroomyprime/cattle-weight-detection-model-dataset-12k](https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k). CC BY 4.0. Funded by the Bill & Melinda Gates Foundation.
- **Acme AI methodology brief:** `www.acmeai.tech BMGF - LivestockWeight - CV.pdf` (bundled inside the Kaggle dataset).
- **Weight formula:** Schaeffer (heart girth² × body length / 300, in inches → pounds) — the published smallholder formula Acme references.
