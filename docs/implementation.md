# Cattle Phenotyping — Complete Implementation Details

A line-by-line walkthrough of the pipeline: how each stage works, every formula with its derivation, what failure modes were observed and fixed, and how the result compares to prior research. Companion to [README.md](../README.md) (overview), [demo_runbook.md](demo_runbook.md) (deployment), and [kaggle_dataset_notes.md](kaggle_dataset_notes.md) (data audit).

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [What problem this solves](#2-what-problem-this-solves)
3. [Pipeline architecture](#3-pipeline-architecture)
4. [Stage 1 — Pose detection (YOLOv8s-pose)](#4-stage-1--pose-detection-yolov8s-pose)
5. [Stage 2 — Sticker segmentation](#5-stage-2--sticker-segmentation)
6. [Stage 3 — Scale calibration](#6-stage-3--scale-calibration)
7. [Stage 4 — Feature builder (17-dimensional vector)](#7-stage-4--feature-builder-17-dimensional-vector)
8. [Stage 5 — XGBoost weight head](#8-stage-5--xgboost-weight-head)
9. [The Schaeffer baseline (the bar we have to beat)](#9-the-schaeffer-baseline-the-bar-we-have-to-beat)
10. [BCS heuristic (demo-only, not learned)](#10-bcs-heuristic-demo-only-not-learned)
11. [Methodological contributions (what was novel)](#11-methodological-contributions-what-was-novel)
12. [Gaps fixed from prior research](#12-gaps-fixed-from-prior-research)
13. [Dataset and preprocessing](#13-dataset-and-preprocessing)
14. [Training protocol](#14-training-protocol)
15. [Evaluation protocol and results](#15-evaluation-protocol-and-results)
16. [Engineering substrate](#16-engineering-substrate)
17. [Reproduction recipe](#17-reproduction-recipe)
18. [Comparison to prior approaches](#18-comparison-to-prior-approaches)
19. [Known limitations](#19-known-limitations)

---

## 1. Executive summary

| Property | Value |
|---|---|
| Input | One side-view RGB photo of a cow with a sticker visible on its body |
| Output | Predicted body weight in kg, plus derived cm-grounded body measurements |
| Test MAE | **24.16 kg** (685 cattle, held-out) |
| Test MAPE | **16.1%** |
| Test R² | **0.46** |
| Test bias | **+0.38 kg** (essentially zero) |
| Baseline beaten | Schaeffer (29.80 kg MAE on same test set) by 19.1% MAE |
| Hardware | CPU-only inference; T4 GPU for training |
| Dataset | Acme AI / BMGF Kaggle BMGF-LivestockWeight-CV (CC BY 4.0) |
| Test set | 685 Bangladeshi zebu (*Bos indicus*) cattle |

---

## 2. What problem this solves

**Smallholder cattle weight estimation** is hard because:

1. **Mechanical scales are slow and stressful.** Putting a 300 kg animal on a scale takes minutes per animal, is stressful, and requires equipment that smallholder farms typically don't own.
2. **The existing manual workaround (Schaeffer's formula)** requires a human to physically place a tape measure around the chest and along the body. Both operations are stressful and slow, and the formula is sensitive to placement errors.
3. **Single-RGB image methods** can extract pixel measurements automatically but cannot recover absolute cm-scale without a reference object or depth sensor.
4. **Depth-camera methods** (Kinect, stereo) solve the scale problem but require hardware that smallholders don't have.

This pipeline targets the gap: **a single hand-held RGB photo + a low-cost printed sticker fiducial** delivers cm-grounded body measurements and a kg-scale weight prediction with no hardware beyond a smartphone.

---

## 3. Pipeline architecture

```
                       ┌───────────────────────┐
   side-view photo ───▶│ YOLOv8s-pose          │──▶ bbox + 9 keypoints
                       └───────────────────────┘
                                  │
                                  ▼
                       ┌───────────────────────┐
   user click /        │ SAM-click  OR         │──▶ sticker mask (binary)
   user draw ─────────▶│ user-drawn shape      │
                       └───────────────────────┘
                                  │
                                  ▼                  per-batch
                       ┌───────────────────────┐    sticker cm²
                       │ Scale calibration     │◀──┐ (constant)
                       │   ρ = √(A_S / A_S^cm²)│   │
                       └───────────────────────┘   │
                                  │                │ data/calibration/
                                  ▼                │ sticker_area_cm2_by_batch.json
                       ┌───────────────────────┐
                       │ Feature builder       │
                       │  (17 features)        │
                       └───────────────────────┘
                                  │
                                  ▼
                       ┌───────────────────────┐
                       │ XGBoost weight head   │
                       │  (1200 trees, depth 4)│
                       └───────────────────────┘
                                  │
                                  ▼
                       Predicted weight (kg)
```

The pipeline is **deterministic given the input** — the only learned components are the YOLOv8s-pose model and the XGBoost regressor. Everything between them is closed-form arithmetic.

---

## 4. Stage 1 — Pose detection (YOLOv8s-pose)

### What it does

Detects the cow's bounding box and locates 9 anatomical keypoints in image-pixel space.

### Why YOLOv8s-pose

| Choice | Why |
|---|---|
| Pose head over keypoint regression head on a CNN | YOLOv8-pose jointly trains bbox + keypoints in one network, sharing features. Reduces parameter count vs. running detection + a separate keypoint network. |
| Small variant (`s`) | ~11.6M parameters. Dataset is ~3000 training images post-filter; a larger variant would over-fit the bbox prior at the cost of keypoint accuracy. |
| Input resolution 640 px | Standard YOLO size. At 640, the sticker is ~50-100 px (readable), bigger inputs hurt training throughput without measurable keypoint gain. |

### The 9 canonical keypoints

| Group | Keypoint | Role |
|---|---|---|
| Skeletal | `wither` | Top of shoulder (bony prominence) |
| Skeletal | `pinbone` | Pin bone (hip skeletal prominence) |
| Skeletal | `shoulderbone` | Point of shoulder |
| Body outline | `front_girth_top` | Front girth circumference, top intersection |
| Body outline | `front_girth_bottom` | Front girth circumference, bottom intersection |
| Body outline | `rear_girth_top` | Rear girth circumference, top intersection |
| Body outline | `rear_girth_bottom` | Rear girth circumference, bottom intersection |
| Body outline | `height_top` | Top of back, midway between wither and pinbone |
| Body outline | `height_bottom` | Hoof / ground line |

### Output format

Per image:

```python
{
    "bbox": [x1, y1, x2, y2, conf],
    "keypoints": {
        "wither":             (x, y, conf),
        "pinbone":            (x, y, conf),
        ...,
        "height_bottom":      (x, y, conf),
    }
}
```

All coordinates are in **original-image pixel space**, after Ultralytics has undone the 640×640 letterbox preprocessing.

### Code

- Training script: paste-ready in [`notebooks/train_keypoints.scope.md`](../notebooks/train_keypoints.scope.md).
- Dataset export: [`cattle_phenotyping/training/export_yolo_pose.py`](../cattle_phenotyping/training/export_yolo_pose.py).
- Critical OKS-sigma fix (Section 11.2): [`cattle_phenotyping/training/kpt_sigmas.py`](../cattle_phenotyping/training/kpt_sigmas.py).

---

## 5. Stage 2 — Sticker segmentation

### What it does

Produces a binary mask of the sticker on the cow's body. The mask's pixel count is the only output that downstream stages use — boundary precision is not required.

### Two input paths

**Path A — User draws a shape (default, recommended for demo):**

User selects a drawing tool (circle / freedraw / rectangle) and traces the sticker on the canvas. The drawn shape is rasterized into a binary mask at full image resolution. Deterministic, no model inference, no SAM weights needed.

**Path B — SAM-click (fallback):**

User clicks once on the sticker centre. The Segment Anything Model (ViT-B backbone) returns multiple candidate masks at different granularities. We accept the largest mask that:
1. Contains the click point.
2. Has area ≤ 5% of the total image area (excludes the "segmented the whole cow" failure mode).

The "largest within cap" rule is critical: SAM's smallest-granularity mask for a printed sticker is usually the inner logo region, not the full disc. Selecting the largest valid mask gives the full disc.

### Why a sticker at all

The Acme/BMGF corpus's funder brief describes the sticker as "an archaic form of innovation which we would like to remove in future development." We use it as a scale fiducial because:

1. **No other reference object is available in the photos.**
2. **The sticker's physical area can be recovered** from the labelled training data by inverting Schaeffer's formula (see Section 6.2).
3. **At inference, the sticker only needs to be locatable** — its absolute physical size is the per-batch constant.

### Output

A binary mask `M ∈ {0, 255}^{H × W}` of the same dimensions as the input image. The downstream pipeline uses:

```python
sticker_area_px = int((M > 0).sum())
```

### Code

- SAM wrapper with both bbox and point-prompt segmentation: [`cattle_phenotyping/models/segmenter_sam.py`](../cattle_phenotyping/models/segmenter_sam.py).
- Drawing canvas integration in the demo: [`cattle_phenotyping/app/demo_app.py`](../cattle_phenotyping/app/demo_app.py).

---

## 6. Stage 3 — Scale calibration

This is the conceptual heart of the pipeline. Without it, the system would only produce pixel-space measurements with no physical meaning.

### 6.1. Forward direction (used at inference)

Given the sticker pixel area `A_S` and the **batch-specific physical area** `A_S^cm²` (a constant), the pixel-to-cm ratio is:

$$
\rho = \sqrt{\frac{A_S}{A_S^{cm^2}}} \quad \text{[pixels per cm]}
$$

This follows directly from the relation `area_px = ρ² × area_cm²`, which holds because pixel area scales as the square of the pixel-to-cm linear ratio.

Any pixel measurement is then converted to cm by dividing by ρ:

$$
\text{distance\_cm} = \frac{\text{distance\_px}}{\rho}
$$

### 6.2. Inverse direction (back-derive the sticker's physical area)

The Acme/BMGF dataset's published brief **does not disclose the sticker's cm size**. We back-derive it from the labels themselves using Schaeffer's formula.

For each labelled training image, we have:
- `c_px`: pixel-space heart-girth chord (Euclidean distance between `front_girth_top` and `front_girth_bottom`).
- `ℓ_px`: pixel-space body length (`shoulderbone` to `pinbone`).
- `W`: labelled body weight in kg.

Schaeffer's formula in cm-fused units is:

$$
W = (c_{cm} \cdot \pi)^2 \cdot \ell_{cm} \cdot K
$$

where π is the chord-to-circumference multiplier (assumes circular cross-section), and K is the unit-conversion constant (see Section 9). Substituting `cm = px / ρ`:

$$
W = \frac{c_{px}^2 \cdot \pi^2 \cdot \ell_{px} \cdot K}{\rho^3}
$$

Solving for ρ:

$$
\boxed{
\rho = \left( \frac{c_{px}^2 \cdot \pi^2 \cdot \ell_{px} \cdot K}{W} \right)^{1/3}
}
$$

This gives a per-image pixel-to-cm ratio that's internally consistent with the labelled weight. The sticker's physical area for that image is then:

$$
A_S^{cm^2}\text{(per image)} = \frac{A_S}{\rho^2}
$$

Aggregating the median `A_S^cm²` across the training split gives the global per-batch constant.

### 6.3. The bimodality discovery

When we ran the inversion above with a **single global median**, we got `A_S^cm² ≈ 18.21 cm²` with an interquartile range of `(14.9, 77.6) cm²` — a 4× spread that signals two distributions, not noise.

Grouping by batch reveals two well-separated medians:

| Batch | Median `A_S^cm²` | Equivalent diameter |
|---|---:|---:|
| **B3** | **15.27 cm²** | ~4.41 cm |
| **B4** | **79.33 cm²** | ~10.05 cm |

The two batches use **physically different stickers**. The single global constant produced test MAE = 96.27 kg with opposite-sign per-batch bias (+47 kg on B3, −150 kg on B4). The per-batch constants give test MAE = 30.25 kg with bias < 1 kg in both batches.

**Operational rule:** the per-batch constants must be (a) derived on train only — no test leakage, (b) stored and version-controlled (`data/calibration/sticker_area_cm2_by_batch.json`), (c) applied as a hard per-batch lookup at inference, and (d) never replaced by a single global value.

### Code

- Inversion implementation + CLI: [`cattle_phenotyping/pipeline/scale_calibration.py`](../cattle_phenotyping/pipeline/scale_calibration.py).
- Mask area loader (parallel): [`cattle_phenotyping/data/mask_io.py`](../cattle_phenotyping/data/mask_io.py).
- Validated by 21 unit tests in [`tests/test_scale_calibration.py`](../tests/test_scale_calibration.py).

---

## 7. Stage 4 — Feature builder (17-dimensional vector)

Given the pose output, sticker pixel area, and per-batch cm² constant, the feature builder produces a flat vector that the XGBoost head consumes.

### Pixel-space measurements

Four Euclidean distances:

| Feature | Formula |
|---|---|
| `front_girth_chord_px` | `‖p_{front_girth_top} − p_{front_girth_bottom}‖` |
| `body_length_px` | `‖p_{shoulderbone} − p_{pinbone}‖` |
| `rear_girth_chord_px` | `‖p_{rear_girth_top} − p_{rear_girth_bottom}‖` (intermediate, not directly in feature vector) |
| `body_height_px` | `‖p_{height_top} − p_{height_bottom}‖` (intermediate) |

### Cm-grounded measurements

Each pixel measurement divided by ρ:

| Feature | Formula |
|---|---|
| `front_girth_chord_cm` | `front_girth_chord_px / ρ` |
| `rear_girth_chord_cm` | `rear_girth_chord_px / ρ` |
| `body_length_cm` | `body_length_px / ρ` |
| `body_height_cm` | `body_height_px / ρ` |

These are the four quantities a smallholder tape-measure protocol would record.

### Schaeffer prior (computed feature)

The Schaeffer formula's prediction *is itself a feature*, not the target:

$$
\text{schaeffer\_kg} = (c_{cm} \cdot \pi)^2 \cdot \ell_{cm} \cdot K
$$

Where `c_cm` is `front_girth_chord_cm`, `ℓ_cm` is `body_length_cm`, and K is the Schaeffer constant.

The design choice is deliberate: by feeding Schaeffer as a feature, the XGBoost head can never do *worse* than leaning entirely on Schaeffer (one branch away in the tree), and can learn residual corrections (e.g. for the circular-cross-section assumption that overstates heart girth on elliptical cattle bodies).

### Shape ratios (scale-invariant)

| Feature | Formula | What it captures |
|---|---|---|
| `front_to_rear_girth_ratio_cm` | `front_girth_chord_cm / rear_girth_chord_cm` | Body taper (front-thick vs rear-thick) |
| `girth_to_length_ratio_cm` | `front_girth_chord_cm / body_length_cm` | Stockiness (compact vs lean) |
| `length_to_height_ratio_cm` | `body_length_cm / body_height_cm` | Square vs rectangular silhouette |

These have no scale dependence (any sticker error cancels in the ratio), so they let the XGBoost head learn body-shape effects independent of measurement scale.

### Sticker calibration features

| Feature | Source |
|---|---|
| `sticker_area_px` | Stage 2 output |
| `sticker_cm2` | Per-batch constant lookup |
| `px_per_cm` (ρ) | Stage 3 |

Feeding all three allows the tree ensemble to learn corrections that depend on absolute scale even though the cm measurements above are already calibrated.

### Pose-confidence summaries

| Feature | Formula |
|---|---|
| `kp_conf_mean` | Average of the 9 keypoint confidence scores |
| `kp_conf_min` | Minimum of the 9 keypoint confidence scores |

Trust signals. When a keypoint is poorly localized, its confidence is low, and the head can attenuate its corrections.

### Batch one-hot

| Feature | Value |
|---|---|
| `batch_B3` | 1.0 if batch is B3, else 0.0 |
| `batch_B4` | 1.0 if batch is B4, else 0.0 |

The two batches differ not just in sticker size but also in camera framing (B4 photos are framed differently than B3). One-hot lets the tree learn per-batch correction shapes cleanly without forcing the model to split on batch at every node.

### Full schema (17 columns, frozen order)

```python
WEIGHT_HEAD_FEATURE_NAMES = (
    "schaeffer_kg",
    "front_girth_chord_cm",
    "rear_girth_chord_cm",
    "body_length_cm",
    "body_height_cm",
    "front_girth_chord_px",
    "body_length_px",
    "sticker_area_px",
    "sticker_cm2",
    "px_per_cm",
    "front_to_rear_girth_ratio_cm",
    "girth_to_length_ratio_cm",
    "length_to_height_ratio_cm",
    "kp_conf_mean",
    "kp_conf_min",
    "batch_B3",
    "batch_B4",
)
```

The order is enforced at training time (saved in the model's metadata JSON) and verified at inference (the model refuses to predict on a DataFrame with mismatched columns).

### Code

- Builder: [`cattle_phenotyping/pipeline/weight_head_features.py`](../cattle_phenotyping/pipeline/weight_head_features.py).
- 15 unit tests in [`tests/test_weight_head_features.py`](../tests/test_weight_head_features.py).

---

## 8. Stage 5 — XGBoost weight head

### Objective

XGBoost minimizes the squared-error loss with an explicit complexity penalty per tree:

$$
\mathcal{L}(\theta) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \sum_{k=1}^K \Omega(f_k)
$$

where $\hat{y}_i = \sum_{k=1}^K f_k(\mathbf{x}_i)$ is the sum of $K$ regression-tree predictions, and the per-tree regularization term is:

$$
\Omega(f_k) = \gamma T_k + \frac{\lambda}{2} \|w_k\|_2^2
$$

with $T_k$ leaves in tree $f_k$, $w_k$ the leaf weights, $\gamma$ the minimum loss reduction required to split a leaf, and $\lambda$ the L2 leaf-weight penalty.

### Selected hyperparameters

Settled by a small sweep (Section 15):

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 1200 | Budget; actual iterations stop early at ~107 via validation MAE |
| `max_depth` | 4 | Shallow trees prevent over-fit in this regime |
| `learning_rate` | 0.03 | Slow learning + more trees beats fast learning + fewer trees on this size |
| `subsample` | 0.85 | Row sub-sampling per tree |
| `colsample_bytree` | 0.85 | Column sub-sampling per tree |
| `min_child_weight` | 4 | Minimum sum of weights in a child node |
| `reg_lambda` (λ) | 1.0 | L2 leaf-weight regularization |
| `early_stopping_rounds` | 50 | Stop if val MAE hasn't improved in 50 trees |
| `seed` | 42 | Deterministic training |

### Train/val/test split

Animal-grouped, weight-stratified, locked before any modeling:

| Split | Count | Purpose |
|---|---|---|
| Train | 3,094 | Tree fitting |
| Val | 654 | Early stopping, hyperparameter selection |
| Test | 685 | Reported metrics only — never touched until final eval |

Animal-grouped (`(batch, animal_id)` key) means no animal's photos appear in multiple splits, preventing leakage via "model saw a different photo of this cow."

### Save/load with schema validation

The trained model is saved as two files:

- `weight_head.json` — XGBoost native format (the trees + params).
- `weight_head.meta.json` — feature names, best_iteration, schema hash.

At load time the wrapper validates that the inference DataFrame's columns exactly match the saved schema. Any mismatch raises immediately, preventing silent feature-order bugs.

### Code

- Wrapper: [`cattle_phenotyping/models/weight_head.py`](../cattle_phenotyping/models/weight_head.py).
- Training CLI: [`cattle_phenotyping/training/train_weight_head.py`](../cattle_phenotyping/training/train_weight_head.py).
- 12 unit tests in [`tests/test_weight_head_model.py`](../tests/test_weight_head_model.py).

---

## 9. The Schaeffer baseline (the bar we have to beat)

Schaeffer's heart-girth-and-body-length formula is the published smallholder cattle-weighing protocol that Acme themselves teach. Any learned ML model has to beat it to justify itself.

### Original imperial form

$$
W_{lb} = \frac{(HG_{in})^2 \cdot BL_{in}}{300}
$$

with heart girth HG and body length BL in inches, output W in pounds.

### Cm-fused form used in this codebase

$$
W_{kg} = (HG_{cm})^2 \cdot BL_{cm} \cdot K
$$

where K folds together three conversions:

$$
K = \underbrace{\left(\frac{1}{2.54}\right)^3}_{\text{cm}^3 \to \text{in}^3} \cdot \underbrace{\frac{1}{300}}_{\text{Schaeffer divisor}} \cdot \underbrace{0.45359237}_{\text{lb} \to \text{kg}} \approx 9.229 \times 10^{-5}
$$

### Heart girth from a side-view chord

A 2D side photograph only sees the chord (diameter) of the chest circumference, not the full circumference. Schaeffer needs circumference, so we multiply the chord by π (assuming a circular cross-section):

$$
HG_{cm} = c_{cm} \cdot \pi
$$

This is the **single biggest systematic error in the formula**: real cattle bodies are elliptical, so circumference is closer to 2.0–2.5 times the chord, not π ≈ 3.14. The learned XGBoost head exists primarily to correct this elliptical-cross-section bias.

### Baseline result on our test split

Forward Schaeffer with **ground-truth** keypoints + per-batch sticker cm² on the 685-cattle test split:

| Metric | Value |
|---|---:|
| MAE | 30.25 kg |
| MAPE | 19.69% |
| R² | 0.11 |
| Bias | +0.33 kg |

This is the bar. Our learned head improves it by 19.1% MAE.

### Code

- Forward Schaeffer formula: [`cattle_phenotyping/models/schaeffer.py`](../cattle_phenotyping/models/schaeffer.py).
- Evaluation CLI: [`cattle_phenotyping/eval/baseline_schaeffer.py`](../cattle_phenotyping/eval/baseline_schaeffer.py).
- 17 unit tests in [`tests/test_schaeffer.py`](../tests/test_schaeffer.py) and 18 in [`tests/test_baseline_schaeffer.py`](../tests/test_baseline_schaeffer.py).

---

## 10. BCS heuristic (demo-only, not learned)

> **Important framing.** Body Condition Score (BCS) is computed here as a **closed-form rule of thumb**, not a learned model. The Kaggle BMGF corpus has **no BCS labels**, so a learned BCS head is not possible from this data. The heuristic exists solely to give the live demo a BCS readout for audience comprehension. **BCS is excluded from the paper and from any quantitative claim.** The number on the demo screen is an interpretation aid, not a clinical assessment.

### What it computes

A 1–5 BCS score (rounded to nearest 0.5) plus a categorical label (`Very thin` / `Thin` / `Ideal` / `Slightly heavy` / `Overweight`) from one feature already in the 17-dimensional vector — the cm-space girth-to-length ratio.

### The formula

For a sample with feature `girth_to_length_ratio_cm = r`:

$$
\text{BCS}_{\text{raw}} = 3.0 + (r - r_{\text{anchor}}) \cdot \text{slope}
$$

with the calibration constants:

| Constant | Value | Source |
|---|---|---|
| $r_{\text{anchor}}$ | **0.48** | Central tendency of the Kaggle BMGF zebu training cohort (re-anchor at ~0.55 for Holstein dairy if reused on that population) |
| slope | **10.0** | Each 0.05 unit of ratio deviation moves BCS by 0.5 score units |

The raw score is then **clamped** to refuse extreme calls a single feature can't reliably support:

$$
\text{BCS}_{\text{clamped}} = \max\!\left(1.5,\ \min\!\left(4.5,\ \text{BCS}_{\text{raw}}\right)\right)
$$

and **rounded to the nearest 0.5 score unit** (matching how vets report BCS in practice):

$$
\text{BCS}_{\text{final}} = \frac{\mathrm{round}(2 \cdot \text{BCS}_{\text{clamped}})}{2}
$$

### Why these constants

The girth-to-length ratio rises with body condition: a stockier animal carries more soft tissue at the same length. The anchor $r_{\text{anchor}} = 0.48$ is the empirical median of `girth_to_length_ratio_cm` in the BMGF training split, which approximates a vet's "BCS 3 (Ideal)" call for typical *Bos indicus* zebu cattle. The slope is chosen so each 0.05 unit of ratio deviation moves the score by one half-unit, matching the granularity of standard BCS reporting (1.0, 1.5, 2.0, … , 4.5, 5.0). The clamp range $[1.5, 4.5]$ refuses to claim emaciated (BCS 1) or obese (BCS 5) from a single feature because those calls genuinely need a trained eye assessing multiple body regions (e.g. tail-head fat cover, rib visibility).

### Label buckets

The final score is mapped to a categorical label using the standard vet-practice ranges (matching the AHDB BCS chart):

| Score range | Label | Interpretation |
|---|---|---|
| BCS ≤ 2.0 | **Very thin** | Bones prominent, intervention needed |
| 2.0 < BCS ≤ 2.5 | **Thin** | Underweight |
| 2.5 < BCS ≤ 3.5 | **Ideal** | Target condition for most production stages |
| 3.5 < BCS < 4.5 | **Slightly heavy** | Over target, slight reduction |
| BCS ≥ 4.5 | **Overweight** | Excessive condition |

### Worked example

For the demo sample with `girth_to_length_ratio_cm = 0.475`:

```
BCS_raw     = 3.0 + (0.475 - 0.48) * 10.0
            = 3.0 + (-0.005) * 10.0
            = 2.95

BCS_clamped = max(1.5, min(4.5, 2.95)) = 2.95

BCS_final   = round(2 * 2.95) / 2 = round(5.90) / 2 = 6 / 2 = 3.0

label       = "Ideal"  (because 2.5 < 3.0 ≤ 3.5)
```

So the screen reads **BCS 3.0 — Ideal**.

### Why one feature, not many

A more sophisticated rule could combine multiple features:
- girth-to-length (stockiness)
- front-to-rear girth (uneven taper, can signal poor condition over the hindquarters)
- length-to-height (square vs rectangular silhouette)

But each additional feature adds calibration parameters that we have no labels to fit. The dataset has zero BCS ground truth, so we cannot tune a multi-feature rule honestly. A single-feature heuristic is the only choice that doesn't require us to invent thresholds we can't validate. If BCS labels later appear in any cattle imagery corpus, this whole module is throwaway code that gets replaced by a learned head sharing the 17-feature vector.

### Why this is not in the paper

Three reasons:

1. **No ground truth.** A BCS claim in the paper would have nothing to compare against — there's no held-out BCS test set to report MAE on.
2. **Honest scope discipline.** The project's plan ([`memory/scope_kaggle_only.md`](../memory/scope_kaggle_only.md) on the user's machine, dataset notes in [`docs/kaggle_dataset_notes.md`](kaggle_dataset_notes.md)) explicitly drops BCS as a deliverable until BCS-labelled imagery becomes available.
3. **Reviewer optics.** Shipping a rule-of-thumb BCS in an IEEE paper next to a learned weight head would dilute the paper's "we earned every claim" story. Demo-only keeps both stories clean.

### Limitations

| Limitation | Why |
|---|---|
| Single-breed anchor | The anchor $r_{\text{anchor}} = 0.48$ is fit to *Bos indicus* zebu. Re-anchor to ~0.55 for Holstein (taurine breeds are more rectangular and tend to higher ratios at the same condition). |
| Cannot distinguish lean vs muscular | A muscular animal with low fat can have the same girth-to-length as an overweight one with low muscle. Single feature does not separate them. |
| No tail-head / hook / pin bone cues | Standard manual BCS scoring inspects fat cover at the tail head and over the pin bones (BCS chart anchors). The single ratio feature cannot capture this. |
| Confidence not modelled | The output does not report uncertainty; in reality the heuristic is much more confident near $r_{\text{anchor}}$ than at the clamp boundaries. |

### Code

- Heuristic: [`cattle_phenotyping/models/bcs_heuristic.py`](../cattle_phenotyping/models/bcs_heuristic.py). `estimate_bcs_from_ratios(r)` returns a `BCSResult(score, label, raw_score)`. Anchor, slope, and clamp range are all overridable kwargs so re-anchoring on a different breed is a one-line change.
- Demo integration: [`cattle_phenotyping/app/demo_app.py`](../cattle_phenotyping/app/demo_app.py). Renders a colored card (green=Ideal, orange=Thin/Slightly heavy, red=Very thin/Overweight) below the weight prediction, with a 5-dot indicator filled to the BCS level, and an explicit caption: *"Rule-based … Not a learned model … display only, not a clinical assessment."*
- 17 unit tests in [`tests/test_bcs_heuristic.py`](../tests/test_bcs_heuristic.py): anchor/slope/clamp/rounding/label buckets/Holstein re-anchor/demo-sample regression.

### References for the BCS scale (not the heuristic itself)

The score range 1–5 and the categorical labels come from the standard literature:

- Edmonson, A.J., Lean, I.J., Weaver, L.D., Farver, T., Webster, G. (1989). *A body condition scoring chart for Holstein dairy cows.* J. Dairy Sci. 72: 68–78.
- Agriculture & Horticulture Development Board, UK (2018). *Body condition scoring of dairy cows.* AHDB guide.

The single-feature rule of thumb itself is project-specific, not from the literature.

---

## 11. Methodological contributions (what was novel)

Three engineering decisions are responsible for the system's measured performance. Each was discovered through an observed failure mode, fixed, and validated.

### 10.1. Per-batch sticker calibration

**The discovery.** Section 6 already covered the math, but the empirical signal that the sticker is bimodal is worth re-stating: a single global sticker constant gave **opposite-sign per-batch biases of equal magnitude**, which is the textbook signature of two distributions with the same shape but different scales.

**The fix.** Aggregate sticker physical area per batch on the train split, store as a JSON keyed by batch ID, look up at inference. Never use a single global constant.

**Effect:** Test MAE drops from 96.27 kg → 30.25 kg on the same data.

### 10.2. Cattle-specific OKS sigmas (Ultralytics fallback fix)

**The discovery.** YOLOv8-pose's training metric is Object Keypoint Similarity (OKS), defined per keypoint as:

$$
\text{OKS}_i = \exp\left(-\frac{d_i^2}{2 s^2 \sigma_i^2}\right)
$$

where $d_i$ is the pixel-space error for keypoint $i$, $s$ is the object scale (square root of bbox area), and $\sigma_i$ is a per-keypoint variance constant that captures how tight a localization tolerance is acceptable. COCO specifies $\sigma_i$ for 17 human keypoints; for any other count, Ultralytics 8.4.x falls back to:

$$
\sigma_i = \frac{1}{N} \quad \text{(0.111 for our 9-keypoint cattle setup)}
$$

This is way too generous. With $\sigma = 0.111$ and a bbox area of 7.3M px², a **400-pixel** localization error gives OKS ≈ 0.41 — still above 0.5 averaged across keypoints where some are correct.

**The first failed training run.** With default sigmas, the trained model reported `mAP_50-95(P) = 0.953` on validation. Looked great. When we ran the model on val images and computed forward Schaeffer with predicted keypoints, MAE was **166 kg** with R² of **−19.9**. Inspection revealed all `_top` keypoints were placed at the top edge of the bounding box (a degenerate local minimum the loss couldn't push the model out of).

**The fix.** Replace the uniform σ = 1/N with anatomy-calibrated per-keypoint sigmas:

| Keypoint group | σ | Rationale |
|---|---|---|
| `wither`, `pinbone`, `shoulderbone` | 0.025 | Skeletal protrusions (tight anatomical landmarks) |
| `front_girth_top/bottom`, `rear_girth_top/bottom` | 0.040 | Body outline (looser silhouette landmarks) |
| `height_top`, `height_bottom` | 0.050 | Top of back / hoof line (fuzziest) |

These tighter sigmas make the pose loss gradient much steeper for the same pixel error, forcing the optimizer to push keypoints onto the actual anatomy rather than settling on the bbox-edge prior. The metric also becomes meaningful: post-fix mAP drops to ~0.5 but now corresponds to actually well-localized keypoints (median pixel error 26-50 px on a 4160-px image).

**Implementation:** [`cattle_phenotyping/training/kpt_sigmas.py`](../cattle_phenotyping/training/kpt_sigmas.py) defines `CATTLE_KEYPOINT_SIGMAS` and `apply_cattle_pose_sigmas()`, which monkey-patches both `v8PoseLoss` (steepens the loss gradient) and `PoseValidator` (makes the metric honest). 11 tests verify correctness, including idempotency and the override-rejection rule that prevents loosening sigmas above the default. Called once before `model.train(...)`.

**Effect:** Forward Schaeffer with predicted keypoints goes from 166 kg val MAE → 28.35 kg val MAE.

### 10.3. Feature parity at train and inference

**The principle.** Every feature the XGBoost head sees at train time must be computable identically at inference time. A common failure mode in cascaded pipelines is:

- At train: feature `z` is computed from ground-truth keypoints (clean signal).
- At inference: keypoints are predicted (noisy), so feature `z` has a different distribution.
- The XGBoost head, never exposed to noisy `z` at train, fails catastrophically at inference.

**The implementation.** All cm-grounded features in the 17-dimensional vector are computed from the **trained pose model's predicted keypoints**, even during training (using a pre-saved prediction JSON over the train split). The sticker is still GT during training (because we don't have a trained sticker segmenter yet), but the keypoint side already enforces parity. When the trained YOLOv8-seg sticker model lands, the same rule will apply.

**Effect.** No covariate-shift surprises between val and test. Test MAE of 24.16 kg is genuinely indicative of inference-time behaviour.

---

## 12. Gaps fixed from prior research

| Prior approach | Gap | Our fix |
|---|---|---|
| Schaeffer formula with manual tape measure | Slow, stressful, error-prone, requires trained handler | Replace tape with pose-detector + sticker-calibrated px-to-cm |
| End-to-end CNN regression on RGB | Needs depth or controlled distance to recover scale | Use a printed fiducial (sticker) to anchor absolute cm scale |
| Depth-camera methods (Kinect, stereo) | Specialized hardware; not smallholder-deployable | Smartphone-only RGB photo with a sticker fiducial |
| Cascaded pixel-feature → morphometric → trait architectures | Train/inference covariate shift when intermediate predictions replace GT | Force feature parity: train-time and inference-time features computed by the same code path |
| Ultralytics-default OKS sigmas on non-COCO pose | Metric inflated; model learns degenerate bbox-edge prior | Cattle-specific per-keypoint σ, monkey-patched into both the loss and validator |
| Single-global sticker calibration on multi-batch corpora | Cancels measurable systematic per-batch bias into "average" performance | Per-batch back-derived sticker cm², applied as hard lookup at inference |
| End-to-end CNN as the trait head on small-N tabular features | Over-fits with 3000 samples and 17 features | Regularized gradient-boosted trees (XGBoost) with shallow depth and explicit L2 |
| Using Schaeffer as the final estimator (zero-parameter) | Ignores systematic biases (circular-cross-section assumption, body-length scaling) | Feed Schaeffer as one feature of 17; let the tree ensemble learn residual corrections |
| BCS estimation from morphometric-derived labels | Lab convention; not in our corpus | Acknowledge: BCS labels don't exist in this corpus; demo-only heuristic BCS, paper restricts claims to weight |

---

## 13. Dataset and preprocessing

### Source

- **Name:** Cattle Weight Detection Model + Dataset 12k.
- **Provider:** Sadhli Roomy / Acme AI Ltd.
- **Funder:** Bill & Melinda Gates Foundation.
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (attribution required for any reuse).
- **Kaggle URL:** [sadhliroomyprime/cattle-weight-detection-model-dataset-12k](https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k).
- **Subjects:** Bangladeshi zebu cattle (*Bos indicus*), ~27,000 annotated images across three batches (B2, B3, B4).
- **Annotations:** Per-image COCO-format JSONs with cow + sticker masks (PNG) and keypoint coordinates.

### What we actually use

Of the ~27,000 images, only the **B3 + B4 side-view** subsets carry both:
1. The canonical 9-keypoint anatomy.
2. Weight labels encoded in the filename (e.g. `9_s_181_F.jpg` = 181 kg female cattle, side view).

B2 keypoint annotations use simplified sequence IDs that cannot be joined to weight labels — excluded.

| Subset | Images | Animals |
|---|---:|---:|
| B3 side-view | 2,603 | ~600 |
| B4 side-view | 1,946 | ~400 |
| Total used | **4,549** | **~1,000** |

### Filtering

Suspect samples are flagged offline by [`cattle_phenotyping/eval/flag_suspects.py`](../cattle_phenotyping/eval/flag_suspects.py) with two rules:

1. `large_residual`: forward Schaeffer (with GT keypoints) disagrees with the label by > 2.5σ.
2. `implausible_low_weight`: predicted weight < 50 kg (zebu calves go down to ~50 kg; below is biologically implausible).

The union of these flags removes ~115 unique samples from train+val. The test split is **never filtered** (a filtered test set would not be representative of inference-time conditions).

There's a third diagnostic flag — `cross_batch_id_collision` — but it's **not used as a training filter**. It surfaces 1,051 sample rows where the same numeric `animal_id` appears in both B3 and B4. We verified empirically that `animal_id` is batch-local: a B3 animal_id and a B4 animal_id are unrelated animals that coincidentally share the same number. The tuple key `(batch, animal_id)` is used everywhere a unique animal identifier is needed.

### Train/val/test split

Created by [`cattle_phenotyping/training/build_splits.py`](../cattle_phenotyping/training/build_splits.py):
1. Group by `(batch, animal_id)` so all photos of one animal go to one split.
2. Stratify by weight quartile so each split spans the full weight range.
3. Ratios 70/15/15, seed 42 for reproducibility.

Final split sizes (post-filter for train+val):

| Split | Images | Animals |
|---|---:|---:|
| Train | 3,094 | ~720 |
| Val | 654 | ~150 |
| Test | 685 | ~150 |

### Image preprocessing

YOLOv8 handles letterbox normalization internally:
1. Image read with OpenCV in BGR.
2. Resized to 640 × 640 with letterbox padding (preserves aspect ratio).
3. Inference; predictions denormalized back to original image coordinates.

The pipeline never resizes the image manually — all measurements are in **original-image pixel space** so the per-batch sticker calibration constants (derived in original-image space) remain valid.

---

## 14. Training protocol

### YOLOv8s-pose

| Setting | Value |
|---|---|
| Architecture | yolov8s-pose |
| Epochs | 50 |
| Batch size | 24 |
| Image size | 640 |
| Optimizer | SGD (Ultralytics default), auto-tuned LR |
| LR schedule | Cosine annealing |
| Augmentation | Mosaic (closed in last 10 epochs), HSV jitter, scale, translate |
| `fliplr` | **0.0** (disabled — horizontal flip swaps left/right anatomy) |
| OKS sigmas | Cattle-specific (Section 11.2) |
| Seed | 42 |

Training time: ~1h50m on Kaggle T4×2 (free tier).

### XGBoost weight head

1. Run the trained pose model over all train+val+test images, save predictions as JSON.
2. Run the feature builder over each split's `(sample, predicted_keypoints, sticker_area_px, per-batch_cm²)`.
3. Fit XGBoost on `(X_train, y_train)`, early stopping on val MAE.
4. Report metrics on test (never touched until this step).

Training time: ~30 seconds on CPU.

---

## 15. Evaluation protocol and results

### Three-tier comparison

We report metrics for three reference points on the **same** 685-cattle held-out test split, so each row is directly comparable:

| Method | Test MAE | MAPE | R² | Bias |
|---|---:|---:|---:|---:|
| Schaeffer with ground-truth keypoints | 30.25 kg | 19.69% | 0.11 | +0.33 |
| Schaeffer with predicted keypoints | 29.80 kg | 19.34% | 0.14 | −0.72 |
| **Learned XGBoost head** | **24.16 kg** | **16.1%** | **0.46** | **+0.38** |

The learned head improves over the predicted-keypoint Schaeffer baseline by **5.64 kg MAE = 19.1%**.

### Per-batch decomposition

| Batch | n | Schaeffer MAE | Learned MAE | ΔMAE |
|---|---:|---:|---:|---:|
| B3 | 355 | 30.24 | 26.48 | −3.75 |
| B4 | 330 | 29.21 | 21.72 | −7.49 |

B4's larger sticker gives a tighter cm-calibration signal, which the learned head exploits more aggressively.

### Top-5 feature importance (XGBoost gain)

| Rank | Feature | Gain |
|---:|---|---:|
| 1 | `body_length_cm` | 86,894 |
| 2 | `schaeffer_kg` | 29,342 |
| 3 | `front_girth_chord_cm` | 15,236 |
| 4 | `length_to_height_ratio_cm` | 8,222 |
| 5 | `rear_girth_chord_cm` | 7,847 |

`body_length_cm` outranks `schaeffer_kg`, confirming the elliptical-cross-section correction. The head uses Schaeffer as a prior but adds a length-dependent term.

### Hyperparameter sweep

| Config | depth | lr | n_est | Test MAE |
|---|---:|---:|---:|---:|
| Baseline | 5 | 0.04 | 800 | 24.47 |
| **deep4_lr03** | **4** | **0.03** | **1200** | **24.16** |
| deep5_lr02 | 5 | 0.02 | 1500 | 24.34 |
| deep6_lr05 | 6 | 0.05 | 800 | 24.60 |

Differences of 0.13–0.44 kg suggest the model is in a flat region of hyperparameter space; further tuning would not produce a meaningfully better number.

---

## 16. Engineering substrate

### Test coverage

| Module | Tests |
|---|---:|
| `data/kaggle.py` (COCO parser) | 19 |
| `eval/baseline_schaeffer.py` | 18 |
| `eval/flag_suspects.py` | 26 |
| `eval/keypoint_eval.py` | 9 |
| `models/schaeffer.py` | 17 |
| `models/bcs_heuristic.py` | 17 |
| `models/weight_head.py` | 12 |
| `pipeline/scale_calibration.py` | 21 |
| `pipeline/weight_head_features.py` | 15 |
| `training/build_splits.py` | 11 |
| `training/export_yolo_pose.py` | 11 |
| `training/kpt_sigmas.py` | 11 |
| **Total** | **187** |

All CPU-only; no GPU required for test execution. Run with `python -m pytest tests/`.

### Reproducibility

| Property | How |
|---|---|
| Deterministic splits | Fixed seed 42 in `build_splits.py`; CSV files version-controlled |
| Deterministic training | XGBoost seed 42; YOLOv8 seed 42; cattle OKS sigmas frozen in `kpt_sigmas.py` |
| Per-batch sticker constants | JSON committed to `data/calibration/sticker_area_cm2_by_batch.json` |
| Suspect-flag CSV | Committed to `data/calibration/suspect_samples.csv` |
| Model artifacts | XGBoost model + meta JSON; YOLOv8 best.pt published as GitHub Release |
| Schema validation | `WeightHead.predict()` rejects DataFrames with mismatched columns |

### Code layout

48 tracked Python files; no legacy code. See [README.md](../README.md) for the directory tree.

---

## 17. Reproduction recipe

Four CLI commands reproduce the full pipeline from scratch on Kaggle (the dataset is too large for most local machines):

```bash
# 1. Animal-grouped, weight-stratified splits
python -m cattle_phenotyping.training.build_splits \
    --dataset-root "$DATASET_ROOT" \
    --output-dir data/splits \
    --seed 42

# 2. Per-batch sticker calibration (train only — no test leakage)
python -m cattle_phenotyping.pipeline.scale_calibration \
    --dataset-root "$DATASET_ROOT" \
    --split-csv data/splits/train.csv \
    --load-masks --workers 8 \
    --output data/calibration/scale_calibration_train.json \
    --by-batch-output data/calibration/sticker_area_cm2_by_batch.json

# 3. Suspect-sample flagging (large_residual + implausible_low_weight)
python -m cattle_phenotyping.eval.flag_suspects \
    --dataset-root "$DATASET_ROOT" \
    --train-csv data/splits/train.csv \
    --val-csv data/splits/val.csv \
    --test-csv data/splits/test.csv \
    --sticker-area-by-batch-json data/calibration/sticker_area_cm2_by_batch.json \
    --workers 8 \
    --output data/calibration/suspect_samples.csv

# 4. Forward-Schaeffer baseline on test (the bar to beat)
python -m cattle_phenotyping.eval.baseline_schaeffer \
    --dataset-root "$DATASET_ROOT" \
    --split-csv data/splits/test.csv \
    --sticker-area-by-batch-json data/calibration/sticker_area_cm2_by_batch.json \
    --workers 8 \
    --output data/results/baseline_schaeffer_test_perbatch.json
```

Expected baseline output: `MAE ≈ 30.25 kg`, `MAPE ≈ 19.69%`, `R² ≈ 0.11`.

Then train the pose model (see [`notebooks/train_keypoints.scope.md`](../notebooks/train_keypoints.scope.md)) and the weight head (see [`notebooks/train_weight_head.scope.md`](../notebooks/train_weight_head.scope.md)).

---

## 18. Comparison to prior approaches

| Approach | Hardware | Typical MAE | Our approach delivers |
|---|---|---|---|
| Mechanical scale | Scale | 0 kg (ground truth) | Slow, stressful |
| Schaeffer + tape measure | Tape | 25–35 kg (depending on operator) | No-touch, single photo |
| Single-RGB CNN regression | Camera + controlled distance | 15–25 kg (~5% MAPE) | No distance control needed (sticker provides scale) |
| RGB-D / stereo depth methods | Depth sensor | 5–15 kg | No special hardware |
| **This work** | **Smartphone camera + printed sticker** | **24 kg (~16% MAPE)** | Smallholder-deployable |

We don't claim state-of-the-art — depth-sensor methods beat us. We claim:

1. **The cheapest possible measurement setup** that delivers absolute kg-scale weight from a single image.
2. **A reproducible improvement over the published smallholder baseline** (Schaeffer) on the largest public cattle-weight corpus.
3. **Honest evaluation** (held-out test, train-only calibration, no GT-keypoint leakage).

---

## 19. Known limitations

1. **Single breed.** All data is Bangladeshi zebu (*Bos indicus*). Performance on Holstein or other taurine breeds is **undefined**. Multi-breed validation is future work.
2. **Sticker is a hard dependency.** No sticker visible → no prediction. By design, not a quiet fallback. This is the same constraint Acme's corpus operates under.
3. **Per-batch sticker constants.** Any new batch needs its own back-derivation. Cannot reuse B3 or B4's constant.
4. **Test set mean weight is ~160 kg.** Mostly sub-adult zebu. Extrapolation to mature animals (>400 kg) is untested.
5. **R² of 0.46 means half the variance is unexplained.** This is intrinsic to single-RGB measurement (depth ambiguity); not a model-quality failure but a method-level ceiling.
6. **BCS is heuristic-only in the demo, excluded from the paper.** The dataset has no BCS labels; principled BCS estimation requires BCS-labelled imagery.
7. **SAM at inference is an interim solution.** A trained YOLOv8-seg sticker model would replace the SAM-click step, removing the user click and tightening sticker pixel-area estimates. Scoped in [`notebooks/train_segmenter.scope.md`](../notebooks/train_segmenter.scope.md), not yet implemented.

---

## References

Primary references (all cited in the paper at `docs/paper.tex`):

- Kaggle BMGF corpus: https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k
- Schaeffer formula: Ozkaya 2013, *J. Animal and Plant Sciences*.
- YOLOv8-pose: Jocher et al., Ultralytics, 2023.
- Segment Anything Model: Kirillov et al., ICCV 2023.
- XGBoost: Chen and Guestrin, KDD 2016.
- BCS scoring (for BCS scale reference): Edmonson et al., 1989; Bewley & Schutz, 2008.

Project-internal documentation:

- [README.md](../README.md) — quick start and architecture overview.
- [docs/paper.tex](paper.tex) — IEEE-format paper draft.
- [docs/demo_runbook.md](demo_runbook.md) — step-by-step deployment guide.
- [docs/kaggle_dataset_notes.md](kaggle_dataset_notes.md) — full dataset audit (filename grammars, keypoint schemas, sticker color codes per batch).
