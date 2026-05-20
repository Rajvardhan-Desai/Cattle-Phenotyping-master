# Kaggle dataset notes — `sadhliroomyprime/cattle-weight-detection-model-dataset-12k`

Findings from `notebooks/inspect_kaggle.ipynb` run on Kaggle, 2026-05-20. Source of truth for our annotation parser.

- **Provenance.** Inner folder name is `www.acmeai.tech Dataset - BMGF-LivestockWeight-CV` → produced by Acme AI, **funded by the Bill & Melinda Gates Foundation (BMGF)**. There is a `Readme.md` and a `www.acmeai.tech BMGF - LivestockWeight - CV.pdf` inside the dataset; check the PDF for the sticker physical dimensions (still unconfirmed — gating fact for scale calibration).
- **License.** CC BY 4.0. Attribution block required in our README: cite Sadhli Roomy / Acme AI Ltd. and acknowledge BMGF funding.
- **Dataset root path** on Kaggle: `/kaggle/input/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k/www.acmeai.tech Dataset - BMGF-LivestockWeight-CV/`. The leading `www.acmeai.tech ...` folder is part of the path — quote it everywhere.

## Annotation layout

- **`Vector/` holds keypoint annotations as COCO JSON.** One JSON per (batch, view, optional sub-split). 8 files total:

  | Batch | View | File | Images | Annotations |
  |---|---|---|---:|---:|
  | B2 | Side | `Vector/B2/Side/data/Side/COCO_Side.json` | 195 | 195 |
  | B2 | Side | `Vector/B2/Side/data/Side_2/COCO_Side_2.json` | 184 | 184 |
  | B2 | Rear | `Vector/B2/Rear/data/Rear/COCO_Rear.json` | 320 | 323 |
  | B2 | Rear | `Vector/B2/Rear/data/Rear_2/COCO_Rear_2.json` | 195 | 195 |
  | B3 | Side | `Vector/B3/Side/data/COCO_Side.json` | 2,603 | 2,604 |
  | B3 | Rear | `Vector/B3/Rear/data/COCO_B3_rear.json` | 2,601 | 2,619 |
  | B4 | Side | `Vector/B4/Side/data/coco_b4_side.json` | ? | ? |
  | B4 | Rear | `Vector/B4/Rear/data/coco_b4_rear.json` | 1,944 | 1,944 |

- **`Pixel/` holds segmentation as colored PNG masks** (not in the JSON inventory because the inspector only scanned text annotation extensions). Class colors documented per batch (B2 has Ground class, B3/B4 don't).

## Side-view keypoint schemas — they DIFFER per batch

**The canonical 9 anatomical keypoint names** (target schema for our parser):

```
wither, pinbone, shoulderbone,
front_girth_top, front_girth_bottom,
rear_girth_top,  rear_girth_bottom,
height_top,      height_bottom
```

### B3 Side (`category.name == "b3_side"`) — the gold reference (2,603 images)

The COCO `keypoints` array order is **NOT in name-number order**. Pairs of `_top`/`_bottom` keypoints are swapped (annotator listed `_bottom` first inside each pair):

| Array index | Keypoint name in JSON | Canonical name |
|---:|---|---|
| 0 | `1_wither` | `wither` |
| 1 | `2_pinbone` | `pinbone` |
| 2 | `3_shoulderbone` | `shoulderbone` |
| 3 | `5_front_girth_bottom` | `front_girth_bottom` |
| 4 | `4_front_girth_top` | `front_girth_top` |
| 5 | `9_Height_bottom` | `height_bottom` |
| 6 | `8_Height_top` | `height_top` |
| 7 | `7_rear_girth_bottom` | `rear_girth_bottom` |
| 8 | `6_rear_girth_top` | `rear_girth_top` |

**Parser must read the `categories[].keypoints` name array from each COCO file and remap by name, not by position.** The skeleton field is also non-standard and should be derived from canonical names, not copied.

### B2 Side / `Side/` (`category.name == "cattle-biometrics  NEW"`) — 6 keypoints, partial

Missing 3 of the canonical 9 (no shoulderbone, no height_top, no height_bottom):

```
["wither", "pinbone", "front-bottom", "rear-top", "rear-bottom", "front-top"]
```

The names use hyphens not underscores. Skeleton `[[1,2],[8,4],[5,6]]` references keypoint index 8 which doesn't exist in this 6-keypoint schema — skeleton is broken; ignore it.

Use this batch for **segmentation training but skip for the 9-keypoint head**.

### B2 Side / `Side_2/` — 23 anonymous numeric keypoints, two categories (`Cow side left`, `Cow Side Right`)

All keypoints are anonymous numeric IDs (`"1"`, `"3"`, `"4"`, ...). Without an anatomy mapping there's no way to use these. **Skip for keypoint training** until the anatomy mapping is found (likely in the PDF Readme).

### B4 Side (`category.id=2, name="cattle_side"`) — 1,945 images, **canonical order** (no swap)

| Array index | JSON name | Canonical name |
|---:|---|---|
| 0 | `1_wither` | `wither` |
| 1 | `2_pinbone` | `pinbone` |
| 2 | `3_shoulderbone` | `shoulderbone` |
| 3 | `4_front_girth_top` | `front_girth_top` |
| 4 | `5_front_girth_bottom` | `front_girth_bottom` |
| 5 | `6_rear_girth_top` | `rear_girth_top` |
| 6 | `7_rear_girth_bottom` | `rear_girth_bottom` |
| 7 | `8_Height_top` | `height_top` |
| 8 | `9_Height_bottom` | `height_bottom` |

Skeleton `[[1,2],[2,3],[4,5],[6,7],[8,9]]` — sensible anatomical edges. **Array order matches the name numbers**, unlike B3 Side. The name-based remap in the parser handles both layouts transparently.

## Rear-view keypoint schemas

**Canonical 4 names:** `top, bottom, left, right`.

| Source | Category | Array order |
|---|---|---|
| B2 Rear (both files) | `Cow back` | `["27","26","24","25"]` — anonymous; inferred from sample coords as **right, left, top, bottom** |
| B3 Rear | `b3_rear` (cat_id 1) | `["4_right","3_left","1_top","2_bottom"]` |
| B3 Rear | `Cattle Rear` (cat_id 2077) | `["Left","Right","Top","Bottom"]` — swap left/right vs cat 1! |
| B4 Rear | `cattle_rear` | `["1_top","2_bottom","3_left","4_right"]` |

Parser must dispatch on `(batch, category_id)`. The inference about B2 Rear's anonymous IDs being right/left/top/bottom comes from the sample annotation: kp0=(1098,692) high-x ⇒ right, kp1=(806,680) low-x ⇒ left, kp2=(970,378) low-y ⇒ top, kp3=(937,1274) high-y ⇒ bottom.

## Weight ground truth is encoded in filenames

The annotation JSONs contain bbox + keypoints + metadata but **no weight field**. Filename grammars confirmed by spot-check (50 random samples per batch):

| Batch | Example | Grammar | Has weight? | Animal count |
|---|---|---|---|---:|
| B2 (rich, real images) | `450_s_172_3.6_F.jpg`, `5.0_r_157_10.0_F.jpg` | `<animal_id>_<r|s>_<weight>_<EXTRA>_<M|F>.jpg` | **Yes** | 893 total |
| B2 (simplified, Vector COCO only) | `1_r.jpg` | `<seq_id>_<r|s>.jpg` | **No** | renamed seq IDs |
| B3 | `9_r_144_M.jpg`, `356_s_240_F.jpg` | `<animal_id>_<r|s>_<weight>_<M|F>.jpg` | **Yes** | 5,204 total |
| B4 | `100_b4-1_r_124_F.jpg`, `269_b4-3_s_172_F.jpg` | `<animal_id>_b4-<sub>_<r|s>_<weight>_<M|F>.jpg` | **Yes** | 5,833 total |

**Critical B2 caveat — two coexisting filename grammars:**
* The **rich grammar** is used by the actual image files on disk and by `Pixel/B2/*/annotations/*.png` masks. Animal IDs are sometimes integers (`450`), sometimes floats (`113.0`). The parser normalizes `113.0` → `"113"` so both render to the same `animal_key`. The 5th field is a decimal score whose meaning is **unconfirmed** (BCS, age, or condition score — possibly 1-10 scale given examples like `5.0_r_157_10.0_F.jpg`). It's captured in `FilenameMeta.extra` as a raw float for downstream interpretation once the PDF Readme clarifies.
* The **simplified grammar** is what `Vector/B2/.../COCO_*.json` `image.file_name` entries use — sequential IDs (`1_r.jpg`, `2_s.jpg`) that do **not** correspond to the rich animal IDs. This means **B2 Vector keypoint annotations cannot be joined to Pixel masks or to weight labels** without a separate mapping table. The parser accepts these via `FilenameMeta.is_b2_seq_only=True` and sets `animal_id = weight_kg = sex = None`.

**Implication for training:**
* B3 Side (2,603 imgs) + B4 Side (1,945 imgs) is the clean trainable set for weight regression. ≈4,500 weight-labelled side-view images, the largest 9-keypoint annotated cattle weight dataset I'm aware of in this scope.
* B2 keypoint annotations are effectively unusable for weight training; B2 masks are usable for segmentation pretraining if joined by filename (independent of the Vector COCO).
* `(batch, animal_id)` is the global animal key. The same animal appears in both `_r` and `_s` views with the same `animal_id` (verified: `100_b4-1_r_124_F` and `100_b4-1_s_124_F` are the same animal). Use this for paired multi-view training.

## Segmentation mask file naming

`Pixel/<batch>/[<Side|Rear>/]annotations/<image_filename>___fuse.png` (note the **triple underscore** before `fuse`):

| Batch | Mask root | Example mask filename |
|---|---|---|
| B2 | `Pixel/B2/Side/annotations/` (and `Rear/`) | `333.0_s_146_5.0_F.jpg___fuse.png` |
| B3 | `Pixel/B3/annotations/` (no Side/Rear split) | `264_s_197_F.jpg___fuse.png` |
| B4 | `Pixel/B4/Side/annotations/` (and `Rear/`) | `74_b4-4_s_129_F.jpg___fuse.png` |

For **B3 and B4**, the COCO `image.file_name` matches the leading portion of the mask filename, so joining is a clean suffix swap (`<file_name>___fuse.png`). The parser populates `KaggleSample.mask_path` in those cases.

**B2 masks are not joinable from Vector COCO** because the COCO entries use simplified seq IDs (`1_r.jpg`) while masks use the rich grammar (`333.0_s_146_5.0_F.jpg___fuse.png`). The parser returns `mask_path=None` for B2 — see test `test_mask_path_b2_is_none`.

## BCS labels

**Not present** in the dataset (keyword grep didn't surface any). BCS is dropped from the project deliverable per the scope pivot. Do not ship heuristic BCS predictions.

## Sticker physical dimensions — NOT in any public Acme AI document

Verified by pdfminer extraction of `www.acmeai.tech BMGF - LivestockWeight - CV.pdf` (2026-05-20). The PDF is a **funder brief**, not a methodology paper, and intentionally does not disclose the sticker's cm size. Key quotes:

> "Two images, one from the side and another from the rear to isolate hearth-girth, body length, height, and use a reference object (a **human or a cola bottle**) to understand depth/distance."
>
> "Can't expect intermediaries to carry Pepsi bottles. So we **developed stickers as a low-cost alternative which can be put on a cattle body** to make adjustments to depth deviations."
>
> "Stickers are an archaic form of innovation which we would like to remove in future development."

**Implications for our pipeline:**

* The sticker is applied to the cow's body (not held next to it), is custom-made by Acme, and the cm dimension is internal.
* Segmentation color codes in the Readme: B2 yellow `255,240,0`, B3/B4 blue `0,117,255`. These identify the sticker mask but don't tell us its physical size.

**Three paths to scale calibration:**

1. **Ask Acme** — `info@acmeai.tech`. Could be days or weeks. Free.
2. **Back-derive from labels (recommended).** The PDF discloses the Schaeffer weight formula: `weight_lb = (HG_in × HG_in × BL_in) / 300`. With keypoint-derived `HG_px` (heart girth chord from `front_girth_top↔bottom` or `rear_girth_top↔bottom`) and `BL_px` (`shoulderbone↔pinbone`), plus filename-encoded `weight_kg`:
   ```
   weight_lb = weight_kg * 2.2046
   weight_lb = (HG_cm² × BL_cm) / (2.54³ × 300)   # convert in→cm
   ⇒ HG_cm² × BL_cm = weight_lb × 2.54³ × 300
   ```
   With one unknown px-per-cm shared between HG and BL, every labelled image gives one equation. Solve the system → recover px-per-cm per image, then compare to that image's sticker pixel area. The sticker should consistently subtend a single physical cm² across the dataset; that constant **is** the sticker's real size.
3. **Skip absolute scale.** Train an end-to-end weight regressor on raw image + pixel keypoints + sticker pixel area; let the network learn depth implicitly. Loses the cm-morphometric deliverable.

Path 2 is the cleanest because it (a) recovers the sticker size as a side effect, (b) self-validates keypoint annotations (Schaeffer ↔ label disagreement flags bad rows), and (c) gives us a publishable Schaeffer baseline the ML model has to beat. Implement in `cattle_phenotyping/pipeline/scale_calibration.py`.

## B2 5th decimal field — almost certainly age in years

The PDF flags age as a known per-animal attribute they wanted in future datasets:

> "Future dataset needs diverse range of classes inclusive of breed, **age, teeth, species-based landmarks** etc. for better accuracy."

Combined with the value distribution we see in B2 filenames (`1.6, 2.0, 2.6, 2.8, 3.0, 3.2, 3.6, 4.0, 4.2, 4.5, 5.0, 5.5, 5.6, 6.0, 10.0`), the field is almost certainly **age in years**. Decimals like `2.6` rule out BCS (1-5 integer-ish) and teeth count (0-8 integer); range 1.6-10 fits cattle lifespan; the PDF explicitly names age as a covariate they track.

Parser still stores it as `FilenameMeta.extra` (opaque) until we verify with a second sample, but treat it as `age_years` for any downstream code that wants the field. **It is not a learnable target for our weight head** — it's a covariate / future feature.

## Resolved questions (post-PDF inspection)

- ✓ **B4 Side schema** — canonical 9-keypoint anatomy, name-number-correct array order.
- ✓ **Filename grammars** — confirmed for B3 and B4 across 50-sample spot-checks. **B2 has two grammars** (rich + simplified); parser handles both.
- ✓ **Mask filename convention** — `<image_filename>___fuse.png`. Joins cleanly for B3 and B4; B2 cannot join Vector→Pixel.
- ✓ **Animal IDs are batch-local** — `(batch, animal_id)` is the global animal key. Same animal_id across `_r` and `_s` views.
- ✓ **B2 5th field** — age in years (high-confidence inference from PDF + value distribution).
- ✓ **Sticker dimensions methodology** — Acme made custom stickers, applied to the cow's body; cm size not publicly documented. Recover via Schaeffer back-derivation (see Sticker physical dimensions section).
- ✓ **Acme's weight model formula** — Schaeffer: `weight_lb = (HG_in × HG_in × BL_in) / 300`. This is the baseline our ML head has to beat.

## Open questions (low priority, do not block Phase 3)

1. **B2 Side anonymous 23-keypoint mapping** in `Side_2/`. Likely in a figure in the PDF (which is text-only extracted) — would need to view the PDF visually. Low priority: we train on B3+B4 side (~4,548 weight-labelled images with the canonical 9-keypoint schema).
2. **B2 Vector ↔ Pixel filename mapping table.** Doesn't appear to exist in the dataset based on our inspection. If one is later discovered, B2 keypoint training (currently impossible) becomes viable. Low priority.
3. **Exact sticker cm size from Acme.** Email `info@acmeai.tech` if a quick reply is preferred over Schaeffer back-derivation. Optional — the back-derivation gives us the value as a side effect of validating the labels.
