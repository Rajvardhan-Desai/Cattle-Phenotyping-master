# Demo runbook (tomorrow's exam)

End-to-end checklist to get the Streamlit demo running on your laptop. Time budget: ~25 minutes from Kaggle → working demo if everything downloads cleanly.

## 1. Download the trained artifacts from Kaggle

In your Kaggle notebook's last cell, copy these to `/kaggle/working/demo_bundle/` and zip them:

```python
import shutil, zipfile, os
from pathlib import Path

bundle = Path("/kaggle/working/demo_bundle")
bundle.mkdir(exist_ok=True)
(bundle / "pose").mkdir(exist_ok=True)

# 1. Trained pose model
shutil.copy("/kaggle/working/runs/pose/kpt_001/weights/best.pt", bundle / "pose" / "best.pt")

# 2. Trained weight head (XGBoost dump + meta)
shutil.copy("/kaggle/working/data/results/weight_head.json", bundle / "weight_head.json")
shutil.copy("/kaggle/working/data/results/weight_head.meta.json", bundle / "weight_head.meta.json")

# 3. Per-batch sticker calibration JSON
shutil.copy("/kaggle/working/data/calibration/sticker_area_cm2_by_batch.json", bundle / "sticker_area_cm2_by_batch.json")

# 4. (Optional but recommended) the weight_head_report.json for the PPT
shutil.copy("/kaggle/working/data/results/weight_head_report.json", bundle / "weight_head_report.json")

# Zip it
zip_path = Path("/kaggle/working/demo_bundle.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for p in bundle.rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(bundle))
print(f"Zip ready: {zip_path}  ({zip_path.stat().st_size/1e6:.1f} MB)")
print(f"Download via: Kaggle right sidebar → Output → demo_bundle.zip")
```

Download `demo_bundle.zip` from the Kaggle Output panel. **Total size ≈ 92 MB** (pose weights dominate).

## 2. Local setup

In the repo root:

```bash
# Install the package + demo extras
pip install -e .
pip install streamlit-image-coordinates   # for click-to-segment UX (optional but recommended)

# Layout the artifacts
mkdir -p weights/pose data/calibration
unzip ~/Downloads/demo_bundle.zip -d /tmp/bundle
cp /tmp/bundle/pose/best.pt          weights/pose/best.pt
cp /tmp/bundle/weight_head.json      weights/weight_head.json
cp /tmp/bundle/weight_head.meta.json weights/weight_head.meta.json
cp /tmp/bundle/sticker_area_cm2_by_batch.json data/calibration/sticker_area_cm2_by_batch.json

# Download SAM ViT-B (375 MB) if you don't already have it
# Place in the repo root as sam_vit_b_01ec64.pth
curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## 3. Run

```bash
streamlit run cattle_phenotyping/app/demo_app.py
```

The browser tab opens at `http://localhost:8501`. Sidebar shows green ticks for all four loaded models.

## 4. Demo flow (rehearse once before going on stage)

1. **Upload** a side-view cattle photo. (Tip: have 2-3 photos pre-loaded on your desktop — one from each batch B3 and B4 for variety.)
2. **Wait ~2 seconds** for Stage 1 (pose). Image appears with the bbox + 9 keypoint dots colored by group: red=spine, green=girth, blue=height.
3. **Click the sticker** in the click image (left). The yellow cross marks your click; SAM segments the sticker in ~3 seconds and renders the mask in red on the right.
4. **Read the result** — three big cards across the bottom:
   - Predicted weight (kg) — what the learned head outputs.
   - Schaeffer baseline — the formula's prediction from the same keypoints.
   - Learned correction — the difference. Useful talking point: "the model is correcting Schaeffer by N kg using elliptical-cross-section information."
5. **Derived measurements** below (body length, girth chords, height in cm). These come straight from the keypoints + sticker scale.

### Batch selector

In the sidebar, "Batch" defaults to B4 (larger sticker, ~79 cm²). If your demo photo has a smaller sticker (the B3 type, ~15 cm²), switch to B3 first. Wrong batch → predicted weight will be off by a factor of ~3 in either direction.

## 5. Talking points for the audience

- **Dataset:** Kaggle BMGF cattle weight dataset, 4,549 side-view images of Bangladeshi zebu cattle, CC BY 4.0 (Acme AI / Bill & Melinda Gates Foundation).
- **Pipeline stages on screen:** detector + 9-keypoint pose head (trained YOLOv8s-pose) → sticker mask (SAM with one click; will be replaced by trained YOLOv8-seg in next phase) → cm-grounded measurements → XGBoost weight head.
- **Headline numbers:** Test MAE **24.47 kg** / MAPE **16.3%** / R² **0.46** on the held-out test split (685 cattle). Schaeffer baseline on the same split: 30.25 kg / 19.7% / 0.11. **~19% MAE reduction.**
- **What the learned head is correcting:** Schaeffer assumes a circular cross-section (multiplier = π). Real cattle are elliptical. The head's top features are `body_length_cm` (gain ~87k), `schaeffer_kg` (~29k), and `front_girth_chord_cm` (~15k) — it uses Schaeffer as a strong prior but adds a body-length-dependent correction.
- **Known limitations:** trained on one breed (Bos indicus zebu, Bangladesh). Requires a sticker in frame for scale. Not validated on other breeds.
- **License:** Trained weights inherit CC BY 4.0 from the dataset.

## 6. PPT / IEEE paper sources

- **Results table**: `data/results/weight_head_report.json` has the per-batch breakdown + feature importance ranking.
- **Per-keypoint pixel-error table**: `data/results/yolov8s_pose_val_eval.json` (from the re-trained keypoint run).
- **Baseline table**: `data/results/baseline_schaeffer_test_perbatch.json`.
- **Dataset audit** (filename grammars, mask color codes, keypoint schemas per batch): `docs/kaggle_dataset_notes.md`.
- **Architecture diagram seed** (paste into draw.io / PowerPoint):
  ```
  Image
    └─► YOLOv8s-pose (trained) ──► 9 anatomical keypoints
    └─► SAM ViT-B (point prompt) ─► sticker mask (→ pixel area)
                                          │
                                          ▼
                              per-batch sticker cm² ─► px-per-cm
                                          │
                                          ▼
                                   Feature builder (17 features)
                                          │
                                          ▼
                                   XGBoost weight head
                                          │
                                          ▼
                                   Predicted weight (kg)
  ```

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar reports "Pose: file not found" | Sidebar path doesn't match where you put best.pt. Edit it in the running app (changes take effect on next upload). |
| "SAM ViT-B not found" | Place `sam_vit_b_01ec64.pth` in the repo root (default sidebar path). Or edit the sidebar path. |
| "WeightHead model file missing" | Sidebar path's stem should NOT include `.json`. Default `weights/weight_head` looks for `weights/weight_head.json` + `.meta.json`. |
| Predicted weight is wildly off | Wrong batch selected in sidebar (B3 vs B4 sticker calibration). Switch and re-click sticker. |
| "Stage 2 — SAM: no mask under the size cap" | Your click missed the sticker. Click again more precisely. Or raise "SAM sticker cap" slider in the sidebar (default 5% of image area). |
| Pose detects nothing | Lower the "Pose confidence threshold" slider (sidebar). Default 0.25; try 0.10. |
| Streamlit shows a stale model after editing weights file | Click "Reload Models" button (if visible) or restart with `Ctrl+C` + `streamlit run ...` again. |
