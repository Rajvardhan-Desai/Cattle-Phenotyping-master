# Livestock Phenotyping MVP

A research prototype that estimates cattle traits (body condition score and weight) from a single image using computer vision and machine learning.

## Pipeline Architecture

```
Image → Cow Detection (YOLOv8n) → Segmentation (SAM ViT-B) → Feature Extraction (OpenCV) → Trait Prediction (XGBoost) → Output
```

### Models Used

| Stage | Model | Purpose |
|---|---|---|
| Detection | YOLOv8n (Ultralytics, COCO pretrained) | Detect cow bounding box |
| Segmentation | Segment Anything Model (SAM ViT-B) | Generate binary cow mask |
| Trait Prediction | XGBoost Regressor | Predict weight (kg) and BCS |

## Project Structure

```
livestock-phenotyping-mvp/
├── models/
│   ├── detector_yolov8.py       # YOLOv8 cow detector
│   ├── segmenter_sam.py         # SAM segmenter
│   └── trait_model_xgboost.py   # XGBoost trait predictor
├── pipeline/
│   ├── feature_extractor.py     # Morphological feature extraction
│   └── phenotyping_pipeline.py  # Full pipeline orchestration
├── training/
│   └── train_trait_model.py     # Train XGBoost on labelled data
├── app/
│   └── streamlit_app.py         # Streamlit web interface
├── data/
│   ├── images/                  # Place cow images here
│   └── labels.csv               # Training labels
├── utils/
│   └── visualization.py         # Drawing / overlay utilities
├── main.py                      # CLI entry point
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SAM Checkpoint

The SAM ViT-B checkpoint (~375 MB) will be **downloaded automatically** on first run. To download manually:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Place the file in the project root directory.

### 3. YOLOv8 Weights

YOLOv8n weights are downloaded automatically by Ultralytics on first use.

## Usage

### CLI — Single Image

```bash
cd livestock-phenotyping-mvp
python main.py --image path/to/cow.jpg
```

Optional: save annotated output image:

```bash
python main.py --image path/to/cow.jpg --save_vis output.jpg
```

### Streamlit App

```bash
cd livestock-phenotyping-mvp
streamlit run app/streamlit_app.py
```

Then open the URL shown in your terminal (usually http://localhost:8501).

### Train XGBoost Models

1. Place labelled cow images in `data/images/`.
2. Fill in `data/labels.csv` with columns: `image_name`, `weight`, `bcs`.
3. Run training:

```bash
python training/train_trait_model.py --data_dir data/images --labels data/labels.csv
```

Trained models are saved to `saved_models/`.

The training script now evaluates the same deployed chain used by the app:

```
pixel morphology -> predicted morphometric measurements -> weight / BCS
```

This avoids validating with manual body measurements that are unavailable at
inference time. If `animal_id` is present in `data/labels.csv`, validation uses
group-aware folds so multiple images of the same animal do not leak between
training and validation. Without `animal_id`, rows with identical manual
measurements are grouped conservatively.

The script also writes:

- `saved_models/cv_results.json` for paper tables.
- `saved_models/extracted_features.csv` for reproducible feature analysis.

## Paper-Oriented Experimental Plan

For a publishable small-data study, report this as a feasibility/prototype
system rather than a fully generalized cattle weighing product.

Recommended experiments:

1. Deployed protocol: image-derived features plus predicted morphometrics.
2. Pixel-only ablation: image-derived features without predicted cm values.
3. Manual-measurement upper bound: image features plus true measured cm values.
4. Baselines: linear regression, random forest, SVR, and XGBoost.
5. Group-aware validation if repeated images come from the same animal.
6. Metrics: MAE, RMSE, MAPE, R2, and BCS accuracy within +/-0.5 score.
7. Error analysis by body weight range, BCS class, detection confidence, and
   segmentation quality.

## Kaggle Weight Model

The Kaggle-trained Random Forest model from the Cattle Weight Detection Model +
Dataset 12k notebook should be copied to:

```text
saved_models/kaggle_weight/final_cattle_weight_model.pkl
```

This model uses side-view keypoints, sticker-calibrated scale features, and
segmentation-derived cattle area. It is supported by:

- `models/keypoint_scale_weight_model.py`
- `pipeline/keypoint_scale_features.py`

The current Streamlit raw-image app does not directly use this model yet because
the app still needs inference models for cattle keypoints and sticker/cattle
segmentation. See `docs/kaggle_model_integration.md` for the integration path.

## Example Output

```json
{
  "cow_detected": true,
  "detection_confidence": 0.87,
  "bbox": [120, 80, 740, 450],
  "body_area_px": 152343,
  "body_length_px": 620,
  "body_height_px": 270,
  "bbox_width": 620,
  "bbox_height": 370,
  "contour_perimeter": 1842.5,
  "convex_hull_area": 165200,
  "aspect_ratio": 2.3,
  "estimated_weight_kg": 430.0,
  "body_condition_score": 3.1
}
```

## Notes

- The system works on **single side-view images** of cattle.
- Without trained XGBoost models, **heuristic estimates** are used for weight and BCS.
- For best accuracy, train the XGBoost models on your own labelled dataset.
- GPU is recommended for SAM inference but CPU works (slower).
- `data/labels.csv` is the active label file. `data/labels_legacy_all_bcs_3.csv`
  is retained only as a legacy snapshot and should not be used for paper results.
