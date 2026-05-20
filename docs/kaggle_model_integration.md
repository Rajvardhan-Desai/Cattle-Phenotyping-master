# Kaggle Weight Model Integration

This project now supports loading the model bundle produced by the Kaggle
keypoint + sticker-scale notebook.

## Put Kaggle Outputs Here

Copy these files from `/kaggle/working` into:

```text
saved_models/kaggle_weight/
```

Recommended files:

```text
final_cattle_weight_model.pkl
final_model_summary.json
final_model_comparison.csv
final_per_batch_metrics.csv
final_test_predictions.csv
final_training_features.csv
```

`saved_models/` is ignored by Git, so these artifacts stay local.

## What The Model Expects

The selected Random Forest does not accept a raw image. It expects engineered
features from:

1. Side-view cattle keypoints.
2. Sticker segmentation for pixel-to-cm scale.
3. Cattle segmentation area.
4. Detection bbox dimensions.

The expected feature schema is in:

```text
pipeline/keypoint_scale_features.py
```

The model loader is:

```text
models/keypoint_scale_weight_model.py
```

## Why The Streamlit App Does Not Use It Yet

The current Streamlit app accepts a raw image and runs:

```text
YOLO cow box -> SAM cow mask -> pixel morphology -> XGBoost trait model
```

Your Kaggle model was trained on ground-truth COCO keypoints and segmentation
masks. To use it on arbitrary uploads, the app needs two additional inference
models:

1. A side-view cattle keypoint detector, for example `YOLOv8-pose`.
2. A sticker/cattle segmentation model, for example `YOLOv8-seg`.

After those are trained, the app can run:

```text
raw image -> keypoints + sticker/cattle masks -> keypoint-scale features -> Random Forest weight
```

## Recommended Next Training

Train these on the same Kaggle dataset:

```text
YOLOv8-pose on Vector COCO keypoints
YOLOv8-seg on Pixel masks
```

Then add inference wrappers that output canonical keypoints, bbox dimensions,
sticker area, sticker diameter, cm-per-pixel, and cattle area. Those values can
be passed into `build_keypoint_scale_features`, then into
`KeypointScaleWeightPredictor.predict`.
