"""Legacy XGBoost trainer for the Horqin / Lili Bai Mendeley dataset.

.. deprecated:: 0.2.0
    Project scope changed on 2026-05-20 to use the Kaggle Bangladeshi zebu
    dataset only. This trainer targets the now-deprecated Horqin labels
    (``data/labels.csv``) and the orphan ``saved_models/{weight,bcs}_model.json``
    artifacts. It is retained for historical reference; new training lives in
    the Kaggle-native scripts under ``cattle_phenotyping/training/`` (Phase 3).

Usage (deprecated)::

    python -m cattle_phenotyping.training.train_trait_model \\
        --data_dir data/images --labels data/labels.csv
"""

import argparse
import json
import os
import sys
import warnings

import cv2

warnings.warn(
    "train_trait_model targets the deprecated Horqin/Mendeley dataset. "
    "Use the Kaggle-native training scripts instead.",
    DeprecationWarning,
    stacklevel=2,
)
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, r2_score

from cattle_phenotyping.models.detector_yolov8 import CowDetector
from cattle_phenotyping.models.segmenter_sam import CowSegmenter
from cattle_phenotyping.models.trait_model_xgboost import TraitPredictor
from cattle_phenotyping.pipeline.feature_extractor import FeatureExtractor
from cattle_phenotyping.utils.log import get_logger, setup_logging
from cattle_phenotyping.utils.seed import seed_everything

log = get_logger(__name__)


TARGET_WIDTH = 1024


def resize_image_keep_aspect(image, target_width=TARGET_WIDTH):
    """
    Resize image to a fixed width while keeping aspect ratio.
    Ensures consistent scale between training and inference.
    """
    h, w = image.shape[:2]

    if w == target_width:
        return image

    scale = target_width / w
    new_h = int(h * scale)

    resized = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)

    return resized


def extract_features_from_images(
    data_dir: str,
    labels_df: pd.DataFrame,
    detector: CowDetector,
    segmenter: CowSegmenter,
    extractor: FeatureExtractor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Extract features for every labelled image."""

    feature_names = [
        "body_area_px",
        "body_length_px",
        "body_height_px",
        "bbox_width",
        "bbox_height",
        "contour_perimeter",
        "convex_hull_area",
        "aspect_ratio",
        "body_area_ratio",
        "solidity",
        "compactness",
    ]

    morphometric_features = [
        "body_length_cm",
        "withers_height_cm",
        "heart_girth_cm",
        "hip_length_cm",
    ]

    X_list, y_weight, y_bcs, y_morph, groups, image_names = [], [], [], [], [], []

    for _, row in labels_df.iterrows():
        img_path = os.path.join(data_dir, row["image_name"])

        if not os.path.exists(img_path):
            log.warning("Skipping %s (file not found)", row["image_name"])
            continue

        image = cv2.imread(img_path)

        if image is None:
            log.warning("Skipping %s (unreadable)", row["image_name"])
            continue

        image = resize_image_keep_aspect(image)

        detection = detector.detect(image)

        if not detection["cow_detected"]:
            log.warning("Skipping %s (no cow detected)", row["image_name"])
            continue

        mask = segmenter.segment(image, detection["bbox"])

        features = extractor.extract(mask)

        vec = [features[name] for name in feature_names]

        X_list.append(vec)
        y_weight.append(row["weight"])
        y_bcs.append(row["bcs"])
        y_morph.append([row[morph_feat] for morph_feat in morphometric_features])
        groups.append(row["group_id"])
        image_names.append(row["image_name"])

        log.info("Processed %s", row["image_name"])

    return (
        np.array(X_list, dtype=float),
        np.array(y_weight, dtype=float),
        np.array(y_bcs, dtype=float),
        np.array(y_morph, dtype=float),
        groups,
        image_names,
    )


def prepare_labels(labels_df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
    """Validate labels and add a conservative grouping column for CV."""

    required_columns = {
        "image_name",
        "weight",
        "body_length_cm",
        "withers_height_cm",
        "heart_girth_cm",
        "hip_length_cm",
        "bcs",
    }
    missing_columns = sorted(required_columns - set(labels_df.columns))
    if missing_columns:
        raise ValueError(f"Labels file is missing columns: {missing_columns}")

    labels_df = labels_df.copy()

    images_on_disk = {
        name
        for name in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, name))
    } if os.path.isdir(data_dir) else set()

    missing_images = sorted(set(labels_df["image_name"]) - images_on_disk)
    extra_images = sorted(images_on_disk - set(labels_df["image_name"]))

    if missing_images:
        log.warning("%d labelled image(s) missing from disk: %s", len(missing_images), missing_images)
    if extra_images:
        log.warning("%d image(s) have no labels: %s", len(extra_images), extra_images)

    if "animal_id" in labels_df.columns:
        labels_df["group_id"] = labels_df["animal_id"].astype(str)
        log.info("Using animal_id for group-aware cross-validation.")
    else:
        duplicate_cols = [
            "weight",
            "body_length_cm",
            "withers_height_cm",
            "heart_girth_cm",
            "hip_length_cm",
        ]
        duplicate_mask = labels_df.duplicated(duplicate_cols, keep=False)
        labels_df["group_id"] = labels_df[duplicate_cols].astype(str).agg("|".join, axis=1)

        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count:
            log.warning(
                "animal_id column not found. Using identical measurement rows as "
                "provisional groups (%d rows affected).",
                duplicate_count,
            )
        else:
            labels_df["group_id"] = labels_df["image_name"].astype(str)

    return labels_df


def build_feature_matrix(
    X_pixel: np.ndarray,
    y_morph: np.ndarray | None,
    morph_models: list[xgb.XGBRegressor] | None = None,
) -> tuple[np.ndarray, list[xgb.XGBRegressor] | None]:
    """
    Build deployed feature matrix.

    During training, morphometric measurements are first learned from image-derived
    pixel features, then those predictions are passed to the trait regressors. This
    mirrors inference and avoids validating with manual measurements unavailable
    during deployment.
    """

    if morph_models is None:
        if y_morph is None:
            raise ValueError("y_morph is required when fitting morphometric models.")

        morph_models = []
        morph_predictions = []

        for idx in range(y_morph.shape[1]):
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
            )
            model.fit(X_pixel, y_morph[:, idx])
            morph_models.append(model)
            morph_predictions.append(model.predict(X_pixel))

        predicted_morph = np.vstack(morph_predictions).T
    else:
        predicted_morph = np.vstack([model.predict(X_pixel) for model in morph_models]).T

    return np.hstack([X_pixel, predicted_morph]), morph_models


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return publication-friendly regression metrics."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100)

    return {
        "mae": float(mae),
        "rmse": rmse,
        "mape": mape,
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost trait models")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/images",
        help="Directory containing cow images",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="data/labels.csv",
        help="CSV file with columns: image_name, weight, bcs",
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models",
        help="Directory to save trained models",
    )

    args = parser.parse_args()
    setup_logging()
    seed_everything()

    if not os.path.exists(args.labels):
        log.error("Labels file not found: %s", args.labels)
        sys.exit(1)

    labels_df = pd.read_csv(args.labels)
    labels_df = prepare_labels(labels_df, args.data_dir)

    log.info("Loaded %d labelled samples from %s", len(labels_df), args.labels)

    detector = CowDetector()
    segmenter = CowSegmenter()
    extractor = FeatureExtractor()

    log.info("Extracting features from images...")

    X_pixel, y_weight, y_bcs, y_morph, groups, image_names = extract_features_from_images(
        args.data_dir,
        labels_df,
        detector,
        segmenter,
        extractor,
    )

    if len(X_pixel) == 0:
        log.error("No valid samples found. Cannot train models.")
        sys.exit(1)

    log.info("Training on %d samples...", len(X_pixel))

    os.makedirs(args.model_dir, exist_ok=True)

    feature_cache_path = os.path.join(args.model_dir, "extracted_features.csv")
    pixel_feature_names = [
        "body_area_px",
        "body_length_px",
        "body_height_px",
        "bbox_width",
        "bbox_height",
        "contour_perimeter",
        "convex_hull_area",
        "aspect_ratio",
        "body_area_ratio",
        "solidity",
        "compactness",
    ]
    morphometric_feature_names = [
        "body_length_cm",
        "withers_height_cm",
        "heart_girth_cm",
        "hip_length_cm",
    ]
    feature_cache = pd.DataFrame(X_pixel, columns=pixel_feature_names)
    feature_cache.insert(0, "image_name", image_names)
    for idx, feature_name in enumerate(morphometric_feature_names):
        feature_cache[feature_name] = y_morph[:, idx]
    feature_cache["weight"] = y_weight
    feature_cache["bcs"] = y_bcs
    feature_cache["group_id"] = groups
    feature_cache.to_csv(feature_cache_path, index=False)
    log.info("Extracted feature table saved to %s", feature_cache_path)

    # =========================================================================
    # 5-FOLD CROSS-VALIDATION, MATCHING DEPLOYED INFERENCE CHAIN
    # =========================================================================
    unique_groups = sorted(set(groups))
    n_splits = min(5, len(unique_groups), len(X_pixel))
    if n_splits < 2:
        log.error("At least two valid groups/samples are required for cross-validation.")
        sys.exit(1)

    use_group_kfold = len(unique_groups) < len(groups)
    if use_group_kfold:
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X_pixel, y_weight, groups)
        cv_strategy = "GroupKFold"
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_iter = splitter.split(X_pixel)
        cv_strategy = "KFold"

    cv_results = {
        "weight_mae_scores": [],
        "weight_r2_scores": [],
        "bcs_mae_scores": [],
        "bcs_r2_scores": [],
        "folds": [],
        "n_samples": int(len(X_pixel)),
        "n_groups": int(len(unique_groups)),
        "cv_strategy": cv_strategy,
        "feature_protocol": "pixel_features_plus_predicted_morphometrics",
    }

    log.info("Starting %d-fold %s cross-validation", n_splits, cv_strategy.upper())

    fold_num = 0
    for train_idx, val_idx in split_iter:
        fold_num += 1
        X_train_pixel, X_val_pixel = X_pixel[train_idx], X_pixel[val_idx]
        y_weight_train, y_weight_val = y_weight[train_idx], y_weight[val_idx]
        y_bcs_train, y_bcs_val = y_bcs[train_idx], y_bcs[val_idx]
        y_morph_train = y_morph[train_idx]

        log.info(
            "Fold %d/%d: train=%d val=%d",
            fold_num, n_splits, len(X_train_pixel), len(X_val_pixel),
        )

        X_train, morph_models = build_feature_matrix(X_train_pixel, y_morph_train)
        X_val, _ = build_feature_matrix(X_val_pixel, None, morph_models)

        # Train weight model for this fold
        weight_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        weight_model.fit(X_train, y_weight_train)

        # Evaluate weight model
        y_weight_pred = weight_model.predict(X_val)
        weight_metrics = regression_metrics(y_weight_val, y_weight_pred)

        cv_results["weight_mae_scores"].append(weight_metrics["mae"])
        cv_results["weight_r2_scores"].append(weight_metrics["r2"])

        # Train BCS model for this fold
        bcs_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        bcs_model.fit(X_train, y_bcs_train)

        # Evaluate BCS model
        y_bcs_pred = bcs_model.predict(X_val)
        bcs_metrics = regression_metrics(y_bcs_val, y_bcs_pred)

        cv_results["bcs_mae_scores"].append(bcs_metrics["mae"])
        cv_results["bcs_r2_scores"].append(bcs_metrics["r2"])

        fold_result = {
            "fold": fold_num,
            "train_samples": int(len(train_idx)),
            "validation_samples": int(len(val_idx)),
            "weight": weight_metrics,
            "bcs": bcs_metrics,
        }
        cv_results["folds"].append(fold_result)

        log.info(
            "  Weight - MAE: %.4f, RMSE: %.4f, R^2: %.4f",
            weight_metrics["mae"], weight_metrics["rmse"], weight_metrics["r2"],
        )
        log.info(
            "  BCS    - MAE: %.4f, RMSE: %.4f, R^2: %.4f",
            bcs_metrics["mae"], bcs_metrics["rmse"], bcs_metrics["r2"],
        )

    # =========================================================================
    # CROSS-VALIDATION SUMMARY
    # =========================================================================
    log.info("Cross-validation summary:")
    log.info("%-15s %-20s %-20s", "Metric", "Weight", "BCS")

    weight_mae_mean = np.mean(cv_results["weight_mae_scores"])
    weight_mae_std = np.std(cv_results["weight_mae_scores"])
    weight_r2_mean = np.mean(cv_results["weight_r2_scores"])
    weight_r2_std = np.std(cv_results["weight_r2_scores"])

    bcs_mae_mean = np.mean(cv_results["bcs_mae_scores"])
    bcs_mae_std = np.std(cv_results["bcs_mae_scores"])
    bcs_r2_mean = np.mean(cv_results["bcs_r2_scores"])
    bcs_r2_std = np.std(cv_results["bcs_r2_scores"])

    log.info(
        "%-15s %.4f +- %.4f    %.4f +- %.4f",
        "MAE", weight_mae_mean, weight_mae_std, bcs_mae_mean, bcs_mae_std,
    )
    log.info(
        "%-15s %.4f +- %.4f    %.4f +- %.4f",
        "R^2 Score", weight_r2_mean, weight_r2_std, bcs_r2_mean, bcs_r2_std,
    )

    # Store summary statistics
    cv_results["summary"] = {
        "weight_mae_mean": float(weight_mae_mean),
        "weight_mae_std": float(weight_mae_std),
        "weight_r2_mean": float(weight_r2_mean),
        "weight_r2_std": float(weight_r2_std),
        "bcs_mae_mean": float(bcs_mae_mean),
        "bcs_mae_std": float(bcs_mae_std),
        "bcs_r2_mean": float(bcs_r2_mean),
        "bcs_r2_std": float(bcs_r2_std),
    }

    # =========================================================================
    # TRAIN FINAL MODEL ON FULL DATASET
    # =========================================================================
    log.info("Training final models on full dataset...")

    predictor = TraitPredictor(model_dir=args.model_dir)
    predictor.train(X_pixel, y_weight, y_bcs, y_morph)

    # =========================================================================
    # SAVE CV RESULTS
    # =========================================================================
    cv_results_path = os.path.join(args.model_dir, "cv_results.json")
    with open(cv_results_path, "w") as f:
        json.dump(cv_results, f, indent=2)

    log.info("CV results saved to %s", cv_results_path)
    log.info("Training complete.")


if __name__ == "__main__":
    main()
