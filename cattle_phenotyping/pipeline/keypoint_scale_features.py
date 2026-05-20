"""
Feature schema for the Kaggle keypoint + sticker-scale cattle weight model.

The current Streamlit app does not yet infer these features from arbitrary raw
images. Use this module once a keypoint detector and sticker/cattle segmentation
model are available, or when converting annotated dataset rows into features.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


FEATURE_COLUMNS = [
    "body_length_px",
    "wither_pin_px",
    "shoulder_wither_px",
    "front_girth_px",
    "rear_girth_px",
    "avg_girth_px",
    "height_px",
    "bbox_width_px",
    "bbox_height_px",
    "bbox_area_px",
    "bbox_aspect_ratio",
    "length_height_ratio",
    "front_rear_girth_ratio",
    "girth_height_ratio",
    "area_proxy",
    "girth_area_proxy",
    "volume_proxy",
    "bbox_volume_proxy",
    "sticker_area_px",
    "sticker_diameter_px",
    "cm_per_px",
    "cattle_area_px",
    "body_length_cm_est",
    "height_cm_est",
    "front_girth_cm_est",
    "rear_girth_cm_est",
    "avg_girth_cm_est",
    "cattle_area_cm2_est",
    "bbox_area_cm2_est",
    "volume_cm_proxy",
    "area_volume_cm_proxy",
]


def _distance(points: Mapping[str, tuple[float, float]], a: str, b: str) -> float:
    if a not in points or b not in points:
        return float("nan")
    ax, ay = points[a]
    bx, by = points[b]
    return math.hypot(ax - bx, ay - by)


def _ratio(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b) or b == 0:
        return float("nan")
    return a / b


def build_keypoint_scale_features(
    points: Mapping[str, tuple[float, float]],
    bbox: Mapping[str, float],
    mask_stats: Mapping[str, float],
) -> dict[str, float]:
    """
    Build the feature vector expected by final_cattle_weight_model.pkl.

    Args:
        points: Canonical side-view keypoints. Expected keys are wither,
            pinbone, shoulderbone, front_girth_top, front_girth_bottom,
            rear_girth_top, rear_girth_bottom, height_top, height_bottom.
        bbox: Detection box values with keys width, height, and area.
        mask_stats: Sticker/cattle segmentation stats with keys sticker_area_px,
            sticker_diameter_px, cm_per_px, and cattle_area_px.
    """

    body_length_px = _distance(points, "shoulderbone", "pinbone")
    if np.isnan(body_length_px):
        body_length_px = _distance(points, "wither", "pinbone")

    wither_pin_px = _distance(points, "wither", "pinbone")
    shoulder_wither_px = _distance(points, "shoulderbone", "wither")
    front_girth_px = _distance(points, "front_girth_top", "front_girth_bottom")
    rear_girth_px = _distance(points, "rear_girth_top", "rear_girth_bottom")
    avg_girth_px = float(np.nanmean([front_girth_px, rear_girth_px]))

    height_px = _distance(points, "height_top", "height_bottom")
    if np.isnan(height_px):
        height_px = float(bbox["height"])

    bbox_width_px = float(bbox["width"])
    bbox_height_px = float(bbox["height"])
    bbox_area_px = float(bbox.get("area", bbox_width_px * bbox_height_px))

    cm_per_px = float(mask_stats["cm_per_px"])
    cattle_area_px = float(mask_stats["cattle_area_px"])

    features = {
        "body_length_px": body_length_px,
        "wither_pin_px": wither_pin_px,
        "shoulder_wither_px": shoulder_wither_px,
        "front_girth_px": front_girth_px,
        "rear_girth_px": rear_girth_px,
        "avg_girth_px": avg_girth_px,
        "height_px": height_px,
        "bbox_width_px": bbox_width_px,
        "bbox_height_px": bbox_height_px,
        "bbox_area_px": bbox_area_px,
        "bbox_aspect_ratio": _ratio(bbox_width_px, bbox_height_px),
        "length_height_ratio": _ratio(body_length_px, height_px),
        "front_rear_girth_ratio": _ratio(front_girth_px, rear_girth_px),
        "girth_height_ratio": _ratio(avg_girth_px, height_px),
        "area_proxy": body_length_px * height_px,
        "girth_area_proxy": avg_girth_px**2,
        "volume_proxy": body_length_px * (avg_girth_px**2),
        "bbox_volume_proxy": bbox_width_px * bbox_height_px * avg_girth_px,
        "sticker_area_px": float(mask_stats["sticker_area_px"]),
        "sticker_diameter_px": float(mask_stats["sticker_diameter_px"]),
        "cm_per_px": cm_per_px,
        "cattle_area_px": cattle_area_px,
    }

    features["body_length_cm_est"] = body_length_px * cm_per_px
    features["height_cm_est"] = height_px * cm_per_px
    features["front_girth_cm_est"] = front_girth_px * cm_per_px
    features["rear_girth_cm_est"] = rear_girth_px * cm_per_px
    features["avg_girth_cm_est"] = avg_girth_px * cm_per_px
    features["cattle_area_cm2_est"] = cattle_area_px * (cm_per_px**2)
    features["bbox_area_cm2_est"] = bbox_area_px * (cm_per_px**2)
    features["volume_cm_proxy"] = (
        features["body_length_cm_est"] * (features["avg_girth_cm_est"] ** 2)
    )
    features["area_volume_cm_proxy"] = (
        features["body_length_cm_est"]
        * features["height_cm_est"]
        * features["avg_girth_cm_est"]
    )

    return {name: float(features[name]) for name in FEATURE_COLUMNS}
