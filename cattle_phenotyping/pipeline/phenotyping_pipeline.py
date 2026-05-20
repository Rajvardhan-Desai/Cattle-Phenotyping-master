"""Phenotyping pipeline.

Orchestrates detection → segmentation → feature extraction → trait prediction.
"""

from __future__ import annotations

import cv2
import numpy as np

from cattle_phenotyping.models.detector_yolov8 import CowDetector
from cattle_phenotyping.models.segmenter_sam import CowSegmenter
from cattle_phenotyping.models.trait_model_xgboost import TraitPredictor
from cattle_phenotyping.pipeline.feature_extractor import FeatureExtractor
from cattle_phenotyping.utils.config import Config, load_config
from cattle_phenotyping.utils.log import get_logger

log = get_logger(__name__)


def resize_image_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize an image to ``target_width`` keeping aspect ratio.

    Identical pre-processing on the training and inference paths is what
    keeps mask-derived features comparable between the two.
    """
    h, w = image.shape[:2]
    if w == target_width:
        return image

    scale = target_width / w
    new_h = int(h * scale)
    return cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)


class PhenotypingPipeline:
    """Detect cow → segment → extract features → predict traits."""

    def __init__(
        self,
        config_path: str | None = None,
        config: Config | None = None,
    ):
        self.config = config if config is not None else load_config(config_path)
        self.target_width = self.config.target_width

        det_cfg = self.config.detector
        seg_cfg = self.config.segmenter

        log.info("Loading cow detector (YOLOv8) from %s", det_cfg.get("model_path"))
        self.detector = CowDetector(
            model_path=det_cfg.get("model_path", "yolov8n.pt"),
            confidence=float(det_cfg.get("confidence", 0.3)),
        )

        log.info("Loading segmenter (%s) from %s", seg_cfg.get("model_type"), seg_cfg.get("checkpoint"))
        self.segmenter = CowSegmenter(
            checkpoint_path=seg_cfg.get("checkpoint", "sam_vit_b_01ec64.pth"),
            allow_auto_download=bool(seg_cfg.get("allow_auto_download", False)),
        )

        self.feature_extractor = FeatureExtractor()

        trait_model_dir = self.config.path("saved_models_dir")
        log.info("Loading trait predictor (XGBoost) from %s", trait_model_dir)
        self.predictor = TraitPredictor(model_dir=trait_model_dir)

        log.info("Pipeline ready.")

    def run(self, image: np.ndarray) -> dict:
        """Run the full pipeline on a single BGR image."""
        result: dict = {}

        image = resize_image_keep_aspect(image, self.target_width)
        result["processed_image"] = image

        # --- Stage 1: detection ------------------------------------------------
        detection = self.detector.detect(image)
        result["cow_detected"] = detection["cow_detected"]
        result["detection_confidence"] = detection["confidence"]
        result["bbox"] = detection["bbox"]

        if not detection["cow_detected"]:
            result["message"] = "No cow detected in the image."
            return result

        # --- Stage 2: segmentation --------------------------------------------
        mask = self.segmenter.segment(image, detection["bbox"])
        result["mask"] = mask

        # --- Stage 3: feature extraction --------------------------------------
        features = self.feature_extractor.extract(mask)
        result["features"] = features

        # --- Stage 4: trait prediction ----------------------------------------
        traits = self.predictor.predict(features)
        result["estimated_weight_kg"] = traits["estimated_weight_kg"]
        result["body_condition_score"] = traits["body_condition_score"]
        result["body_length_cm"] = traits.get("body_length_cm", 0.0)
        result["withers_height_cm"] = traits.get("withers_height_cm", 0.0)
        result["heart_girth_cm"] = traits.get("heart_girth_cm", 0.0)
        result["hip_length_cm"] = traits.get("hip_length_cm", 0.0)

        return result
