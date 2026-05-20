"""
Predict cattle weight from keypoint and sticker-calibrated morphometric features.

This loader is for the Kaggle-trained Random Forest / tree-model bundle produced
from the Cattle Weight Detection Model + Dataset 12k notebook. It intentionally
does not extract keypoints or sticker masks from raw images; those must be
provided by a keypoint/segmentation model or by dataset annotations.
"""

import os
import pickle
from typing import Mapping

import numpy as np


DEFAULT_MODEL_PATH = os.path.join(
    "saved_models",
    "kaggle_weight",
    "final_cattle_weight_model.pkl",
)


class KeypointScaleWeightPredictor:
    """Load and run the Kaggle-trained keypoint-scale weight model."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.bundle = None
        self.model = None
        self.feature_cols: list[str] = []
        self.model_name = None
        self.load_error = None

        if os.path.exists(model_path):
            try:
                self._load(model_path)
            except Exception as exc:
                self.load_error = str(exc)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _load(self, model_path: str):
        with open(model_path, "rb") as f:
            self.bundle = pickle.load(f)

        self.model = self.bundle["model"]
        self.feature_cols = list(self.bundle["feature_cols"])
        self.model_name = self.bundle.get("model_name", self.model.__class__.__name__)

    def missing_features(self, features: Mapping[str, float]) -> list[str]:
        """Return feature names required by the model but absent from input."""
        return [name for name in self.feature_cols if name not in features]

    def predict(self, features: Mapping[str, float]) -> float:
        """Predict weight in kg from a complete feature mapping."""
        if not self.is_loaded:
            raise FileNotFoundError(
                f"Kaggle keypoint-scale model not found at {self.model_path}"
            )

        missing = self.missing_features(features)
        if missing:
            raise ValueError(f"Missing required model features: {missing}")

        vector = np.array(
            [[float(features[name]) for name in self.feature_cols]],
            dtype=float,
        )
        return float(self.model.predict(vector)[0])
