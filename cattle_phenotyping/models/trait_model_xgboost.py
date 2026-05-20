"""Legacy XGBoost trait predictor.

.. deprecated:: 0.2.0
    This predictor was trained against the Horqin/Mendeley dataset (now out of
    scope). It is kept so the Streamlit app and legacy CLI invocations keep
    working until the Phase 3 Kaggle-native predictor lands. New code should
    not import :class:`TraitPredictor`; use the upcoming
    ``cattle_phenotyping.models.weight_regressor`` instead.

The heuristic fallbacks in this module (``_heuristic_weight``, ``_heuristic_bcs``)
are particularly suspect on out-of-distribution images — see the plan's risk
callout about not shipping heuristic BCS predictions.
"""

import os
import numpy as np
import xgboost as xgb
import joblib

from cattle_phenotyping.utils.log import get_logger

log = get_logger(__name__)


class TraitPredictor:
    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = model_dir
        self.weight_model = None
        self.bcs_model = None
        self.pixel_feature_names = [
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
        self.morphometric_feature_names = [
            "body_length_cm",
            "withers_height_cm",
            "heart_girth_cm",
            "hip_length_cm",
        ]
        self.feature_names = self.pixel_feature_names + self.morphometric_feature_names
        self.morphometric_models = {
            "body_length_cm": None,
            "withers_height_cm": None,
            "heart_girth_cm": None,
            "hip_length_cm": None,
        }
        self._load_models()

    def _load_models(self):
        """Load saved models if they exist."""
        weight_path = os.path.join(self.model_dir, "weight_model.json")
        bcs_path = os.path.join(self.model_dir, "bcs_model.json")

        if os.path.exists(weight_path):
            self.weight_model = xgb.XGBRegressor()
            self.weight_model.load_model(weight_path)

        if os.path.exists(bcs_path):
            self.bcs_model = xgb.XGBRegressor()
            self.bcs_model.load_model(bcs_path)

        for feature_name in self.morphometric_feature_names:
            model_path = os.path.join(self.model_dir, f"{feature_name}_model.json")
            if os.path.exists(model_path):
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                self.morphometric_models[feature_name] = model

    def predict(self, features: dict) -> dict:
        """
        Predict weight and BCS from morphological features.

        Args:
            features: dict with morphological measurements.

        Returns:
            dict with estimated_weight_kg and body_condition_score.
        """
        pixel_vector = np.array(
            [[features.get(name, 0) for name in self.pixel_feature_names]]
        )

        morphometric_values = {}
        for feature_name in self.morphometric_feature_names:
            if feature_name in features:
                morphometric_values[feature_name] = float(features[feature_name])
                continue

            model = self.morphometric_models.get(feature_name)
            if model is not None:
                morphometric_values[feature_name] = float(model.predict(pixel_vector)[0])
            else:
                morphometric_values[feature_name] = 0.0

        feature_vector = np.array(
            [[features.get(name, 0) for name in self.pixel_feature_names]
             + [morphometric_values[name] for name in self.morphometric_feature_names]]
        )

        result = {}
        result.update(morphometric_values)

        if self.weight_model is not None:
            result["estimated_weight_kg"] = round(
                float(self.weight_model.predict(feature_vector)[0]), 1
            )
        else:
            # Fallback heuristic when no trained model is available
            result["estimated_weight_kg"] = self._heuristic_weight(features)

        if self.bcs_model is not None:
            raw_bcs = float(self.bcs_model.predict(feature_vector)[0])
            result["body_condition_score"] = round(max(1.0, min(5.0, raw_bcs)), 1)
        else:
            result["body_condition_score"] = self._heuristic_bcs(features)

        return result

    @staticmethod
    def _heuristic_weight(features: dict) -> float:
        """Simple heuristic weight estimate based on pixel area."""
        area = features.get("body_area_px", 0)
        if area == 0:
            return 0.0
        # Rough linear mapping: ~3 px² per kg (tuned for typical side-view photos)
        estimated = area * 0.003 + 50
        return round(max(100.0, min(900.0, estimated)), 1)

    @staticmethod
    def _heuristic_bcs(features: dict) -> float:
        """Simple heuristic BCS estimate based on aspect ratio and area fill."""
        aspect = features.get("aspect_ratio", 1.0)
        area = features.get("body_area_px", 0)
        hull = features.get("convex_hull_area", 1)
        fill_ratio = area / hull if hull > 0 else 0.5

        # Wider and more filled => higher BCS
        bcs = 2.0 + fill_ratio * 2.0 + (1.0 / max(aspect, 0.5)) * 0.5
        return round(max(1.0, min(5.0, bcs)), 1)

    def train(
        self,
        X: np.ndarray,
        y_weight: np.ndarray,
        y_bcs: np.ndarray,
        y_morph: np.ndarray | None = None,
    ):
        """
        Train weight and BCS models.

        Args:
            X: Pixel feature matrix, or legacy full feature matrix.
            y_weight: Weight targets.
            y_bcs: BCS targets.
            y_morph: Manual morphometric targets. When provided, models learn
                morphometrics from pixel features and train trait models on
                predicted morphometrics to match deployment.
        """
        os.makedirs(self.model_dir, exist_ok=True)

        n_pixel_features = len(self.pixel_feature_names)

        if y_morph is None:
            if X.shape[1] <= n_pixel_features:
                raise ValueError(
                    "y_morph is required when training from pixel features only."
                )
            X_pixel = X[:, :n_pixel_features]
            y_morph = X[:, n_pixel_features:]
        else:
            X_pixel = X[:, :n_pixel_features]

        predicted_morph = []
        for idx, feature_name in enumerate(self.morphometric_feature_names):
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
            )
            model.fit(X_pixel, y_morph[:, idx])
            model.save_model(os.path.join(self.model_dir, f"{feature_name}_model.json"))
            self.morphometric_models[feature_name] = model
            predicted_morph.append(model.predict(X_pixel))

        X_deployed = np.hstack([X_pixel, np.vstack(predicted_morph).T])

        self.weight_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.weight_model.fit(X_deployed, y_weight)
        self.weight_model.save_model(
            os.path.join(self.model_dir, "weight_model.json")
        )

        self.bcs_model = xgb.XGBRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )
        self.bcs_model.fit(X_deployed, y_bcs)
        self.bcs_model.save_model(
            os.path.join(self.model_dir, "bcs_model.json")
        )

        log.info("Models trained and saved.")
