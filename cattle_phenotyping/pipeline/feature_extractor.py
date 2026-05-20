"""
Morphological Feature Extractor
Computes body shape features from a binary segmentation mask using OpenCV.
"""

import cv2
import numpy as np


class FeatureExtractor:
    @staticmethod
    def extract(mask: np.ndarray) -> dict:
        """
        Extract morphological features from a binary cow mask.

        Args:
            mask: Binary mask (0/255, uint8).

        Returns:
            dict of morphological measurements.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return FeatureExtractor._empty_features()

        # Use the largest contour
        contour = max(contours, key=cv2.contourArea)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        x, y, w, h = cv2.boundingRect(contour)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        body_length = w
        body_height = h

        aspect_ratio = w / h if h > 0 else 0.0

        bbox_area = w * h if (w * h) > 0 else 1

        # Normalized morphological features
        body_area_ratio = area / bbox_area
        solidity = area / hull_area if hull_area > 0 else 0.0
        compactness = (perimeter ** 2) / area if area > 0 else 0.0

        return {
            "body_area_px": int(area),
            "body_length_px": int(body_length),
            "body_height_px": int(body_height),
            "bbox_width": int(w),
            "bbox_height": int(h),
            "contour_perimeter": round(perimeter, 2),
            "convex_hull_area": int(hull_area),
            "aspect_ratio": round(aspect_ratio, 2),

            # New scale-invariant features
            "body_area_ratio": round(body_area_ratio, 4),
            "solidity": round(solidity, 4),
            "compactness": round(compactness, 4),
        }

    @staticmethod
    def _empty_features() -> dict:
        return {
            "body_area_px": 0,
            "body_length_px": 0,
            "body_height_px": 0,
            "bbox_width": 0,
            "bbox_height": 0,
            "contour_perimeter": 0.0,
            "convex_hull_area": 0,
            "aspect_ratio": 0.0,

            "body_area_ratio": 0.0,
            "solidity": 0.0,
            "compactness": 0.0,
        }