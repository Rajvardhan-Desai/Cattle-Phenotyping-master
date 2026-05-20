"""Tests for the binary-mask feature extractor."""

import cv2
import numpy as np
import pytest

from cattle_phenotyping.pipeline.feature_extractor import FeatureExtractor


@pytest.fixture
def empty_mask() -> np.ndarray:
    return np.zeros((200, 300), dtype=np.uint8)


@pytest.fixture
def rectangle_mask() -> np.ndarray:
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(mask, (50, 40), (250, 160), 255, -1)
    return mask


@pytest.fixture
def circle_mask() -> np.ndarray:
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 60, 255, -1)
    return mask


def test_empty_mask_returns_zero_features(empty_mask):
    features = FeatureExtractor.extract(empty_mask)
    assert features["body_area_px"] == 0
    assert features["body_length_px"] == 0
    assert features["aspect_ratio"] == 0.0
    assert features["solidity"] == 0.0


def test_rectangle_mask_has_expected_dimensions(rectangle_mask):
    features = FeatureExtractor.extract(rectangle_mask)
    # cv2.boundingRect uses inclusive-exclusive bounds; allow ±1 px tolerance.
    assert features["bbox_width"] == pytest.approx(201, abs=1)
    assert features["bbox_height"] == pytest.approx(121, abs=1)
    # A filled rectangle is convex => solidity ≈ 1.
    assert features["solidity"] == pytest.approx(1.0, abs=0.01)


def test_circle_mask_is_compact(circle_mask):
    features = FeatureExtractor.extract(circle_mask)
    # 4*pi ≈ 12.57 is the theoretical minimum compactness (perfect circle).
    # Discretization pushes it above that; allow generous bound.
    assert 12.0 < features["compactness"] < 16.0
    assert features["solidity"] == pytest.approx(1.0, abs=0.05)
    assert features["aspect_ratio"] == pytest.approx(1.0, abs=0.1)
