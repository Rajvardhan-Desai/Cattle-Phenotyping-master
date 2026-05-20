"""
Visualization Utilities
Drawing helpers for bounding boxes, masks, and feature overlays.
"""

import cv2
import numpy as np


def draw_bbox(image: np.ndarray, bbox: list[int], color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Draw a bounding box on the image."""
    vis = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        vis, "Cow", (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
    )
    return vis


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4, color=(0, 200, 0)) -> np.ndarray:
    """Overlay a semi-transparent mask on the image."""
    vis = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    vis = cv2.addWeighted(vis, 1.0, colored_mask, alpha, 0)
    # Draw contour outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, 2)
    return vis


def create_mask_visualization(mask: np.ndarray) -> np.ndarray:
    """Convert binary mask to a 3-channel image for display."""
    vis = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    vis[mask > 0] = (0, 255, 0)
    return vis
