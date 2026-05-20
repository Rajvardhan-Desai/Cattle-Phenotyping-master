"""
YOLOv8 Cow Detector
Uses YOLOv8n (Ultralytics) pretrained on COCO to detect cows in images.
"""

from ultralytics import YOLO
import numpy as np

# COCO class index for 'cow'
COW_CLASS_ID = 19


class CowDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.3):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> dict:
        """
        Detect cows in the image and return the largest bounding box.

        Args:
            image: BGR numpy array (OpenCV format).

        Returns:
            dict with keys:
                - cow_detected (bool)
                - bbox (list[int]): [x1, y1, x2, y2] or None
                - confidence (float): detection confidence or 0.0
                - cropped_image (np.ndarray): cropped cow region or None
        """
        results = self.model(image, conf=self.confidence, verbose=False)
        detections = results[0]

        cow_boxes = []
        for box in detections.boxes:
            cls_id = int(box.cls[0])
            if cls_id == COW_CLASS_ID:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                cow_boxes.append((coords, conf, area))

        if not cow_boxes:
            return {
                "cow_detected": False,
                "bbox": None,
                "confidence": 0.0,
                "cropped_image": None,
            }

        # Select the largest bounding box
        cow_boxes.sort(key=lambda x: x[2], reverse=True)
        best_box, best_conf, _ = cow_boxes[0]
        x1, y1, x2, y2 = best_box

        cropped = image[y1:y2, x1:x2].copy()

        return {
            "cow_detected": True,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(best_conf, 3),
            "cropped_image": cropped,
        }
