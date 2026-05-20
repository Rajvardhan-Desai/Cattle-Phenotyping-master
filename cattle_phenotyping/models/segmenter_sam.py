"""
SAM Cow Segmenter
Uses Segment Anything Model (ViT-B) to segment the cow given a bounding box prompt.
"""

import os
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

from cattle_phenotyping.utils.log import get_logger

DEFAULT_CHECKPOINT = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

log = get_logger(__name__)


class CowSegmenter:
    def __init__(
        self,
        checkpoint_path: str = DEFAULT_CHECKPOINT,
        allow_auto_download: bool = False,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(checkpoint_path):
            if not allow_auto_download:
                raise FileNotFoundError(
                    f"SAM checkpoint not found at {checkpoint_path}. "
                    "Set segmenter.allow_auto_download: true in the config "
                    f"to fetch it from {CHECKPOINT_URL}, or vendor the file."
                )
            log.warning("Downloading SAM ViT-B checkpoint to %s", checkpoint_path)
            import urllib.request
            urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint_path)
            log.info("SAM checkpoint download complete.")

        sam = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint_path)
        sam.to(device)
        self.predictor = SamPredictor(sam)

    def segment(self, image: np.ndarray, bbox: list[int]) -> np.ndarray:
        """
        Segment the cow using SAM with a bounding box prompt.

        Args:
            image: BGR numpy array.
            bbox: [x1, y1, x2, y2] bounding box of the detected cow.

        Returns:
            Binary mask (uint8, 0 or 255) of the cow region.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        input_box = np.array(bbox)
        masks, scores, _ = self.predictor.predict(
            box=input_box[None, :],
            multimask_output=True,
        )

        # Pick the mask with the highest score
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(np.uint8) * 255

        # Keep only the largest connected component
        mask = self._largest_component(mask)

        return mask

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component in a binary mask."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if num_labels <= 1:
            return mask

        # Label 0 is background; find the largest foreground component
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        clean_mask = np.zeros_like(mask)
        clean_mask[labels == largest_label] = 255
        return clean_mask
