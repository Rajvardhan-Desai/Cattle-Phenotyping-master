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

    def segment_from_point(
        self,
        image: np.ndarray,
        point_xy: tuple[int, int],
        *,
        max_area_fraction: float = 0.05,
    ) -> np.ndarray:
        """Segment a small object from a single foreground point prompt.

        Tuned for the **sticker** use case: a small, locally-coherent blob on
        the cow's body. SAM returns multiple candidate masks at different
        granularities; we pick the smallest mask that contains the prompt
        point and stays under ``max_area_fraction`` of the image (so SAM
        can't accidentally return "the whole cow" because the point landed
        somewhere ambiguous).

        Args:
            image: BGR numpy array.
            point_xy: ``(x_px, y_px)`` click coordinates in image space.
            max_area_fraction: Reject candidate masks larger than this
                fraction of the image area. Default ``0.05`` (5%) — the
                sticker is typically < 1% but the cap leaves headroom for
                low-resolution uploads. Set to ``1.0`` to accept any size.

        Returns:
            Binary mask (uint8, 0 or 255) of the segmented region. Empty
            mask if no candidate satisfies the size cap.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)

        coords = np.array([[int(point_xy[0]), int(point_xy[1])]])
        labels = np.array([1])  # 1 = foreground

        masks, scores, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )

        # Filter to masks that include the prompt point and stay under the
        # area cap. SAM returns 3 multimask candidates ordered by IoU score;
        # we re-rank by "smallest valid", since stickers are small.
        H, W = image_rgb.shape[:2]
        max_pixels = max_area_fraction * H * W
        click_x, click_y = int(point_xy[0]), int(point_xy[1])

        valid: list[tuple[int, np.ndarray, float]] = []  # (area, mask, sam_score)
        for i in range(masks.shape[0]):
            m = masks[i].astype(bool)
            if not m[click_y, click_x]:
                continue
            area = int(m.sum())
            if area == 0 or area > max_pixels:
                continue
            valid.append((area, m, float(scores[i])))

        if not valid:
            log.warning(
                "SAM point prompt produced no mask under %.1f%% of image area; "
                "returning empty mask. Try clicking more precisely on the sticker.",
                max_area_fraction * 100,
            )
            return np.zeros((H, W), dtype=np.uint8)

        # Smallest valid mask wins — for stickers we want the tight crop.
        valid.sort(key=lambda t: t[0])
        best_mask = valid[0][1]
        mask_u8 = (best_mask.astype(np.uint8)) * 255
        return self._largest_component(mask_u8)

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
