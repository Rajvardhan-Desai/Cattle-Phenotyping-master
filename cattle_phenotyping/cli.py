"""CLI entry point for the phenotyping pipeline.

Run a single image through the pipeline and print the predicted traits.
Invoked either via ``python main.py`` (root shim) or
``python -m cattle_phenotyping.cli``.
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2

from cattle_phenotyping.pipeline.phenotyping_pipeline import PhenotypingPipeline
from cattle_phenotyping.utils.log import get_logger, setup_logging
from cattle_phenotyping.utils.seed import seed_everything
from cattle_phenotyping.utils.visualization import draw_bbox, overlay_mask

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cattle phenotyping — estimate traits from a side-view image"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to cow image")
    parser.add_argument(
        "--save_vis",
        type=str,
        default=None,
        help="Path to save annotated output image (optional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a YAML config (defaults to configs/default.yaml)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(level=args.log_level)
    seed_everything()

    image = cv2.imread(args.image)
    if image is None:
        log.error("Could not read image at %s", args.image)
        return 1

    pipeline = PhenotypingPipeline(config_path=args.config)
    result = pipeline.run(image)

    if not result["cow_detected"]:
        log.warning("No cow detected in %s", args.image)
        print(json.dumps({"cow_detected": False}, indent=2))
        return 0

    output = {
        "cow_detected": result["cow_detected"],
        "detection_confidence": result["detection_confidence"],
        "bbox": result["bbox"],
        **result["features"],
        "estimated_weight_kg": result["estimated_weight_kg"],
        "body_condition_score": result["body_condition_score"],
        "body_length_cm": result.get("body_length_cm", 0.0),
        "withers_height_cm": result.get("withers_height_cm", 0.0),
        "heart_girth_cm": result.get("heart_girth_cm", 0.0),
        "hip_length_cm": result.get("hip_length_cm", 0.0),
    }
    print(json.dumps(output, indent=2))

    if args.save_vis:
        vis = draw_bbox(image, result["bbox"])
        vis = overlay_mask(vis, result["mask"])
        cv2.imwrite(args.save_vis, vis)
        log.info("Annotated image saved to %s", args.save_vis)

    return 0


if __name__ == "__main__":
    sys.exit(main())
