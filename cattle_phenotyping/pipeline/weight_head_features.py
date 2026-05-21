"""Feature builder for the learned weight head.

Maps one ``(sample, predicted_keypoints, sticker_area_px, sticker_cm2_by_batch)``
tuple to a flat dict of features that an XGBoost regressor can consume.

Design choices
--------------

* **Schaeffer prediction is a feature, not the target.** The 2026-05-22
  keypoint run hit forward-Schaeffer val MAE = 28.35 kg with predicted
  keypoints — i.e. Schaeffer is the ceiling at the current keypoint quality.
  Feeding Schaeffer in as a single feature lets the tree ensemble keep the
  Schaeffer signal as a strong prior and add residual corrections (e.g. the
  +4.5 kg systematic bias both batches showed). The model can never do worse
  than "lean entirely on Schaeffer" because that's one branch away.

* **All cm-scale measurements use the predicted keypoints + per-batch sticker
  cm².** That mirrors the eval setup. No leakage of GT keypoints or GT
  sticker masks — features at train time exactly match features at inference
  time, modulo the trained pose/seg models.

* **Batch is one-hot encoded.** The two batches use physically different
  stickers and frame cattle differently; collapsing them to a single
  ``batch`` feature would force the tree to split on it at every node. One-hot
  is one column per batch (``batch_B3``, ``batch_B4``) and lets XGBoost
  isolate per-batch effects cleanly.

* **Keypoint confidences enter as ``mean`` and ``min``** — proxies for
  "how trustworthy is this row's geometry." Adding all 9 per-keypoint
  confs would balloon the feature space; the summary captures the same
  signal with two values.

The returned dict's keys are stable — they're enumerated in
:data:`WEIGHT_HEAD_FEATURE_NAMES` and that ordering is what the
``WeightHead`` XGBoost wrapper expects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from cattle_phenotyping.data.kaggle import KaggleSample
from cattle_phenotyping.models.schaeffer import schaeffer_from_keypoints
from cattle_phenotyping.utils.log import get_logger

log = get_logger(__name__)


# Schema --------------------------------------------------------------------
#
# Ordered list of feature column names. The XGBoost model stores this list
# as ``feature_names`` so loaded models reject DataFrames with the wrong
# columns / order.

WEIGHT_HEAD_FEATURE_NAMES: tuple[str, ...] = (
    "schaeffer_kg",
    "front_girth_chord_cm",
    "rear_girth_chord_cm",
    "body_length_cm",
    "body_height_cm",
    "front_girth_chord_px",
    "body_length_px",
    "sticker_area_px",
    "sticker_cm2",
    "px_per_cm",
    "front_to_rear_girth_ratio_cm",
    "girth_to_length_ratio_cm",
    "length_to_height_ratio_cm",
    "kp_conf_mean",
    "kp_conf_min",
    "batch_B3",
    "batch_B4",
)


# Known batches the one-hot encoder recognizes. Adding a new batch (e.g. B5
# in the future) requires updating this set + WEIGHT_HEAD_FEATURE_NAMES.
KNOWN_BATCHES: tuple[str, ...] = ("B3", "B4")


# Build-time machinery -----------------------------------------------------


PredictedKeypoint = tuple[float, float, float]  # (x_px, y_px, confidence)
PredictedKeypoints = Mapping[str, PredictedKeypoint]


def _kp_xy(kps: PredictedKeypoints, name: str) -> tuple[float, float] | None:
    """Return ``(x, y)`` if the keypoint is present, else ``None``.

    Predicted keypoints have no visibility code (Ultralytics gives a
    confidence float instead). We treat presence in the dict as "predicted"
    — the optional confidence threshold is applied separately in
    :func:`build_features`.
    """
    kp = kps.get(name)
    if kp is None:
        return None
    return float(kp[0]), float(kp[1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class FeatureSkip:
    """Sentinel returned when feature construction is impossible.

    The CLI / batch-feature builder uses ``reason`` to histogram skip causes
    in the training summary (so a 30% skip rate doesn't get hidden in row
    counts).
    """

    reason: str


def build_features(
    sample: KaggleSample,
    predicted_keypoints: PredictedKeypoints,
    sticker_area_px: int,
    sticker_cm2_by_batch: Mapping[str, float],
    *,
    girth_multiplier: float = math.pi,
    conf_threshold: float = 0.0,
) -> dict[str, float] | FeatureSkip:
    """Build the flat feature dict for one sample.

    Returns:
        Dict matching :data:`WEIGHT_HEAD_FEATURE_NAMES` order, or a
        :class:`FeatureSkip` when required inputs are missing. The caller
        is responsible for assembling these into a DataFrame.

    Args:
        sample: KaggleSample (used for batch + image dims; never for GT
            keypoints — those are not features here).
        predicted_keypoints: ``{kp_name: (x_px, y_px, conf)}`` from the
            trained pose model. Names must be the canonical 9 side-view
            keypoints.
        sticker_area_px: Predicted (or GT, depending on the inference
            stack) sticker pixel area in this image.
        sticker_cm2_by_batch: Per-batch sticker cm² constant
            (``data/calibration/sticker_area_cm2_by_batch.json``).
        girth_multiplier: Chord→circumference multiplier, passed through to
            Schaeffer. Default ``π`` (circular cross-section).
        conf_threshold: Drop predicted keypoints with confidence below this
            threshold (treat as missing). Default ``0.0`` accepts every
            predicted point — usually right because the OKS-aware pose head
            already gates on its own confidence.
    """
    if sample.batch not in KNOWN_BATCHES:
        return FeatureSkip(f"unknown batch {sample.batch!r}")
    sticker_cm2 = sticker_cm2_by_batch.get(sample.batch)
    if sticker_cm2 is None or sticker_cm2 <= 0:
        return FeatureSkip(f"no sticker cm² calibration for batch {sample.batch}")
    if sticker_area_px <= 0:
        return FeatureSkip("sticker_area_px <= 0")

    # Apply conf threshold up front: predictions below the threshold are
    # treated as "not predicted" for downstream geometry calls.
    filtered: dict[str, PredictedKeypoint] = {
        name: kp
        for name, kp in predicted_keypoints.items()
        if kp[2] >= conf_threshold
    }
    if not filtered:
        return FeatureSkip("all predicted keypoints below conf threshold")

    # Pixel→cm scale from the sticker area: area_px / cm² = (px/cm)²
    px_per_cm = math.sqrt(sticker_area_px / sticker_cm2)

    # Cm-space measurements (None when the required keypoint pair is missing)
    fg_top = _kp_xy(filtered, "front_girth_top")
    fg_bot = _kp_xy(filtered, "front_girth_bottom")
    rg_top = _kp_xy(filtered, "rear_girth_top")
    rg_bot = _kp_xy(filtered, "rear_girth_bottom")
    shldr = _kp_xy(filtered, "shoulderbone")
    pin = _kp_xy(filtered, "pinbone")
    h_top = _kp_xy(filtered, "height_top")
    h_bot = _kp_xy(filtered, "height_bottom")

    if fg_top is None or fg_bot is None:
        return FeatureSkip("front girth keypoints missing")
    if rg_top is None or rg_bot is None:
        return FeatureSkip("rear girth keypoints missing")
    if shldr is None or pin is None:
        return FeatureSkip("length keypoints missing")
    if h_top is None or h_bot is None:
        return FeatureSkip("height keypoints missing")

    front_girth_chord_px = _distance(fg_top, fg_bot)
    rear_girth_chord_px = _distance(rg_top, rg_bot)
    body_length_px = _distance(shldr, pin)
    body_height_px = _distance(h_top, h_bot)

    if min(front_girth_chord_px, rear_girth_chord_px, body_length_px, body_height_px) <= 0:
        return FeatureSkip("degenerate keypoint geometry (zero-distance pair)")

    front_girth_chord_cm = front_girth_chord_px / px_per_cm
    rear_girth_chord_cm = rear_girth_chord_px / px_per_cm
    body_length_cm = body_length_px / px_per_cm
    body_height_cm = body_height_px / px_per_cm

    # Schaeffer prior using the filtered keypoints. Schaeffer's own helper
    # expects the (x, y, visibility) triple, so we synthesize visibility=2
    # for any keypoint that survived the conf threshold filter — already
    # the case after `filtered` above.
    schaeffer_input = {
        name: (kp[0], kp[1], 2) for name, kp in filtered.items()
    }
    schaeffer_kg = schaeffer_from_keypoints(
        schaeffer_input, px_per_cm, girth_chord_to_circumference=girth_multiplier,
    )
    if schaeffer_kg is None:
        # Should not happen because we already verified the front girth +
        # length pairs above, but keep the guard for safety.
        return FeatureSkip("schaeffer returned None despite resolved keypoints")

    # Body-shape ratios (cm-space; px-space ratios would carry the same
    # information modulo per-image scale, which we already removed).
    front_to_rear_girth_ratio_cm = front_girth_chord_cm / rear_girth_chord_cm
    girth_to_length_ratio_cm = front_girth_chord_cm / body_length_cm
    length_to_height_ratio_cm = body_length_cm / body_height_cm

    # Confidence summary (mean + min across whatever survived filtering).
    confs = [kp[2] for kp in filtered.values()]
    kp_conf_mean = sum(confs) / len(confs)
    kp_conf_min = min(confs)

    # One-hot batch.
    batch_B3 = 1.0 if sample.batch == "B3" else 0.0
    batch_B4 = 1.0 if sample.batch == "B4" else 0.0

    features: dict[str, float] = {
        "schaeffer_kg": float(schaeffer_kg),
        "front_girth_chord_cm": float(front_girth_chord_cm),
        "rear_girth_chord_cm": float(rear_girth_chord_cm),
        "body_length_cm": float(body_length_cm),
        "body_height_cm": float(body_height_cm),
        "front_girth_chord_px": float(front_girth_chord_px),
        "body_length_px": float(body_length_px),
        "sticker_area_px": float(sticker_area_px),
        "sticker_cm2": float(sticker_cm2),
        "px_per_cm": float(px_per_cm),
        "front_to_rear_girth_ratio_cm": float(front_to_rear_girth_ratio_cm),
        "girth_to_length_ratio_cm": float(girth_to_length_ratio_cm),
        "length_to_height_ratio_cm": float(length_to_height_ratio_cm),
        "kp_conf_mean": float(kp_conf_mean),
        "kp_conf_min": float(kp_conf_min),
        "batch_B3": batch_B3,
        "batch_B4": batch_B4,
    }

    # Sanity: the keys we built must exactly match the declared schema.
    # An accidental key mismatch would silently corrupt the trained model's
    # feature order at inference time; catch it loud.
    assert tuple(features.keys()) == WEIGHT_HEAD_FEATURE_NAMES, (
        "Feature dict keys diverged from WEIGHT_HEAD_FEATURE_NAMES — update both."
    )
    return features
