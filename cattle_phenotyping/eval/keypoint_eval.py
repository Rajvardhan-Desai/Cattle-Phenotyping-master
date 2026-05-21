"""Evaluation primitives for a trained YOLOv8-pose model.

Two complementary metrics, applied to a dict of predictions keyed by image
filename:

1. **Per-keypoint pixel error** — Euclidean distance between predicted and
   ground-truth keypoint locations, broken out by canonical keypoint name.
   Diagnostic of *where* the keypoint head fails (girth points are usually
   noisier than wither/pinbone). Cheap, no sticker dependency.

2. **Forward Schaeffer with predicted keypoints** — feeds predicted
   keypoints + per-batch sticker scale into
   :func:`cattle_phenotyping.models.schaeffer.schaeffer_from_keypoints` and
   produces ``SchaefferRecord`` lists identical in shape to the GT baseline.
   The MAE / RMSE / MAPE numbers are directly comparable to the
   GT-keypoint baseline number (**30.25 kg MAE / 19.69% MAPE** on test —
   see [[progress-2026-05-21]]).

The Ultralytics YOLO API is **not** imported here — the module operates on
a plain ``{filename: {kp_name: (x, y, conf)}}`` dict. Notebooks construct
that dict from their predictor output and pass it in. This keeps the
module testable without the GPU stack and re-usable across training runs
and frameworks.

Reuses :func:`cattle_phenotyping.eval.baseline_schaeffer.compute_metrics`
and :func:`group_metrics` for downstream aggregation — there's no separate
metrics-summary path for predicted-keypoint weights.
"""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

from cattle_phenotyping.data.kaggle import CANONICAL_SIDE_KEYPOINTS, KaggleSample
from cattle_phenotyping.eval.baseline_schaeffer import SchaefferRecord
from cattle_phenotyping.models.schaeffer import schaeffer_from_keypoints
from cattle_phenotyping.utils.log import get_logger

log = get_logger(__name__)


# ----------------------------------------------------------- type aliases

# One predicted keypoint per canonical name: (x_pixels, y_pixels, confidence_in_[0,1]).
PredictedKeypoint = tuple[float, float, float]

# One image's full prediction: {canonical_kp_name: (x, y, conf)}.
PredictedKeypoints = dict[str, PredictedKeypoint]

# Top-level prediction dict, keyed by image filename (just the base name, no
# directory) so the eval can join against KaggleSample.image_path.name.
PredictionsByFilename = Mapping[str, PredictedKeypoints]


# ------------------------------------- per-keypoint pixel-error computation


_VISIBLE_GT = {1, 2}


def per_keypoint_pixel_errors(
    predictions: PredictionsByFilename,
    samples_by_name: Mapping[str, KaggleSample],
    *,
    keypoint_order: Sequence[str] = CANONICAL_SIDE_KEYPOINTS,
) -> dict[str, list[float]]:
    """Return ``{keypoint_name: [pixel_errors...]}`` over the val set.

    A keypoint contributes one Euclidean distance per image where:

    * the image has a prediction in ``predictions``, AND
    * the canonical keypoint has a visible GT (visibility ∈ {1, 2}), AND
    * the prediction supplied a value for that keypoint name.

    Images / keypoints failing any of these conditions are silently skipped —
    the returned list lengths reveal how many GT keypoints survived. There
    is no "I missed this keypoint" penalty; that's a separate metric (recall
    on visibility) which Ultralytics' OKS-AP covers natively.
    """
    per_kp: dict[str, list[float]] = {name: [] for name in keypoint_order}
    for name, sample in samples_by_name.items():
        pred = predictions.get(name)
        if pred is None:
            continue
        for kp_name in keypoint_order:
            gt = sample.keypoints.get(kp_name)
            if gt is None or gt[2] not in _VISIBLE_GT:
                continue
            pkp = pred.get(kp_name)
            if pkp is None:
                continue
            d = math.hypot(pkp[0] - gt[0], pkp[1] - gt[1])
            per_kp[kp_name].append(d)
    return per_kp


def summarize_pixel_errors(
    per_kp_errs: Mapping[str, list[float]],
) -> dict[str, dict[str, float | int | None]]:
    """Reduce ``per_keypoint_pixel_errors`` output to median / mean / p90 / n.

    Empty lists yield ``{n: 0, median_px: None, mean_px: None, p90_px: None}``
    so a downstream table can render rows for every canonical keypoint name
    even when nothing was evaluated for it.
    """
    out: dict[str, dict[str, float | int | None]] = {}
    for kp_name, errs in per_kp_errs.items():
        if not errs:
            out[kp_name] = {"n": 0, "median_px": None, "mean_px": None, "p90_px": None}
            continue
        sorted_errs = sorted(errs)
        n = len(errs)
        # p90 by floor index — same convention as `numpy.percentile(method='lower')`
        # so the value is always one observed sample, not interpolated.
        p90 = sorted_errs[min(n - 1, int(0.9 * (n - 1)))]
        out[kp_name] = {
            "n": n,
            "median_px": statistics.median(errs),
            "mean_px": statistics.mean(errs),
            "p90_px": p90,
        }
    return out


# --------------------------- forward Schaeffer with predicted keypoints


def _predicted_kp_to_visibility_triple(
    pred: PredictedKeypoint,
    *,
    conf_threshold: float,
) -> tuple[float, float, int]:
    """Convert (x, y, conf) → (x, y, v) for schaeffer_from_keypoints.

    Above the threshold counts as ``v=2`` (visible); below counts as ``v=0``
    (treated as missing by the formula).
    """
    x, y, conf = pred
    return (x, y, 2 if conf >= conf_threshold else 0)


def predict_weights_via_schaeffer(
    predictions: PredictionsByFilename,
    samples: Sequence[KaggleSample],
    sticker_area_px_by_filename: Mapping[str, int],
    sticker_area_cm2_by_batch: Mapping[str, float],
    *,
    girth_multiplier: float = math.pi,
    conf_threshold: float = 0.0,
) -> list[SchaefferRecord]:
    """Apply forward Schaeffer using predicted keypoints + per-batch sticker scale.

    Mirrors the structure of :func:`cattle_phenotyping.eval.baseline_schaeffer.evaluate_sample`
    so the returned ``SchaefferRecord`` list is interchangeable with the
    GT-keypoint baseline records. Compose with
    :func:`cattle_phenotyping.eval.baseline_schaeffer.compute_metrics` and
    :func:`group_metrics` for aggregation.

    Args:
        predictions: Per-image predicted keypoints with confidences. Use the
            adapter that converts your YOLO predictor's output to this shape.
        samples: KaggleSamples corresponding to the dataset rows being evaluated.
            Each sample's ``image_path.name`` is the join key against
            ``predictions`` and ``sticker_area_px_by_filename``.
        sticker_area_px_by_filename: Sticker pixel area per image. Typically
            produced by :func:`cattle_phenotyping.data.mask_io.load_sticker_areas`
            then re-keyed by filename.
        sticker_area_cm2_by_batch: Loaded from
            ``data/calibration/sticker_area_cm2_by_batch.json``.
        girth_multiplier: Chord-to-circumference multiplier; ``π`` by default
            (circular cross-section assumption).
        conf_threshold: Minimum predicted keypoint confidence to count as
            visible (``v=2``). Default ``0.0`` accepts every predicted point.
    """
    records: list[SchaefferRecord] = []
    for sample in samples:
        name = sample.image_path.name
        sticker_px = sticker_area_px_by_filename.get(name)
        labelled = sample.weight_kg
        labelled_for_record = float("nan") if labelled is None else labelled

        if labelled is None:
            records.append(SchaefferRecord(
                sample=sample, labelled_weight_kg=labelled_for_record,
                predicted_weight_kg=None, sticker_area_px=sticker_px,
                px_per_cm=None, residual_kg=None,
                skip_reason="no weight label",
            ))
            continue

        if name not in predictions:
            records.append(SchaefferRecord(
                sample=sample, labelled_weight_kg=labelled,
                predicted_weight_kg=None, sticker_area_px=sticker_px,
                px_per_cm=None, residual_kg=None,
                skip_reason="no model prediction",
            ))
            continue

        if sticker_px is None:
            records.append(SchaefferRecord(
                sample=sample, labelled_weight_kg=labelled,
                predicted_weight_kg=None, sticker_area_px=None,
                px_per_cm=None, residual_kg=None,
                skip_reason="mask file missing or unreadable",
            ))
            continue
        if sticker_px <= 0:
            records.append(SchaefferRecord(
                sample=sample, labelled_weight_kg=labelled,
                predicted_weight_kg=None, sticker_area_px=sticker_px,
                px_per_cm=None, residual_kg=None,
                skip_reason="zero sticker pixels detected",
            ))
            continue

        cm2 = sticker_area_cm2_by_batch.get(sample.batch)
        if cm2 is None or cm2 <= 0:
            records.append(SchaefferRecord(
                sample=sample, labelled_weight_kg=labelled,
                predicted_weight_kg=None, sticker_area_px=sticker_px,
                px_per_cm=None, residual_kg=None,
                skip_reason=f"no sticker cm^2 calibration for batch {sample.batch}",
            ))
            continue

        px_per_cm = math.sqrt(sticker_px / cm2)
        pred_kps = {
            kp_name: _predicted_kp_to_visibility_triple(triple, conf_threshold=conf_threshold)
            for kp_name, triple in predictions[name].items()
        }
        pred_kg = schaeffer_from_keypoints(
            pred_kps, px_per_cm, girth_chord_to_circumference=girth_multiplier,
        )
        if pred_kg is None:
            records.append(SchaefferRecord(
                sample=sample, labelled_weight_kg=labelled,
                predicted_weight_kg=None, sticker_area_px=sticker_px,
                px_per_cm=px_per_cm, residual_kg=None,
                skip_reason="predicted keypoints insufficient",
            ))
            continue

        records.append(SchaefferRecord(
            sample=sample, labelled_weight_kg=labelled,
            predicted_weight_kg=pred_kg, sticker_area_px=sticker_px,
            px_per_cm=px_per_cm, residual_kg=pred_kg - labelled,
        ))
    return records


# ---------------------------------------------- convenience: full eval report


def build_eval_report(
    predictions: PredictionsByFilename,
    samples: Sequence[KaggleSample],
    sticker_area_px_by_filename: Mapping[str, int],
    sticker_area_cm2_by_batch: Mapping[str, float],
    *,
    keypoint_order: Sequence[str] = CANONICAL_SIDE_KEYPOINTS,
    girth_multiplier: float = math.pi,
    conf_threshold: float = 0.0,
) -> dict:
    """End-to-end eval: per-keypoint pixel error + forward-Schaeffer metrics.

    Builds a JSON-serializable report combining both metric layers, ready
    for ``Path("data/results/...").write_text(json.dumps(report, indent=2))``.
    """
    # Local import to avoid a circular import at module load: keypoint_eval is
    # imported by baseline_schaeffer (indirectly) only through SchaefferRecord,
    # but compute_metrics / group_metrics live in baseline_schaeffer too and
    # we want them only at call time.
    from cattle_phenotyping.eval.baseline_schaeffer import compute_metrics, group_metrics

    samples_by_name = {s.image_path.name: s for s in samples}
    per_kp = per_keypoint_pixel_errors(
        predictions, samples_by_name, keypoint_order=keypoint_order,
    )
    pixel_summary = summarize_pixel_errors(per_kp)

    records = predict_weights_via_schaeffer(
        predictions, samples,
        sticker_area_px_by_filename=sticker_area_px_by_filename,
        sticker_area_cm2_by_batch=sticker_area_cm2_by_batch,
        girth_multiplier=girth_multiplier,
        conf_threshold=conf_threshold,
    )
    overall = compute_metrics(records)
    by_batch = group_metrics(records, key="batch")

    # Histogram skip reasons for transparency on what got dropped.
    skip_counts: dict[str, int] = {}
    for r in records:
        if r.skip_reason:
            skip_counts[r.skip_reason] = skip_counts.get(r.skip_reason, 0) + 1

    def _m(m):  # SchaefferRecord-> dict serializer
        if m is None:
            return None
        return {
            "n": m.n, "mae_kg": m.mae_kg, "rmse_kg": m.rmse_kg,
            "mape_pct": m.mape_pct, "r2": m.r2, "bias_kg": m.bias_kg,
            "mean_true_kg": m.mean_true_kg,
        }

    return {
        "config": {
            "sticker_area_cm2_by_batch": dict(sticker_area_cm2_by_batch),
            "girth_multiplier": girth_multiplier,
            "conf_threshold": conf_threshold,
            "keypoint_order": list(keypoint_order),
        },
        "n_samples": len(samples),
        "n_with_prediction": sum(1 for r in records if r.predicted_weight_kg is not None),
        "skip_reasons": skip_counts,
        "pixel_errors_by_keypoint": pixel_summary,
        "weight_overall": _m(overall),
        "weight_by_batch": {k: _m(v) for k, v in by_batch.items()},
    }
