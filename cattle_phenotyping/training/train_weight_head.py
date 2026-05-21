"""Train and evaluate the learned weight head.

End-to-end pipeline:

1. Load predicted keypoints per split (JSON: ``{filename: {kp_name: [x, y, conf]}}``).
2. Load sticker pixel areas per split (either from GT masks via
   :func:`cattle_phenotyping.data.mask_io.load_sticker_areas`, or from a
   precomputed JSON of ``{filename: area_px}`` to avoid recomputing).
3. Build features for train + val + test via
   :func:`cattle_phenotyping.pipeline.weight_head_features.build_features`.
4. Fit :class:`cattle_phenotyping.models.weight_head.WeightHead` with
   early stopping on val.
5. Evaluate on test; compute the Schaeffer-only baseline on the same test
   features for side-by-side comparison.
6. Persist model + JSON report.

The CLI is designed to run unmodified on Kaggle after the keypoint training
notebook saves its three prediction JSONs (one per split). Locally, it can
also be pointed at a small subset for smoke-testing.

Example invocation::

    python -m cattle_phenotyping.training.train_weight_head \\
        --dataset-root "$DATASET_ROOT" \\
        --train-csv data/splits/train.csv \\
        --val-csv   data/splits/val.csv \\
        --test-csv  data/splits/test.csv \\
        --train-predictions  data/predictions/yolov8s_pose_train.json \\
        --val-predictions    data/predictions/yolov8s_pose_val.json \\
        --test-predictions   data/predictions/yolov8s_pose_test.json \\
        --sticker-area-by-batch-json data/calibration/sticker_area_cm2_by_batch.json \\
        --workers 8 \\
        --output-model data/results/weight_head \\
        --output-report data/results/weight_head_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from cattle_phenotyping.data.kaggle import (
    CANONICAL_SIDE_KEYPOINTS,
    KaggleSample,
    iter_samples,
    resolve_dataset_root,
)
from cattle_phenotyping.eval.baseline_schaeffer import load_split_filenames
from cattle_phenotyping.eval.flag_suspects import load_filter_set
from cattle_phenotyping.models.weight_head import (
    WeightHead,
    _compute_metrics,
    metrics_to_dict,
)
from cattle_phenotyping.pipeline.weight_head_features import (
    FeatureSkip,
    WEIGHT_HEAD_FEATURE_NAMES,
    build_features,
)
from cattle_phenotyping.utils.log import get_logger, setup_logging
from cattle_phenotyping.utils.seed import seed_everything

log = get_logger(__name__)


# ----------------------------------------------------------- IO helpers


def load_predictions_json(path: Path) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Load a ``{filename: {kp_name: [x, y, conf]}}`` JSON.

    Tolerates list-of-three vs tuple-of-three (JSON only has lists). Keys
    are kept as-is so the caller can join against KaggleSample filenames.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for fname, kps in raw.items():
        per_kp: dict[str, tuple[float, float, float]] = {}
        for kp_name, triple in kps.items():
            if triple is None:
                continue
            x, y, conf = triple
            per_kp[kp_name] = (float(x), float(y), float(conf))
        out[fname] = per_kp
    return out


def load_sticker_areas_json(path: Path) -> dict[str, int]:
    """Load a ``{filename: area_px}`` JSON.

    Optional — when the user wants to skip the per-run mask-loading step,
    they can pre-build this file from a previous run (or from the predicted
    segmenter output once that lands).
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: int(v) for k, v in raw.items() if v is not None}


# ----------------------------------------------------------- core pipeline


def build_split_features(
    samples: Iterable[KaggleSample],
    *,
    predictions_by_name: Mapping[str, Mapping[str, tuple[float, float, float]]],
    sticker_area_by_name: Mapping[str, int],
    sticker_cm2_by_batch: Mapping[str, float],
    girth_multiplier: float = math.pi,
    conf_threshold: float = 0.0,
):
    """Build ``(DataFrame, y, samples_used, skip_counts)`` for one split.

    Skips images where:
      - no keypoint prediction exists
      - no sticker area exists
      - no labelled weight (B2 simplified rows, etc.)
      - any feature-builder skip condition fires

    Returns a pandas DataFrame in :data:`WEIGHT_HEAD_FEATURE_NAMES` order so
    XGBoost sees a consistent column layout.
    """
    import pandas as pd

    rows: list[dict[str, float]] = []
    targets: list[float] = []
    used_samples: list[KaggleSample] = []
    skip_counts: dict[str, int] = defaultdict(int)

    for s in samples:
        name = s.image_path.name
        if s.weight_kg is None:
            skip_counts["no weight label"] += 1
            continue
        preds = predictions_by_name.get(name)
        if not preds:
            skip_counts["no keypoint prediction"] += 1
            continue
        area = sticker_area_by_name.get(name)
        if area is None:
            skip_counts["no sticker area"] += 1
            continue
        result = build_features(
            s, preds, sticker_area_px=area,
            sticker_cm2_by_batch=sticker_cm2_by_batch,
            girth_multiplier=girth_multiplier,
            conf_threshold=conf_threshold,
        )
        if isinstance(result, FeatureSkip):
            skip_counts[result.reason] += 1
            continue
        rows.append(result)
        targets.append(float(s.weight_kg))
        used_samples.append(s)

    df = pd.DataFrame(rows, columns=list(WEIGHT_HEAD_FEATURE_NAMES))
    return df, targets, used_samples, dict(skip_counts)


def schaeffer_only_metrics_from_features(
    X, y_true: list[float],
):
    """Use the ``schaeffer_kg`` column as the prediction. No model needed."""
    schaeffer_preds = X["schaeffer_kg"].tolist()
    return _compute_metrics(y_true, schaeffer_preds)


def per_batch_metrics_from_features(X, y_true, y_pred):
    """Split ``(y_true, y_pred)`` rows by the batch one-hot columns and aggregate."""
    by_batch: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"true": [], "pred": []})
    for i in range(len(y_true)):
        # Recover the batch from the one-hots (only one is 1.0 by construction)
        batch = "B3" if X["batch_B3"].iloc[i] == 1.0 else "B4"
        by_batch[batch]["true"].append(float(y_true[i]))
        by_batch[batch]["pred"].append(float(y_pred[i]))
    return {
        batch: metrics_to_dict(_compute_metrics(d["true"], d["pred"]))
        for batch, d in by_batch.items()
        if d["true"]
    }


# ----------------------------------------------------------- CLI


def _resolve_sticker_areas(
    samples: list[KaggleSample],
    *,
    precomputed_path: Path | None,
    workers: int,
) -> dict[str, int]:
    """Return ``{filename: area_px}`` — either loaded from JSON or computed via mask_io."""
    if precomputed_path is not None:
        log.info("Loading sticker areas from %s", precomputed_path)
        return load_sticker_areas_json(precomputed_path)

    log.info("Computing sticker areas from GT masks (workers=%d)...", workers)
    from cattle_phenotyping.data.mask_io import load_sticker_areas
    jobs = [(i, s.mask_path, s.batch) for i, s in enumerate(samples) if s.mask_path is not None]
    by_idx = load_sticker_areas(jobs, workers=workers)
    return {
        samples[i].image_path.name: area
        for i, area in by_idx.items() if area is not None
    }


def _load_split(
    root: Path, split_csv: Path,
    drop_set: set[str],
) -> list[KaggleSample]:
    """Iterate samples whose filename is in ``split_csv`` and not in ``drop_set``."""
    wanted = load_split_filenames(split_csv)
    out: list[KaggleSample] = []
    for s in iter_samples(root, batches=("B3", "B4"), views=("side",)):
        name = s.image_path.name
        if name in wanted and name not in drop_set:
            out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train + evaluate the learned weight head against the Schaeffer baseline."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)

    parser.add_argument("--train-predictions", required=True,
                        help="JSON of predicted keypoints for train.csv images.")
    parser.add_argument("--val-predictions", required=True)
    parser.add_argument("--test-predictions", required=True)

    parser.add_argument("--sticker-area-by-batch-json", required=True,
                        help="data/calibration/sticker_area_cm2_by_batch.json")
    parser.add_argument("--train-sticker-areas-json", default=None,
                        help="Optional pre-computed {filename: area_px} for train. "
                             "Omit to compute from GT masks via mask_io.")
    parser.add_argument("--val-sticker-areas-json", default=None)
    parser.add_argument("--test-sticker-areas-json", default=None)

    parser.add_argument("--suspect-csv", default="data/calibration/suspect_samples.csv",
                        help="Path to suspect_samples.csv; empty string disables filtering.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Workers for mask loading; ignored when --*-sticker-areas-json are provided.")
    parser.add_argument("--girth-multiplier", type=float, default=math.pi,
                        help="Chord-to-circumference multiplier passed through to Schaeffer feature.")
    parser.add_argument("--conf-threshold", type=float, default=0.0,
                        help="Min predicted keypoint confidence to count as visible.")

    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--early-stopping-rounds", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output-model", default="data/results/weight_head",
                        help="Output stem (no extension); .json + .meta.json files emitted.")
    parser.add_argument("--output-report", default="data/results/weight_head_report.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)
    seed_everything(seed=args.seed)

    # Suspect filter (train + val only; never apply to test).
    if args.suspect_csv:
        drop_set = load_filter_set(Path(args.suspect_csv))
        log.info("Loaded %d filenames to drop from %s", len(drop_set), args.suspect_csv)
    else:
        drop_set = set()

    root = resolve_dataset_root(args.dataset_root)
    sticker_cm2_by_batch = json.loads(Path(args.sticker_area_by_batch_json).read_text(encoding="utf-8"))
    log.info("Per-batch sticker cm²: %s", sticker_cm2_by_batch)

    # Load splits — drop suspects from train/val but NEVER test.
    train_samples = _load_split(root, Path(args.train_csv), drop_set)
    val_samples = _load_split(root, Path(args.val_csv), drop_set)
    test_samples = _load_split(root, Path(args.test_csv), drop_set=set())
    log.info("Samples loaded — train=%d val=%d test=%d",
             len(train_samples), len(val_samples), len(test_samples))

    # Load predictions (one JSON per split).
    train_preds = load_predictions_json(Path(args.train_predictions))
    val_preds = load_predictions_json(Path(args.val_predictions))
    test_preds = load_predictions_json(Path(args.test_predictions))
    log.info("Predictions loaded — train=%d val=%d test=%d",
             len(train_preds), len(val_preds), len(test_preds))

    # Resolve sticker areas (precomputed or via mask_io).
    train_areas = _resolve_sticker_areas(
        train_samples, precomputed_path=Path(args.train_sticker_areas_json)
        if args.train_sticker_areas_json else None, workers=args.workers)
    val_areas = _resolve_sticker_areas(
        val_samples, precomputed_path=Path(args.val_sticker_areas_json)
        if args.val_sticker_areas_json else None, workers=args.workers)
    test_areas = _resolve_sticker_areas(
        test_samples, precomputed_path=Path(args.test_sticker_areas_json)
        if args.test_sticker_areas_json else None, workers=args.workers)

    # Build features.
    log.info("Building train features...")
    X_train, y_train, _, train_skips = build_split_features(
        train_samples, predictions_by_name=train_preds,
        sticker_area_by_name=train_areas,
        sticker_cm2_by_batch=sticker_cm2_by_batch,
        girth_multiplier=args.girth_multiplier,
        conf_threshold=args.conf_threshold,
    )
    log.info("Building val features...")
    X_val, y_val, _, val_skips = build_split_features(
        val_samples, predictions_by_name=val_preds,
        sticker_area_by_name=val_areas,
        sticker_cm2_by_batch=sticker_cm2_by_batch,
        girth_multiplier=args.girth_multiplier,
        conf_threshold=args.conf_threshold,
    )
    log.info("Building test features...")
    X_test, y_test, _, test_skips = build_split_features(
        test_samples, predictions_by_name=test_preds,
        sticker_area_by_name=test_areas,
        sticker_cm2_by_batch=sticker_cm2_by_batch,
        girth_multiplier=args.girth_multiplier,
        conf_threshold=args.conf_threshold,
    )
    log.info("Feature rows — train=%d val=%d test=%d",
             len(X_train), len(X_val), len(X_test))
    log.info("Skip counts — train=%s val=%s test=%s", train_skips, val_skips, test_skips)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        log.error("Empty split after feature build; refusing to train.")
        return 2

    # Train.
    head = WeightHead(params={
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "random_state": args.seed,
    })
    val_metrics = head.fit_with_validation(
        X_train, y_train, X_val, y_val,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    log.info("Val (early-stopping) metrics: MAE=%.2f kg  MAPE=%.2f%%  R²=%.4f",
             val_metrics.mae_kg, val_metrics.mape_pct, val_metrics.r2)

    # Test eval — and Schaeffer-only on the same test feature set for the
    # apples-to-apples comparison row.
    test_preds_kg = head.predict(X_test)
    test_metrics = _compute_metrics(list(y_test), list(test_preds_kg))
    schaeffer_metrics = schaeffer_only_metrics_from_features(X_test, list(y_test))

    log.info("=== Test set comparison ===")
    log.info("  Learned head  : MAE=%.2f kg  MAPE=%.2f%%  R²=%.4f  bias=%+.2f",
             test_metrics.mae_kg, test_metrics.mape_pct, test_metrics.r2, test_metrics.bias_kg)
    log.info("  Schaeffer only: MAE=%.2f kg  MAPE=%.2f%%  R²=%.4f  bias=%+.2f",
             schaeffer_metrics.mae_kg, schaeffer_metrics.mape_pct,
             schaeffer_metrics.r2, schaeffer_metrics.bias_kg)
    log.info("  Δ MAE = %.2f kg (negative = learned head wins)",
             test_metrics.mae_kg - schaeffer_metrics.mae_kg)

    test_by_batch_learned = per_batch_metrics_from_features(X_test, list(y_test), list(test_preds_kg))
    test_by_batch_schaeffer = per_batch_metrics_from_features(X_test, list(y_test), X_test["schaeffer_kg"].tolist())

    # Persist model + report.
    saved_model_path = head.save(args.output_model)
    log.info("Saved model -> %s", saved_model_path)

    report = {
        "config": {
            "girth_multiplier": args.girth_multiplier,
            "conf_threshold": args.conf_threshold,
            "sticker_area_cm2_by_batch": sticker_cm2_by_batch,
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "early_stopping_rounds": args.early_stopping_rounds,
            "seed": args.seed,
            "feature_names": list(WEIGHT_HEAD_FEATURE_NAMES),
        },
        "best_iteration": head.best_iteration,
        "n_rows": {
            "train": int(len(X_train)),
            "val": int(len(X_val)),
            "test": int(len(X_test)),
        },
        "skip_counts": {
            "train": train_skips, "val": val_skips, "test": test_skips,
        },
        "val_metrics": metrics_to_dict(val_metrics),
        "test_metrics_learned": metrics_to_dict(test_metrics),
        "test_metrics_schaeffer_only": metrics_to_dict(schaeffer_metrics),
        "test_metrics_learned_by_batch": test_by_batch_learned,
        "test_metrics_schaeffer_by_batch": test_by_batch_schaeffer,
        "feature_importance": head.feature_importance,
    }
    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote report -> %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
