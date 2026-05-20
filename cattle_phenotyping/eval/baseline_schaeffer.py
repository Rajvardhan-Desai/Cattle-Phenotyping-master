"""Forward Schaeffer baseline on a Kaggle split.

Given the recovered sticker physical area (from
``cattle_phenotyping/pipeline/scale_calibration.py``), this module:

1. Iterates the dataset and filters to samples in a chosen split CSV.
2. For each test sample, loads its sticker mask, counts sticker pixels,
   and computes ``px_per_cm = sqrt(area_px / sticker_area_cm2)``.
3. Forward-applies Schaeffer's formula via the keypoint-derived heart
   girth + body length to predict ``weight_kg``.
4. Aggregates MAE / RMSE / MAPE / R² overall and per-batch.
5. Identifies large-residual samples (|err| > N×σ_residual) as suspect
   annotations — useful for both label QA and choosing where a learned
   model has the most room to improve.

The output of this module is **the bar every learned model must beat** on
the held-out test split. The baseline is zero-parameter, requires no
training data, and is the smallholder formula Acme themselves teach in
their PDF brief — making it a defensible reference point in any report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from cattle_phenotyping.data.kaggle import KaggleSample, iter_samples, resolve_dataset_root
from cattle_phenotyping.data.mask_io import load_sticker_areas
from cattle_phenotyping.models.schaeffer import schaeffer_from_keypoints
from cattle_phenotyping.utils.log import get_logger, setup_logging
from cattle_phenotyping.utils.seed import seed_everything

log = get_logger(__name__)


# Default sticker physical area in cm² — the median value back-derived on
# 4,539 B3+B4 side images via the inverse-Schaeffer procedure on 2026-05-20.
# See docs/kaggle_dataset_notes.md and the Kaggle memory note.
DEFAULT_STICKER_AREA_CM2 = 18.21


# ------------------------------------------------------------------- types


@dataclass
class SchaefferRecord:
    """Per-sample evaluation row."""

    sample: KaggleSample
    labelled_weight_kg: float
    predicted_weight_kg: float | None
    sticker_area_px: int | None
    px_per_cm: float | None
    residual_kg: float | None  # predicted - labelled
    skip_reason: str = ""


@dataclass
class Metrics:
    """Regression metrics on a sample group."""

    n: int
    mae_kg: float
    rmse_kg: float
    mape_pct: float
    r2: float
    bias_kg: float  # mean(pred - true), positive means over-estimating
    mean_true_kg: float


# --------------------------------------------------------- split CSV loading


def load_split_filenames(split_csv: Path) -> set[str]:
    """Return the set of ``image_filename`` values in a splits CSV."""
    names: set[str] = set()
    with split_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "image_filename" not in (reader.fieldnames or ()):
            raise ValueError(
                f"Split CSV {split_csv} missing 'image_filename' column; "
                f"got columns {reader.fieldnames}"
            )
        for row in reader:
            names.add(row["image_filename"])
    return names


def iter_split_samples(
    dataset_root: Path,
    split_csv: Path,
    *,
    batches: tuple[str, ...] = ("B3", "B4"),
    views: tuple[str, ...] = ("side",),
) -> Iterator[KaggleSample]:
    """Yield :class:`KaggleSample` for each entry of a splits CSV.

    Joins the CSV's ``image_filename`` column against the dataset's COCO
    iteration. Samples whose image isn't in the split are skipped silently.
    """
    wanted = load_split_filenames(split_csv)
    for sample in iter_samples(dataset_root, batches=batches, views=views):  # type: ignore[arg-type]
        if sample.image_path.name in wanted:
            yield sample


# ------------------------------------------------------------- evaluation


def evaluate_sample(
    sample: KaggleSample,
    *,
    sticker_area_cm2: float,
    sticker_area_px: int | None,
    girth_multiplier: float = math.pi,
) -> SchaefferRecord:
    """Forward-Schaeffer one sample using the sticker-derived scale."""
    if sample.weight_kg is None:
        return SchaefferRecord(
            sample=sample, labelled_weight_kg=float("nan"),
            predicted_weight_kg=None, sticker_area_px=sticker_area_px,
            px_per_cm=None, residual_kg=None,
            skip_reason="no weight label",
        )

    if sticker_area_px is None:
        return SchaefferRecord(
            sample=sample, labelled_weight_kg=sample.weight_kg,
            predicted_weight_kg=None, sticker_area_px=None,
            px_per_cm=None, residual_kg=None,
            skip_reason="mask file missing or unreadable",
        )
    if sticker_area_px <= 0:
        return SchaefferRecord(
            sample=sample, labelled_weight_kg=sample.weight_kg,
            predicted_weight_kg=None, sticker_area_px=sticker_area_px,
            px_per_cm=None, residual_kg=None,
            skip_reason="zero sticker pixels detected",
        )

    px_per_cm = math.sqrt(sticker_area_px / sticker_area_cm2)
    pred = schaeffer_from_keypoints(
        sample.keypoints, px_per_cm,
        girth_chord_to_circumference=girth_multiplier,
    )
    if pred is None:
        return SchaefferRecord(
            sample=sample, labelled_weight_kg=sample.weight_kg,
            predicted_weight_kg=None, sticker_area_px=sticker_area_px,
            px_per_cm=px_per_cm, residual_kg=None,
            skip_reason="keypoints missing or invisible",
        )

    return SchaefferRecord(
        sample=sample,
        labelled_weight_kg=sample.weight_kg,
        predicted_weight_kg=pred,
        sticker_area_px=sticker_area_px,
        px_per_cm=px_per_cm,
        residual_kg=pred - sample.weight_kg,
    )


# ------------------------------------------------------------ aggregation


def compute_metrics(records: list[SchaefferRecord]) -> Metrics | None:
    """Aggregate MAE / RMSE / MAPE / R² / bias from valid records."""
    valid = [r for r in records if r.predicted_weight_kg is not None and r.residual_kg is not None]
    if not valid:
        return None

    n = len(valid)
    residuals = [r.residual_kg for r in valid]  # type: ignore[misc]
    abs_err = [abs(e) for e in residuals]
    sq_err = [e * e for e in residuals]
    y_true = [r.labelled_weight_kg for r in valid]
    pct_err = [
        abs(r.residual_kg) / r.labelled_weight_kg * 100  # type: ignore[operator]
        for r in valid
        if r.labelled_weight_kg > 0
    ]

    mean_true = sum(y_true) / n
    ss_res = sum(sq_err)
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return Metrics(
        n=n,
        mae_kg=sum(abs_err) / n,
        rmse_kg=math.sqrt(ss_res / n),
        mape_pct=sum(pct_err) / len(pct_err) if pct_err else float("nan"),
        r2=r2,
        bias_kg=sum(residuals) / n,
        mean_true_kg=mean_true,
    )


def group_metrics(
    records: list[SchaefferRecord],
    *,
    key: str,
) -> dict[str, Metrics]:
    """Compute per-group metrics keyed by an attribute of the sample.

    ``key`` is one of ``"batch"``, ``"sex"``, or ``"view"``. Returns a dict
    of group → Metrics, dropping groups with no valid samples.
    """
    if key not in {"batch", "sex", "view"}:
        raise ValueError(f"Unknown group key: {key!r}")

    groups: dict[str, list[SchaefferRecord]] = defaultdict(list)
    for r in records:
        if key == "batch":
            groups[r.sample.batch].append(r)
        elif key == "sex":
            sex = r.sample.filename_meta.sex or "unknown"
            groups[sex].append(r)
        else:  # key == "view"
            groups[r.sample.view].append(r)

    out: dict[str, Metrics] = {}
    for group, items in groups.items():
        m = compute_metrics(items)
        if m is not None:
            out[group] = m
    return out


def flag_outliers(
    records: list[SchaefferRecord],
    *,
    n_sigma: float = 2.5,
    top_k: int | None = None,
) -> list[SchaefferRecord]:
    """Identify rows with unusually large residuals.

    Uses ``n_sigma`` × residual stdev as the threshold; if ``top_k`` is set,
    returns at most that many regardless of sigma (useful when you just want
    the worst-N).
    """
    valid = [r for r in records if r.residual_kg is not None]
    if not valid:
        return []
    residuals = [r.residual_kg for r in valid]  # type: ignore[misc]
    if len(residuals) < 2:
        return []
    stdev = statistics.stdev(residuals)
    threshold = n_sigma * stdev
    outliers = [r for r in valid if abs(r.residual_kg) > threshold]  # type: ignore[operator]
    outliers.sort(key=lambda r: -abs(r.residual_kg))  # type: ignore[arg-type]
    if top_k is not None:
        outliers = outliers[:top_k]
    return outliers


# ------------------------------------------------------------------- driver


def _metrics_to_dict(m: Metrics) -> dict[str, float]:
    return {
        "n": m.n,
        "mae_kg": m.mae_kg,
        "rmse_kg": m.rmse_kg,
        "mape_pct": m.mape_pct,
        "r2": m.r2,
        "bias_kg": m.bias_kg,
        "mean_true_kg": m.mean_true_kg,
    }


def evaluate(
    dataset_root: Path,
    split_csv: Path,
    *,
    sticker_area_cm2: float = DEFAULT_STICKER_AREA_CM2,
    girth_multiplier: float = math.pi,
    workers: int = 4,
    batches: tuple[str, ...] = ("B3", "B4"),
    views: tuple[str, ...] = ("side",),
) -> dict:
    """Run the full Schaeffer baseline evaluation; return a report dict."""
    samples = list(iter_split_samples(dataset_root, split_csv, batches=batches, views=views))
    log.info(
        "Loaded %d samples from split %s (filtered against %s × %s)",
        len(samples), split_csv.name, batches, views,
    )
    if not samples:
        return {
            "n_samples": 0,
            "error": "no samples matched the split CSV against the dataset iteration",
        }

    # Parallel mask loading.
    jobs = [
        (idx, s.mask_path, s.batch)
        for idx, s in enumerate(samples)
        if s.mask_path is not None
    ]
    areas = load_sticker_areas(jobs, workers=workers)
    log.info("Mask areas loaded: %d / %d", sum(1 for v in areas.values() if v), len(samples))

    # Forward Schaeffer per sample.
    records: list[SchaefferRecord] = []
    for idx, sample in enumerate(samples):
        sticker_px = areas.get(idx) if sample.mask_path is not None else None
        records.append(evaluate_sample(
            sample,
            sticker_area_cm2=sticker_area_cm2,
            sticker_area_px=sticker_px,
            girth_multiplier=girth_multiplier,
        ))

    # Aggregate.
    overall = compute_metrics(records)
    by_batch = group_metrics(records, key="batch")
    by_sex = group_metrics(records, key="sex")
    outliers = flag_outliers(records, n_sigma=2.5)

    # Skip-reason histogram.
    skip_counts: dict[str, int] = defaultdict(int)
    for r in records:
        if r.skip_reason:
            skip_counts[r.skip_reason] += 1

    report: dict = {
        "config": {
            "split_csv": str(split_csv),
            "sticker_area_cm2": sticker_area_cm2,
            "girth_multiplier": girth_multiplier,
            "batches": list(batches),
            "views": list(views),
        },
        "n_samples": len(samples),
        "n_with_prediction": sum(1 for r in records if r.predicted_weight_kg is not None),
        "skip_reasons": dict(skip_counts),
        "overall": _metrics_to_dict(overall) if overall else None,
        "by_batch": {k: _metrics_to_dict(v) for k, v in by_batch.items()},
        "by_sex": {k: _metrics_to_dict(v) for k, v in by_sex.items()},
        "outliers_top": [
            {
                "image_filename": r.sample.image_path.name,
                "batch": r.sample.batch,
                "animal_id": r.sample.animal_id,
                "labelled_weight_kg": r.labelled_weight_kg,
                "predicted_weight_kg": r.predicted_weight_kg,
                "residual_kg": r.residual_kg,
                "sticker_area_px": r.sticker_area_px,
                "px_per_cm": r.px_per_cm,
            }
            for r in outliers[:25]
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forward Schaeffer baseline evaluation on a Kaggle split"
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the Kaggle dataset root")
    parser.add_argument("--split-csv", required=True, help="Path to the split CSV (typically data/splits/test.csv)")
    parser.add_argument(
        "--sticker-area-cm2", type=float, default=DEFAULT_STICKER_AREA_CM2,
        help=f"Sticker physical area in cm^2; default {DEFAULT_STICKER_AREA_CM2} (back-derived).",
    )
    parser.add_argument(
        "--girth-multiplier", type=float, default=math.pi,
        help="Chord-to-circumference multiplier; default pi (circular cross-section).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel mask-loading workers.")
    parser.add_argument("--batches", nargs="+", default=["B3", "B4"])
    parser.add_argument("--views", nargs="+", default=["side"], choices=["side", "rear"])
    parser.add_argument("--output", default="data/results/baseline_schaeffer.json")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)
    seed_everything()

    root = resolve_dataset_root(args.dataset_root)
    log.info("Resolved dataset root: %s", root)

    report = evaluate(
        dataset_root=root,
        split_csv=Path(args.split_csv),
        sticker_area_cm2=args.sticker_area_cm2,
        girth_multiplier=args.girth_multiplier,
        workers=args.workers,
        batches=tuple(args.batches),
        views=tuple(args.views),
    )

    if report.get("overall"):
        m = report["overall"]
        log.info(
            "Baseline overall: n=%d  MAE=%.2f kg  RMSE=%.2f kg  MAPE=%.2f%%  R^2=%.4f  bias=%+.2f kg",
            m["n"], m["mae_kg"], m["rmse_kg"], m["mape_pct"], m["r2"], m["bias_kg"],
        )
    for batch, m in report.get("by_batch", {}).items():
        log.info(
            "  [%s] n=%d  MAE=%.2f kg  RMSE=%.2f kg  R^2=%.4f",
            batch, m["n"], m["mae_kg"], m["rmse_kg"], m["r2"],
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote baseline report -> %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
