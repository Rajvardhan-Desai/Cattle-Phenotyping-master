"""Flag suspect samples in train/val splits via forward Schaeffer.

After the 2026-05-20 per-batch calibration brought the baseline down to ~30 kg
MAE on test, the outlier list surfaced two recurring failure modes:

1. **Sticker-mask contamination.** A handful of B3 samples have sticker_area_px
   3–5× the typical value (likely background or cow regions whose blue channel
   bleeds into the sticker color tolerance). With ``px_per_cm`` inflated, the
   cm-converted geometry shrinks, and Schaeffer predicts biologically
   impossible weights (1–25 kg for animals labelled 100–250 kg).
2. **Cross-batch animal_id collisions.** The same bare ``animal_id`` appears
   in B3 and B4 with very different labelled weights — animal "314" is 325 kg
   in B3 and 181 kg in B4. The kaggle parser already treats ``(batch,
   animal_id)`` as the unique key (`KaggleSample.animal_key`), so these are
   likely *different* animals sharing an ID, not the same animal weighed
   twice. Either way, both labels can't be trusted in isolation.

**2026-05-21 update: cross-batch ID collisions are noise, not signal.** Running
this module on the full splits surfaced 143 colliding animal_ids covering 1051
sample rows (~28% of train+val) — far too many to be real label errors. With
label std ~46 kg on mean ~160 kg, ANY two random animals will disagree by >20%
most of the time, so the cross-batch flag is essentially catching numeric ID
coincidences between unrelated animals. The real Kaggle ``animal_id`` namespace
is per-batch (and likely per B4 sub-batch — see ``b4-<sub>`` infix). The flag
is preserved in the CSV for diagnostic use but is **excluded by default** from
:func:`load_filter_set`. See ``docs/kaggle_dataset_notes.md`` § ID namespace.

This module materializes those flags into ``data/calibration/suspect_samples.csv``
so the keypoint-training notebook can opt-in to filter or down-weight them.
It does **not** modify the split CSVs — the splits stay deterministic and
auditable. The training code is responsible for joining against this file.

The test split is excluded from the residual-based and implausible-weight
flags (we don't peek at test labels for filtering decisions). Cross-batch
animal_id disagreement is scanned across ALL splits because it's a label-
integrity finding, not a model-quality signal — but only train+val rows
appear in the output CSV.

Output schema (CSV, one row per flagged sample)::

    image_filename, split, batch, animal_id,
    labelled_weight_kg, predicted_weight_kg, residual_kg, residual_sigma_z,
    sticker_area_px, px_per_cm,
    flags  # pipe-separated: "large_residual", "implausible_low_weight",
           # "cross_batch_id_collision"
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from cattle_phenotyping.data.kaggle import KaggleSample, iter_samples, resolve_dataset_root
from cattle_phenotyping.data.mask_io import load_sticker_areas
from cattle_phenotyping.eval.baseline_schaeffer import (
    SchaefferRecord,
    evaluate_sample,
    load_split_filenames,
)
from cattle_phenotyping.utils.log import get_logger, setup_logging
from cattle_phenotyping.utils.seed import seed_everything

log = get_logger(__name__)


# ----------------------------------------------------------------- thresholds

DEFAULT_RESIDUAL_SIGMA = 2.5
DEFAULT_MIN_PLAUSIBLE_WEIGHT_KG = 50.0
DEFAULT_CROSS_BATCH_DISAGREEMENT_PCT = 20.0

FLAG_LARGE_RESIDUAL = "large_residual"
FLAG_IMPLAUSIBLE_LOW_WEIGHT = "implausible_low_weight"
FLAG_CROSS_BATCH_ID_COLLISION = "cross_batch_id_collision"

# Flags that indicate a genuine per-sample defect and are safe to use as a
# training filter. ``cross_batch_id_collision`` is INFORMATIONAL ONLY — see
# the 2026-05-21 finding documented in docs/kaggle_dataset_notes.md: the
# Kaggle dataset's ``animal_id`` namespace is per-batch (and likely per
# B4 sub-batch), so cross-batch ID collisions are almost always coincidences
# between unrelated animals, not mislabeled photos of the same animal.
DEFAULT_FILTER_FLAGS: frozenset[str] = frozenset({
    FLAG_LARGE_RESIDUAL,
    FLAG_IMPLAUSIBLE_LOW_WEIGHT,
})


# ----------------------------------------------------------------- data types


@dataclass
class SuspectRow:
    """One row of the suspect-samples CSV."""

    image_filename: str
    split: str
    batch: str
    animal_id: str | None
    labelled_weight_kg: float
    predicted_weight_kg: float | None
    residual_kg: float | None
    residual_sigma_z: float | None
    sticker_area_px: int | None
    px_per_cm: float | None
    flags: list[str] = field(default_factory=list)


# --------------------------------------------------------- record collection


def collect_records_for_split(
    dataset_root: Path,
    split_csv: Path,
    *,
    split_name: str,
    sticker_by_batch: Mapping[str, float],
    girth_multiplier: float = math.pi,
    workers: int = 4,
    batches: tuple[str, ...] = ("B3", "B4"),
    views: tuple[str, ...] = ("side",),
) -> tuple[list[SchaefferRecord], list[KaggleSample]]:
    """Build (record, sample) lists for every sample in one split CSV.

    Returns the SchaefferRecord list (forward-Schaeffer applied per sample)
    and the underlying KaggleSample list in parallel order. Samples that
    don't appear in the dataset iteration are silently dropped — same
    behavior as :func:`iter_split_samples`.
    """
    wanted = load_split_filenames(split_csv)
    samples = [
        s for s in iter_samples(dataset_root, batches=batches, views=views)  # type: ignore[arg-type]
        if s.image_path.name in wanted
    ]
    log.info("[%s] matched %d samples from %s", split_name, len(samples), split_csv.name)

    jobs = [
        (idx, s.mask_path, s.batch)
        for idx, s in enumerate(samples)
        if s.mask_path is not None
    ]
    areas = load_sticker_areas(jobs, workers=workers)

    records: list[SchaefferRecord] = []
    for idx, sample in enumerate(samples):
        sticker_px = areas.get(idx) if sample.mask_path is not None else None
        records.append(evaluate_sample(
            sample,
            sticker_area_cm2=dict(sticker_by_batch),
            sticker_area_px=sticker_px,
            girth_multiplier=girth_multiplier,
        ))
    return records, samples


# ------------------------------------------------------------ flag detectors


def residual_stdev(records: Iterable[SchaefferRecord]) -> float | None:
    """Population stdev of residuals across records with a prediction."""
    residuals = [r.residual_kg for r in records if r.residual_kg is not None]
    if len(residuals) < 2:
        return None
    return statistics.stdev(residuals)


def detect_large_residual(
    record: SchaefferRecord,
    *,
    stdev_kg: float,
    n_sigma: float,
) -> bool:
    """True if |residual| exceeds ``n_sigma`` × stdev."""
    if record.residual_kg is None or stdev_kg <= 0:
        return False
    return abs(record.residual_kg) > n_sigma * stdev_kg


def detect_implausible_low_weight(
    record: SchaefferRecord,
    *,
    threshold_kg: float,
) -> bool:
    """True if the predicted weight is below a biological-floor threshold.

    Adult/sub-adult cattle in this dataset are >100 kg; predictions below
    50 kg almost always indicate sticker contamination or a keypoint
    collapse, not a real lightweight animal.
    """
    if record.predicted_weight_kg is None:
        return False
    return record.predicted_weight_kg < threshold_kg


def detect_cross_batch_id_collisions(
    samples_by_split: Mapping[str, Iterable[KaggleSample]],
    *,
    disagreement_pct: float,
) -> set[str]:
    """Find bare ``animal_id`` values appearing in 2+ batches with disagreeing labels.

    Scans across every split passed in. Returns the SET of bare animal_id
    strings whose labelled-weight range across batches exceeds
    ``disagreement_pct`` of the smaller weight. (Same animal_id in the same
    batch is fine — it's the cross-batch collisions we suspect.)
    """
    # animal_id -> {batch: [weight, weight, ...]}
    by_id: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for samples in samples_by_split.values():
        for s in samples:
            if s.animal_id is None or s.weight_kg is None:
                continue
            by_id[s.animal_id][s.batch].append(s.weight_kg)

    suspect_ids: set[str] = set()
    for animal_id, by_batch in by_id.items():
        if len(by_batch) < 2:
            continue
        # Use the median weight per batch as the representative.
        batch_weights = [statistics.median(ws) for ws in by_batch.values() if ws]
        if len(batch_weights) < 2:
            continue
        lo, hi = min(batch_weights), max(batch_weights)
        if lo <= 0:
            continue
        if (hi - lo) / lo * 100.0 > disagreement_pct:
            suspect_ids.add(animal_id)
    return suspect_ids


# ----------------------------------------------------------- row construction


def build_suspect_rows(
    records_by_split: Mapping[str, list[SchaefferRecord]],
    *,
    stdev_kg: float | None,
    n_sigma: float,
    min_plausible_kg: float,
    cross_batch_suspect_ids: set[str],
) -> list[SuspectRow]:
    """Assemble the final flagged-row list across train+val.

    A sample is included iff at least one flag triggers. Cross-batch
    collisions are added even when the sample also has a successful
    prediction (the flags compose).
    """
    rows: list[SuspectRow] = []
    for split_name, records in records_by_split.items():
        for r in records:
            flags: list[str] = []
            sigma_z = None
            if r.residual_kg is not None and stdev_kg and stdev_kg > 0:
                sigma_z = r.residual_kg / stdev_kg
                if abs(sigma_z) > n_sigma:
                    flags.append(FLAG_LARGE_RESIDUAL)
            if detect_implausible_low_weight(r, threshold_kg=min_plausible_kg):
                flags.append(FLAG_IMPLAUSIBLE_LOW_WEIGHT)
            if r.sample.animal_id in cross_batch_suspect_ids:
                flags.append(FLAG_CROSS_BATCH_ID_COLLISION)

            if not flags:
                continue

            rows.append(SuspectRow(
                image_filename=r.sample.image_path.name,
                split=split_name,
                batch=r.sample.batch,
                animal_id=r.sample.animal_id,
                labelled_weight_kg=r.labelled_weight_kg,
                predicted_weight_kg=r.predicted_weight_kg,
                residual_kg=r.residual_kg,
                residual_sigma_z=sigma_z,
                sticker_area_px=r.sticker_area_px,
                px_per_cm=r.px_per_cm,
                flags=flags,
            ))
    return rows


# ---------------------------------------------------------------- CSV writing


SUSPECT_CSV_COLUMNS = (
    "image_filename",
    "split",
    "batch",
    "animal_id",
    "labelled_weight_kg",
    "predicted_weight_kg",
    "residual_kg",
    "residual_sigma_z",
    "sticker_area_px",
    "px_per_cm",
    "flags",
)


def write_suspect_csv(rows: Iterable[SuspectRow], output: Path) -> int:
    """Write the rows to a CSV; return the row count written."""
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SUSPECT_CSV_COLUMNS)
        for row in rows:
            writer.writerow([
                row.image_filename,
                row.split,
                row.batch,
                row.animal_id if row.animal_id is not None else "",
                f"{row.labelled_weight_kg:.4f}",
                "" if row.predicted_weight_kg is None else f"{row.predicted_weight_kg:.4f}",
                "" if row.residual_kg is None else f"{row.residual_kg:.4f}",
                "" if row.residual_sigma_z is None else f"{row.residual_sigma_z:.4f}",
                "" if row.sticker_area_px is None else str(row.sticker_area_px),
                "" if row.px_per_cm is None else f"{row.px_per_cm:.4f}",
                "|".join(row.flags),
            ])
            n += 1
    return n


# --------------------------------------------------- training-time filter API


def load_filter_set(
    csv_path: Path,
    *,
    flags: Iterable[str] = DEFAULT_FILTER_FLAGS,
) -> set[str]:
    """Return the set of ``image_filename`` strings to exclude from training.

    Parses the suspect-samples CSV written by :func:`run` / :func:`write_suspect_csv`
    and emits every row whose pipe-separated ``flags`` column intersects the
    given filter set.

    The default :data:`DEFAULT_FILTER_FLAGS` includes only the genuine-defect
    flags (``large_residual`` and ``implausible_low_weight``). The
    ``cross_batch_id_collision`` flag is **intentionally excluded** — see the
    module docstring and ``docs/kaggle_dataset_notes.md`` for why those
    collisions are diagnostic noise, not training-set contamination.

    Typical use from a training notebook::

        from cattle_phenotyping.eval.flag_suspects import load_filter_set
        drop = load_filter_set("data/calibration/suspect_samples.csv")
        train_df = train_df[~train_df["image_filename"].isin(drop)]

    Returns an empty set if the CSV doesn't exist (so a training pipeline
    survives a fresh repo where the calibration step hasn't been run yet —
    the caller can log a warning).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.warning(
            "Suspect-samples CSV not found at %s — returning empty filter set. "
            "Re-run cattle_phenotyping.eval.flag_suspects to generate it.",
            csv_path,
        )
        return set()

    wanted = frozenset(flags)
    drop: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "image_filename" not in (reader.fieldnames or ()) or "flags" not in (reader.fieldnames or ()):
            raise ValueError(
                f"Suspect CSV {csv_path} missing required columns "
                f"'image_filename' and/or 'flags'; got {reader.fieldnames}"
            )
        for row in reader:
            row_flags = {f for f in row["flags"].split("|") if f}
            if row_flags & wanted:
                drop.add(row["image_filename"])
    return drop


# -------------------------------------------------------------------- driver


def run(
    dataset_root: Path,
    *,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path | None,
    sticker_by_batch: Mapping[str, float],
    girth_multiplier: float = math.pi,
    n_sigma: float = DEFAULT_RESIDUAL_SIGMA,
    min_plausible_kg: float = DEFAULT_MIN_PLAUSIBLE_WEIGHT_KG,
    cross_batch_pct: float = DEFAULT_CROSS_BATCH_DISAGREEMENT_PCT,
    workers: int = 4,
    output: Path = Path("data/calibration/suspect_samples.csv"),
    batches: tuple[str, ...] = ("B3", "B4"),
    views: tuple[str, ...] = ("side",),
) -> dict:
    """End-to-end: collect records, derive flags, write CSV, return summary."""

    train_records, train_samples = collect_records_for_split(
        dataset_root, train_csv, split_name="train",
        sticker_by_batch=sticker_by_batch, girth_multiplier=girth_multiplier,
        workers=workers, batches=batches, views=views,
    )
    val_records, val_samples = collect_records_for_split(
        dataset_root, val_csv, split_name="val",
        sticker_by_batch=sticker_by_batch, girth_multiplier=girth_multiplier,
        workers=workers, batches=batches, views=views,
    )

    # Residual stdev computed from TRAIN only (val numbers are downstream).
    stdev_kg = residual_stdev(train_records)
    if stdev_kg is None:
        log.warning("Train residual stdev is undefined (n<2 with predictions).")
    else:
        log.info("Train residual stdev: %.2f kg (threshold = %.1f-sigma = %.2f kg)",
                 stdev_kg, n_sigma, n_sigma * stdev_kg)

    # Cross-batch collision scan uses every split's samples for label inspection.
    samples_by_split: dict[str, list[KaggleSample]] = {
        "train": train_samples, "val": val_samples,
    }
    if test_csv is not None:
        # Test samples participate in collision-detection but NOT in the output CSV.
        test_wanted = load_split_filenames(test_csv)
        test_samples = [
            s for s in iter_samples(dataset_root, batches=batches, views=views)  # type: ignore[arg-type]
            if s.image_path.name in test_wanted
        ]
        samples_by_split["test"] = test_samples
        log.info("[test] matched %d samples (used for cross-batch scan only)", len(test_samples))

    cross_ids = detect_cross_batch_id_collisions(
        samples_by_split, disagreement_pct=cross_batch_pct,
    )
    log.info("Cross-batch animal_id collisions with >%.0f%% weight disagreement: %d ids",
             cross_batch_pct, len(cross_ids))

    rows = build_suspect_rows(
        records_by_split={"train": train_records, "val": val_records},
        stdev_kg=stdev_kg,
        n_sigma=n_sigma,
        min_plausible_kg=min_plausible_kg,
        cross_batch_suspect_ids=cross_ids,
    )

    n_written = write_suspect_csv(rows, output)
    log.info("Wrote %d suspect rows -> %s", n_written, output)

    flag_counts: dict[str, int] = defaultdict(int)
    split_counts: dict[str, int] = defaultdict(int)
    batch_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        split_counts[row.split] += 1
        batch_counts[row.batch] += 1
        for f in row.flags:
            flag_counts[f] += 1

    summary = {
        "n_train_samples": len(train_records),
        "n_val_samples": len(val_records),
        "train_residual_stdev_kg": stdev_kg,
        "n_cross_batch_collision_ids": len(cross_ids),
        "n_suspect_rows": n_written,
        "by_flag": dict(flag_counts),
        "by_split": dict(split_counts),
        "by_batch": dict(batch_counts),
        "config": {
            "sticker_area_cm2_by_batch": dict(sticker_by_batch),
            "girth_multiplier": girth_multiplier,
            "n_sigma": n_sigma,
            "min_plausible_kg": min_plausible_kg,
            "cross_batch_disagreement_pct": cross_batch_pct,
        },
    }
    log.info("Suspect summary: %s", json.dumps(summary["by_flag"]))
    log.info("By split: %s", json.dumps(summary["by_split"]))
    log.info("By batch: %s", json.dumps(summary["by_batch"]))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Flag suspect train/val samples (large residual, implausible weight, "
                    "cross-batch id collision) under per-batch Schaeffer calibration."
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the Kaggle dataset root")
    parser.add_argument("--train-csv", required=True, help="Path to data/splits/train.csv")
    parser.add_argument("--val-csv", required=True, help="Path to data/splits/val.csv")
    parser.add_argument(
        "--test-csv", default=None,
        help="Optional test.csv — when provided, its samples participate in the cross-batch "
             "id-collision scan but are NOT written to the suspect CSV.",
    )
    parser.add_argument(
        "--sticker-area-by-batch-json", required=True,
        help="Path to {batch: cm^2} JSON produced by scale_calibration --by-batch-output.",
    )
    parser.add_argument(
        "--n-sigma", type=float, default=DEFAULT_RESIDUAL_SIGMA,
        help=f"Large-residual threshold in train-set residual stdevs; default {DEFAULT_RESIDUAL_SIGMA}.",
    )
    parser.add_argument(
        "--min-plausible-weight-kg", type=float, default=DEFAULT_MIN_PLAUSIBLE_WEIGHT_KG,
        help=f"Predicted weights below this are flagged implausible; default {DEFAULT_MIN_PLAUSIBLE_WEIGHT_KG}.",
    )
    parser.add_argument(
        "--cross-batch-disagreement-pct", type=float,
        default=DEFAULT_CROSS_BATCH_DISAGREEMENT_PCT,
        help=f"Same animal_id across batches differing by more than this percent is flagged; "
             f"default {DEFAULT_CROSS_BATCH_DISAGREEMENT_PCT}.",
    )
    parser.add_argument(
        "--girth-multiplier", type=float, default=math.pi,
        help="Chord-to-circumference multiplier; default pi.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batches", nargs="+", default=["B3", "B4"])
    parser.add_argument("--views", nargs="+", default=["side"], choices=["side", "rear"])
    parser.add_argument("--output", default="data/calibration/suspect_samples.csv")
    parser.add_argument(
        "--summary-output", default=None,
        help="Optional JSON path that receives the run summary alongside the CSV.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)
    seed_everything()

    root = resolve_dataset_root(args.dataset_root)
    log.info("Resolved dataset root: %s", root)

    sticker_by_batch = {
        str(k): float(v)
        for k, v in json.loads(Path(args.sticker_area_by_batch_json).read_text(encoding="utf-8")).items()
    }
    log.info("Per-batch sticker cm^2: %s", sticker_by_batch)

    summary = run(
        dataset_root=root,
        train_csv=Path(args.train_csv),
        val_csv=Path(args.val_csv),
        test_csv=Path(args.test_csv) if args.test_csv else None,
        sticker_by_batch=sticker_by_batch,
        girth_multiplier=args.girth_multiplier,
        n_sigma=args.n_sigma,
        min_plausible_kg=args.min_plausible_weight_kg,
        cross_batch_pct=args.cross_batch_disagreement_pct,
        workers=args.workers,
        output=Path(args.output),
        batches=tuple(args.batches),
        views=tuple(args.views),
    )

    if args.summary_output:
        sp = Path(args.summary_output)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        log.info("Wrote summary JSON -> %s", sp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
