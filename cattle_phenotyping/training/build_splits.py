"""Build train/val/test splits from the Kaggle Bangladeshi-zebu dataset.

The split is **animal-grouped** (each ``(batch, animal_id)`` lives in exactly
one split — no leakage between side and rear views of the same cow) and
**stratified by weight quartile** (each split covers the full kg range).

Default coverage: ``B3+B4`` × ``side`` view — the only subset where every
sample has a filename-encoded weight label and the canonical 9-keypoint
side-view anatomy. B2 is excluded because its Vector COCO uses sequential
IDs that don't map to its actual animal IDs (see
``docs/kaggle_dataset_notes.md``). Pass extra batches/views via CLI flags
when you want segmentation-only or multi-view splits.

Outputs to ``data/splits/`` (configurable):

* ``train.csv``, ``val.csv``, ``test.csv`` — one row per sample with
  ``image_filename, batch, view, animal_id, weight_kg, sex, age_years_hyp,
  mask_filename``.
* ``manifest.json`` — seed, ratios, per-split animal/sample counts, weight
  summary stats per split (for audit + reproducibility).

Usage::

    python -m cattle_phenotyping.training.build_splits \\
        --dataset-root /kaggle/input/.../www.acmeai.tech\\ Dataset...  \\
        --output-dir data/splits \\
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Literal, Sequence

from cattle_phenotyping.data.kaggle import (
    KaggleSample,
    iter_samples,
    resolve_dataset_root,
)
from cattle_phenotyping.utils.log import get_logger, setup_logging
from cattle_phenotyping.utils.seed import seed_everything

log = get_logger(__name__)


SplitName = Literal["train", "val", "test"]
AnimalKey = tuple[str, str]  # (batch, animal_id)


# --------------------------------------------------------------------- filtering


def filter_weight_samples(samples: Iterable[KaggleSample]) -> list[KaggleSample]:
    """Keep only samples usable for weight-supervised training.

    Drops:
    * B2 simplified-grammar samples (no animal_id, no weight).
    * Any other sample missing a weight label or animal_key (defensive).
    """
    keep: list[KaggleSample] = []
    dropped_no_weight = 0
    dropped_no_animal = 0
    dropped_b2_seq = 0
    for s in samples:
        if s.filename_meta.is_b2_seq_only:
            dropped_b2_seq += 1
            continue
        if s.weight_kg is None:
            dropped_no_weight += 1
            continue
        if s.animal_key is None:
            dropped_no_animal += 1
            continue
        keep.append(s)
    log.info(
        "Sample filter: kept=%d, dropped_b2_seq=%d, dropped_no_weight=%d, "
        "dropped_no_animal=%d",
        len(keep), dropped_b2_seq, dropped_no_weight, dropped_no_animal,
    )
    return keep


# --------------------------------------------------------- per-animal aggregation


def animal_weights(samples: Sequence[KaggleSample]) -> dict[AnimalKey, float]:
    """Return mean weight (kg) per animal across all of that animal's samples.

    Most animals have the same filename-encoded weight across views; the mean
    is robust to the rare cases where they differ slightly.
    """
    sums: dict[AnimalKey, float] = defaultdict(float)
    counts: dict[AnimalKey, int] = defaultdict(int)
    for s in samples:
        assert s.animal_key is not None  # filtered upstream
        assert s.weight_kg is not None
        sums[s.animal_key] += s.weight_kg
        counts[s.animal_key] += 1
    return {key: sums[key] / counts[key] for key in sums}


# ------------------------------------------------------------ stratified split

# Stratify by weight quartile of per-animal mean weight. Quartiles (not deciles)
# keep each stratum populated even for ~500-animal subsets.
_N_STRATA = 4


def _assign_strata(weights: dict[AnimalKey, float]) -> dict[AnimalKey, int]:
    """Bucket animals into ``_N_STRATA`` quantile bins by weight.

    Uses simple rank-based bucketing (not numpy quantiles) so the function
    is deterministic and dependency-light. Ties resolve in animal-key order.
    """
    ordered = sorted(weights.items(), key=lambda kv: (kv[1], kv[0]))
    n = len(ordered)
    out: dict[AnimalKey, int] = {}
    for idx, (key, _) in enumerate(ordered):
        # Floor division of the rank gives the bucket; cap at _N_STRATA-1.
        bucket = min(_N_STRATA - 1, idx * _N_STRATA // max(n, 1))
        out[key] = bucket
    return out


def stratified_animal_split(
    weights: dict[AnimalKey, float],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[SplitName, set[AnimalKey]]:
    """Partition animals into train/val/test, weight-stratified, animal-grouped.

    Within each weight stratum the animals are shuffled (seeded) and split by
    ratio. This keeps the marginal weight distribution similar across splits.
    """
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0; got {ratios} -> {sum(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"Negative ratio in {ratios}")

    strata = _assign_strata(weights)
    rng = random.Random(seed)

    splits: dict[SplitName, set[AnimalKey]] = {"train": set(), "val": set(), "test": set()}

    # Group animal keys by stratum, then split each stratum proportionally.
    by_stratum: dict[int, list[AnimalKey]] = defaultdict(list)
    for key, stratum in strata.items():
        by_stratum[stratum].append(key)

    for stratum, keys in sorted(by_stratum.items()):
        # Sort first for determinism, then shuffle with the seeded rng.
        keys_sorted = sorted(keys)
        rng.shuffle(keys_sorted)
        n = len(keys_sorted)

        # Hamilton's largest-remainder method: round-trip-safe and unbiased,
        # avoids the failure mode where round() turns a small-N stratum's
        # smallest split into zero (e.g. n=5, ratios (0.7,0.15,0.15) under
        # naive rounding gives [4,1,0]; Hamilton gives [3,1,1]).
        exact = [n * r for r in ratios]
        counts = [int(x) for x in exact]  # floor
        remainder = n - sum(counts)
        fractional = sorted(
            ((exact[i] - counts[i], i) for i in range(3)),
            key=lambda t: (-t[0], t[1]),  # largest frac first; tie-break by split order
        )
        for i in range(remainder):
            counts[fractional[i][1]] += 1
        n_train, n_val, n_test = counts

        splits["train"].update(keys_sorted[:n_train])
        splits["val"].update(keys_sorted[n_train : n_train + n_val])
        splits["test"].update(keys_sorted[n_train + n_val :])

    # Sanity: animals shouldn't appear in more than one split.
    inter = (splits["train"] & splits["val"]) | (splits["train"] & splits["test"]) | (
        splits["val"] & splits["test"]
    )
    assert not inter, f"Animal-level leak detected across splits: {inter}"

    return splits


# ---------------------------------------------------------------- CSV / manifest


_CSV_FIELDS = [
    "image_filename",
    "batch",
    "view",
    "animal_id",
    "weight_kg",
    "sex",
    "age_years_hyp",
    "mask_filename",
]


def _row_for(sample: KaggleSample) -> dict[str, str]:
    return {
        "image_filename": sample.image_path.name,
        "batch": sample.batch,
        "view": sample.view,
        "animal_id": sample.animal_id or "",
        "weight_kg": "" if sample.weight_kg is None else f"{sample.weight_kg:g}",
        "sex": sample.filename_meta.sex or "",
        "age_years_hyp": (
            "" if sample.filename_meta.extra is None else f"{sample.filename_meta.extra:g}"
        ),
        "mask_filename": sample.mask_path.name if sample.mask_path else "",
    }


def _weight_summary(samples: Sequence[KaggleSample]) -> dict[str, float]:
    weights = [s.weight_kg for s in samples if s.weight_kg is not None]
    if not weights:
        return {}
    weights.sort()
    n = len(weights)
    return {
        "n": float(n),
        "min": weights[0],
        "max": weights[-1],
        "mean": sum(weights) / n,
        "median": weights[n // 2],
    }


def write_splits(
    samples_by_split: dict[SplitName, list[KaggleSample]],
    *,
    output_dir: Path,
    seed: int,
    ratios: tuple[float, float, float],
    dataset_root: Path,
) -> None:
    """Write split CSVs plus a manifest.json describing how they were built."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "dataset_root": str(dataset_root),
        "splits": {},
    }

    for split_name, samples in samples_by_split.items():
        csv_path = output_dir / f"{split_name}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for s in sorted(samples, key=lambda x: (x.batch, x.animal_id or "", x.view, x.image_path.name)):
                writer.writerow(_row_for(s))

        unique_animals = {s.animal_key for s in samples}
        manifest["splits"][split_name] = {
            "n_samples": len(samples),
            "n_animals": len(unique_animals),
            "weight_kg": _weight_summary(samples),
            "batches": sorted({s.batch for s in samples}),
            "views": sorted({s.view for s in samples}),
        }
        log.info(
            "Wrote %s: %d samples across %d animals -> %s",
            split_name, len(samples), len(unique_animals), csv_path,
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote split manifest -> %s", manifest_path)


# ------------------------------------------------------------------------ driver


def build_splits(
    dataset_root: str | Path,
    *,
    output_dir: str | Path = "data/splits",
    batches: tuple[str, ...] = ("B3", "B4"),
    views: tuple[Literal["side", "rear"], ...] = ("side",),
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[SplitName, list[KaggleSample]]:
    """End-to-end split builder. Returns the in-memory split dict and writes files."""
    resolved_root = resolve_dataset_root(dataset_root)
    log.info("Resolved dataset root: %s", resolved_root)

    raw_samples = list(iter_samples(resolved_root, batches=batches, views=views))
    log.info("Loaded %d raw samples from %s × %s", len(raw_samples), batches, views)

    samples = filter_weight_samples(raw_samples)
    weights = animal_weights(samples)
    log.info("Unique animals with weight labels: %d", len(weights))

    splits = stratified_animal_split(weights, ratios=ratios, seed=seed)

    samples_by_split: dict[SplitName, list[KaggleSample]] = {"train": [], "val": [], "test": []}
    for s in samples:
        for name in ("train", "val", "test"):
            if s.animal_key in splits[name]:  # type: ignore[index]
                samples_by_split[name].append(s)  # type: ignore[index]
                break

    output_dir = Path(output_dir)
    write_splits(
        samples_by_split,
        output_dir=output_dir,
        seed=seed,
        ratios=ratios,
        dataset_root=resolved_root,
    )
    return samples_by_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Kaggle train/val/test splits")
    parser.add_argument("--dataset-root", required=True, help="Path to the Kaggle dataset root")
    parser.add_argument("--output-dir", default="data/splits", help="Where to write split CSVs")
    parser.add_argument(
        "--batches", nargs="+", default=["B3", "B4"],
        help="Batches to include (default: B3 B4). B2 has no usable weight labels.",
    )
    parser.add_argument(
        "--views", nargs="+", default=["side"], choices=["side", "rear"],
        help="Views to include (default: side, the 9-keypoint set).",
    )
    parser.add_argument(
        "--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)
    seed_everything(args.seed)

    build_splits(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        batches=tuple(args.batches),
        views=tuple(args.views),
        ratios=tuple(args.ratios),  # type: ignore[arg-type]
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
