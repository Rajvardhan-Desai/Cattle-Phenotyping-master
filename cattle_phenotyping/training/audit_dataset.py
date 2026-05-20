"""Audit labels and (optionally) image coverage; propose animal_id assignments.

Usage::

    python -m cattle_phenotyping.training.audit_dataset \\
        --labels data/labels.csv \\
        [--data_dir data/images] \\
        [--propose-animal-id data/labels_with_animal_id.csv]

The script never modifies ``--labels`` in place. The proposed-animal-id mode
writes a new CSV; the user is expected to review and rename it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
import sys
from typing import Iterable

from cattle_phenotyping.utils.log import get_logger, setup_logging

log = get_logger(__name__)


MEASUREMENT_COLUMNS = [
    "weight",
    "body_length_cm",
    "withers_height_cm",
    "heart_girth_cm",
    "hip_length_cm",
]


def _group_by_measurements(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Cluster rows by exact match on the measurement columns."""
    grouped: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        key = tuple(row[col] for col in MEASUREMENT_COLUMNS)
        grouped[key].append(row)
    return grouped


def _assign_animal_ids(grouped: dict[tuple, list[dict]]) -> dict[int, list[dict]]:
    """Assign sequential animal_id N>=1 per cluster, ordered by first image_name."""
    ordered = sorted(
        grouped.values(),
        key=lambda rs: min(r["image_name"] for r in rs),
    )
    assignments: dict[int, list[dict]] = {}
    for animal_id, cluster in enumerate(ordered, start=1):
        assignments[animal_id] = cluster
    return assignments


def _write_with_animal_id(
    rows: list[dict],
    assignments: dict[int, list[dict]],
    fieldnames: list[str],
    out_path: str,
) -> None:
    image_to_animal = {
        row["image_name"]: animal_id
        for animal_id, cluster in assignments.items()
        for row in cluster
    }

    new_fields = (
        fieldnames if "animal_id" in fieldnames else fieldnames + ["animal_id"]
    )

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["animal_id"] = image_to_animal[row["image_name"]]
            writer.writerow(out)


def _format_bcs_distribution(rows: Iterable[dict]) -> str:
    counts = collections.Counter(row["bcs"] for row in rows)
    lines = []
    for bcs, count in sorted(counts.items(), key=lambda item: float(item[0])):
        lines.append(f"  {bcs}: {count}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit cattle phenotyping labels")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--data_dir", default=None, help="Optional images dir for coverage check")
    parser.add_argument(
        "--propose-animal-id",
        metavar="OUT_CSV",
        default=None,
        help="Write a copy of labels with a proposed animal_id column to OUT_CSV.",
    )
    args = parser.parse_args(argv)
    setup_logging()

    if not os.path.exists(args.labels):
        log.error("Labels file not found: %s", args.labels)
        return 1

    with open(args.labels, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    log.info("Loaded %d label rows from %s", len(rows), args.labels)

    if args.data_dir and os.path.isdir(args.data_dir):
        image_names = {
            name
            for name in os.listdir(args.data_dir)
            if os.path.isfile(os.path.join(args.data_dir, name))
        }
        labelled_names = {row["image_name"] for row in rows}
        log.info("Image files: %d", len(image_names))
        log.info("Missing labelled images: %s", sorted(labelled_names - image_names))
        log.info("Unlabelled images: %s", sorted(image_names - labelled_names))
    else:
        log.warning(
            "data_dir not provided or does not exist; skipping image coverage check."
        )

    log.info("BCS distribution:\n%s", _format_bcs_distribution(rows))

    grouped = _group_by_measurements(rows)
    duplicate_groups = [cluster for cluster in grouped.values() if len(cluster) > 1]
    rows_in_dups = sum(len(c) for c in duplicate_groups)
    log.info("Unique measurement tuples: %d", len(grouped))
    log.info("Duplicate measurement groups (size > 1): %d", len(duplicate_groups))
    log.info("Rows in duplicate groups: %d", rows_in_dups)

    if rows and "animal_id" not in rows[0]:
        log.warning(
            "animal_id column missing. %d rows in %d duplicate clusters look like "
            "repeat photos of the same animal — run with --propose-animal-id to "
            "emit a CSV with proposed IDs.",
            rows_in_dups,
            len(duplicate_groups),
        )

    if args.propose_animal_id:
        assignments = _assign_animal_ids(grouped)
        _write_with_animal_id(rows, assignments, fieldnames, args.propose_animal_id)
        log.info(
            "Wrote %d rows with proposed animal_id assignments (%d unique animals) to %s",
            len(rows),
            len(assignments),
            args.propose_animal_id,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
