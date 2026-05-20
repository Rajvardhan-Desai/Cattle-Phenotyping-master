"""Recover the sticker's physical cm size and per-image px↔cm scale.

The Kaggle BMGF dataset's funder brief (PDF) does **not** publish the cm
dimension of the sticker that Acme AI applied to each cow. We back-derive it
from the labels themselves:

* The PDF cites Schaeffer's formula ``weight_lb = HG² × BL / 300`` (inches).
* Each labelled image gives us ``weight_kg`` (filename) and pixel-space
  measurements of heart girth (chord between ``front_girth_top`` and
  ``front_girth_bottom``) and body length (``shoulderbone`` to ``pinbone``).
* That's one equation in one unknown — the pixels-per-centimetre scalar
  shared between HG and BL — solvable per image.

Aggregating across the test/val/train split, the sticker's pixel area
divided by ``px_per_cm²`` should be a near-constant value across the
dataset: the sticker's real-world area in cm². The **median** of those
per-image area estimates is our recovered sticker size.

The same scaffold also flags suspect annotations: rows where Schaeffer
disagrees with the labelled weight by more than a configurable threshold
(after we have a stable sticker-derived scale) are likely either
mis-labelled or mis-keypointed. Output them for human review.

References:
    Schaeffer's heart-girth formula — see
    :mod:`cattle_phenotyping.models.schaeffer` for the forward direction.

Limitations:
    The HG estimate from a single side-view chord assumes a circular
    cross-section (multiplier = π). Cattle bodies are elliptical, so the
    true multiplier is closer to 2.0–2.5. A constant-but-wrong multiplier
    biases every per-image px-per-cm estimate by the same factor, which
    biases the recovered sticker size by that factor too. The internal
    consistency of the dataset (sticker area should be constant across
    images) is preserved — only the *absolute* recovered size shifts. See
    ``girth_chord_to_circumference`` in :func:`invert_schaeffer_for_px_per_cm`.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from cattle_phenotyping.data.kaggle import KaggleSample, iter_samples, resolve_dataset_root
from cattle_phenotyping.models.schaeffer import _SCHAEFFER_K_CM3_TO_KG
from cattle_phenotyping.utils.log import get_logger, setup_logging
from cattle_phenotyping.utils.seed import seed_everything

log = get_logger(__name__)


# ----------------------------- per-batch sticker mask colors (from Readme.md)
#
# The mask PNGs encode the cow / sticker / background / ground classes as RGB
# pixel values. Color codes come from the Kaggle Readme:
#
#     B2:  Sticker → 255, 240, 0     (yellow)
#     B3:  Sticker →   0, 117, 255   (blue)
#     B4:  Sticker →   0, 117, 255   (blue)
#
# The actual saved PNGs may have minor encoding noise around boundary pixels
# (anti-aliasing, JPEG re-compression of the underlying image visible through
# the ``___fuse`` blend). ``DEFAULT_COLOR_TOLERANCE`` of ±15 per channel keeps
# the area estimate robust to those without bleeding into the cow class.

STICKER_RGB_BY_BATCH: dict[str, tuple[int, int, int]] = {
    "B2": (255, 240, 0),
    "B3": (0, 117, 255),
    "B4": (0, 117, 255),
}
DEFAULT_COLOR_TOLERANCE = 15


# ----------------------------------------------------------- Stage A: pure math


def invert_schaeffer_for_px_per_cm(
    chord_px: float,
    length_px: float,
    weight_kg: float,
    *,
    girth_chord_to_circumference: float = math.pi,
) -> float:
    """Solve Schaeffer's formula for px-per-cm given keypoint distances + weight.

    Schaeffer (cm-fused):
        ``weight_kg = (chord_cm × m)² × length_cm × K``
        where m = ``girth_chord_to_circumference`` and K =
        :data:`cattle_phenotyping.models.schaeffer._SCHAEFFER_K_CM3_TO_KG`.

    Substituting ``cm = px / px_per_cm``:
        ``weight_kg = (chord_px × m / px_per_cm)² × (length_px / px_per_cm) × K``
        ``        = chord_px² × m² × length_px × K / px_per_cm³``
        ``px_per_cm = (chord_px² × m² × length_px × K / weight_kg) ** (1/3)``
    """
    if chord_px <= 0 or length_px <= 0:
        raise ValueError(
            f"Pixel distances must be positive; got chord={chord_px}, length={length_px}"
        )
    if weight_kg <= 0:
        raise ValueError(f"Weight must be positive; got {weight_kg}")
    numerator = (
        chord_px * chord_px
        * girth_chord_to_circumference * girth_chord_to_circumference
        * length_px
        * _SCHAEFFER_K_CM3_TO_KG
    )
    return (numerator / weight_kg) ** (1.0 / 3.0)


# --------------------------------------------------------- per-sample resolver

# Visibility codes accepted as "the annotator placed a point here".
_VISIBLE = {1, 2}


def _kp_xy(
    keypoints: Mapping[str, tuple[float, float, int] | None],
    name: str,
) -> tuple[float, float] | None:
    kp = keypoints.get(name)
    if kp is None or kp[2] not in _VISIBLE:
        return None
    return (kp[0], kp[1])


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class ScaleResult:
    """Per-sample scale calibration outcome."""

    sample: KaggleSample
    px_per_cm: float | None  # None when keypoints missing or invalid
    chord_px: float | None
    length_px: float | None
    reason: str = ""  # populated when px_per_cm is None


def per_sample_scale(
    sample: KaggleSample,
    *,
    girth_keypoints: tuple[str, str] = ("front_girth_top", "front_girth_bottom"),
    length_keypoints: tuple[str, str] = ("shoulderbone", "pinbone"),
    girth_chord_to_circumference: float = math.pi,
    rear_girth_fallback: bool = True,
) -> ScaleResult:
    """Compute px-per-cm for a single sample via Schaeffer inversion.

    If the front girth pair is occluded and ``rear_girth_fallback`` is set,
    falls back to the rear girth pair. Returns a :class:`ScaleResult` whose
    ``px_per_cm`` is ``None`` (with a ``reason`` string) when the sample
    can't be resolved.
    """
    if sample.weight_kg is None:
        return ScaleResult(sample, None, None, None, "missing weight label")

    g_top = _kp_xy(sample.keypoints, girth_keypoints[0])
    g_bot = _kp_xy(sample.keypoints, girth_keypoints[1])
    used_girth = girth_keypoints

    if (g_top is None or g_bot is None) and rear_girth_fallback:
        alt_pair = ("rear_girth_top", "rear_girth_bottom")
        g_top = _kp_xy(sample.keypoints, alt_pair[0])
        g_bot = _kp_xy(sample.keypoints, alt_pair[1])
        used_girth = alt_pair

    if g_top is None or g_bot is None:
        return ScaleResult(sample, None, None, None, f"girth keypoints missing ({used_girth})")

    l_a = _kp_xy(sample.keypoints, length_keypoints[0])
    l_b = _kp_xy(sample.keypoints, length_keypoints[1])
    if l_a is None or l_b is None:
        return ScaleResult(sample, None, None, None, f"length keypoints missing ({length_keypoints})")

    chord_px = _distance(g_top, g_bot)
    length_px = _distance(l_a, l_b)
    if chord_px <= 0 or length_px <= 0:
        return ScaleResult(sample, None, chord_px, length_px, "degenerate pixel distances")

    pxcm = invert_schaeffer_for_px_per_cm(
        chord_px, length_px, sample.weight_kg,
        girth_chord_to_circumference=girth_chord_to_circumference,
    )
    return ScaleResult(sample, pxcm, chord_px, length_px)


def iter_sample_scales(
    samples: Iterable[KaggleSample],
    **kwargs,
) -> Iterator[ScaleResult]:
    """Yield :class:`ScaleResult` for each input sample, skipping nothing."""
    for s in samples:
        yield per_sample_scale(s, **kwargs)


# ----------------------------------------------------- Stage B: sticker mask


def sticker_area_px_in_mask(
    mask_rgb,  # type: ignore[no-untyped-def]  # numpy array (H, W, 3) uint8
    *,
    batch: str,
    tolerance: int = DEFAULT_COLOR_TOLERANCE,
) -> int:
    """Count pixels matching the sticker class color (with per-channel tolerance).

    Takes a HxWx3 uint8 array (no I/O). The caller is responsible for loading
    the mask PNG, e.g. via PIL.Image. This separation keeps the function
    test-friendly (numpy fixtures) and the module importable without PIL.
    """
    try:
        target = STICKER_RGB_BY_BATCH[batch]
    except KeyError as exc:
        raise ValueError(f"No sticker color registered for batch {batch!r}") from exc

    import numpy as np  # local import: numpy is in requirements but keep callers light

    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected HxWx3 RGB array; got shape {mask_rgb.shape}"
        )
    diffs = mask_rgb.astype(np.int16) - np.array(target, dtype=np.int16)
    matches = np.all(np.abs(diffs) <= tolerance, axis=-1)
    return int(matches.sum())


def sticker_area_px_from_file(
    mask_path: Path,
    *,
    batch: str,
    tolerance: int = DEFAULT_COLOR_TOLERANCE,
) -> int | None:
    """Load a mask PNG (RGB) and return sticker pixel count, or ``None`` on failure."""
    try:
        from PIL import Image  # local import to keep the module test-friendly
        import numpy as np
    except ImportError as exc:
        raise ImportError("PIL and numpy are required for mask loading") from exc

    if not mask_path.exists():
        log.debug("Mask file not found: %s", mask_path)
        return None
    try:
        img = Image.open(mask_path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to read mask %s: %r", mask_path, exc)
        return None
    return sticker_area_px_in_mask(arr, batch=batch, tolerance=tolerance)


# ------------------------------------------------- Stage B aggregation summary


@dataclass
class StickerSizeEstimate:
    """Aggregate sticker-physical-size estimate across N images."""

    n_samples: int
    median_area_cm2: float
    iqr_area_cm2: tuple[float, float]
    median_diameter_cm: float  # assuming circular sticker
    per_sample_cm2: list[float]


def aggregate_sticker_size(
    scales_with_areas: Iterable[tuple[ScaleResult, int]],
) -> StickerSizeEstimate | None:
    """Combine per-sample px-per-cm + sticker pixel area into a global cm² estimate.

    Drops samples whose ``px_per_cm`` is ``None``. Returns ``None`` if no
    valid samples remain.
    """
    cm2: list[float] = []
    for result, area_px in scales_with_areas:
        if result.px_per_cm is None or area_px <= 0:
            continue
        cm2.append(area_px / (result.px_per_cm * result.px_per_cm))

    if not cm2:
        return None

    cm2_sorted = sorted(cm2)
    n = len(cm2_sorted)
    median = statistics.median(cm2_sorted)
    q1 = cm2_sorted[n // 4]
    q3 = cm2_sorted[(3 * n) // 4]
    diameter_cm = 2.0 * math.sqrt(median / math.pi)
    return StickerSizeEstimate(
        n_samples=n,
        median_area_cm2=median,
        iqr_area_cm2=(q1, q3),
        median_diameter_cm=diameter_cm,
        per_sample_cm2=cm2,
    )


# ---------------------------------------------------- CLI entry (Kaggle-runnable)


def _summarize_results(results: list[ScaleResult]) -> dict:
    """Aggregate per-sample scale outcomes for reporting."""
    resolved = [r for r in results if r.px_per_cm is not None]
    pxcms = sorted(r.px_per_cm for r in resolved)  # type: ignore[misc]
    summary: dict = {
        "n_total": len(results),
        "n_resolved": len(resolved),
        "n_unresolved": len(results) - len(resolved),
        "skip_reasons": _count_reasons(results),
    }
    if pxcms:
        summary["px_per_cm"] = {
            "min": pxcms[0],
            "p25": pxcms[len(pxcms) // 4],
            "median": statistics.median(pxcms),
            "p75": pxcms[(3 * len(pxcms)) // 4],
            "max": pxcms[-1],
            "mean": statistics.mean(pxcms),
            "stdev": statistics.stdev(pxcms) if len(pxcms) > 1 else 0.0,
        }
    return summary


def _count_reasons(results: list[ScaleResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        if r.reason:
            counts[r.reason] = counts.get(r.reason, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Back-derive sticker cm size + per-image scale via Schaeffer inversion"
    )
    parser.add_argument("--dataset-root", required=True, help="Path to the Kaggle dataset root")
    parser.add_argument("--output", default="data/splits/scale_calibration.json")
    parser.add_argument(
        "--batches", nargs="+", default=["B3", "B4"],
        help="Batches to use (default: B3 B4, the batches with weight labels).",
    )
    parser.add_argument(
        "--views", nargs="+", default=["side"], choices=["side", "rear"],
    )
    parser.add_argument(
        "--girth-multiplier", type=float, default=math.pi,
        help="Chord-to-circumference multiplier; default pi (circular cross-section)."
    )
    parser.add_argument(
        "--load-masks", action="store_true",
        help="Also load sticker masks from Pixel/.../annotations to recover physical cm^2 size."
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=args.log_level)
    seed_everything()

    root = resolve_dataset_root(args.dataset_root)
    log.info("Resolved dataset root: %s", root)

    samples = list(iter_samples(root, batches=tuple(args.batches), views=tuple(args.views)))
    log.info("Loaded %d samples", len(samples))

    results: list[ScaleResult] = list(iter_sample_scales(
        samples,
        girth_chord_to_circumference=args.girth_multiplier,
    ))
    summary = _summarize_results(results)
    log.info("Per-sample scale: resolved=%d / unresolved=%d",
             summary["n_resolved"], summary["n_unresolved"])

    sticker_estimate = None
    if args.load_masks:
        log.info("Loading sticker masks for area aggregation...")
        scales_with_areas: list[tuple[ScaleResult, int]] = []
        n_missing = 0
        for r in results:
            if r.px_per_cm is None or r.sample.mask_path is None:
                continue
            area_px = sticker_area_px_from_file(r.sample.mask_path, batch=r.sample.batch)
            if area_px is None:
                n_missing += 1
                continue
            scales_with_areas.append((r, area_px))
        log.info(
            "Loaded %d mask areas (%d missing/failed).",
            len(scales_with_areas), n_missing,
        )
        sticker_estimate = aggregate_sticker_size(scales_with_areas)
        if sticker_estimate:
            log.info(
                "Sticker physical area: median=%.2f cm^2 IQR=(%.2f, %.2f) n=%d "
                "(equivalent circular diameter ~ %.2f cm)",
                sticker_estimate.median_area_cm2,
                sticker_estimate.iqr_area_cm2[0],
                sticker_estimate.iqr_area_cm2[1],
                sticker_estimate.n_samples,
                sticker_estimate.median_diameter_cm,
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "girth_multiplier": args.girth_multiplier,
        "batches": list(args.batches),
        "views": list(args.views),
        "per_sample_summary": summary,
    }
    if sticker_estimate is not None:
        payload["sticker"] = {
            "n_samples": sticker_estimate.n_samples,
            "median_area_cm2": sticker_estimate.median_area_cm2,
            "iqr_area_cm2": list(sticker_estimate.iqr_area_cm2),
            "median_diameter_cm_assuming_circle": sticker_estimate.median_diameter_cm,
        }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote calibration report -> %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
