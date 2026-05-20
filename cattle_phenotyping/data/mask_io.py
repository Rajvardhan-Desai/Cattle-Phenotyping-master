"""Shared helpers for loading sticker-mask pixel areas in parallel.

The same per-sample work — open a Kaggle Pixel/*/annotations PNG, threshold
for the sticker color, count matching pixels — is needed by both the scale
back-derivation (Phase 0) and the forward Schaeffer evaluator (Phase 3). The
loader is process-pool parallel because the bottleneck is disk I/O + PIL
decode; see the GPU rationale in
``cattle_phenotyping/pipeline/scale_calibration.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from cattle_phenotyping.utils.log import get_logger

log = get_logger(__name__)


def _load_one(job: tuple[int, str, str]) -> tuple[int, int | None]:
    """Top-level worker for ProcessPoolExecutor — must be picklable.

    Imports happen inside the worker so the parent process doesn't pay the
    PIL/numpy startup tax during the fork.
    """
    from cattle_phenotyping.pipeline.scale_calibration import sticker_area_px_from_file

    idx, mask_path_str, batch = job
    return idx, sticker_area_px_from_file(Path(mask_path_str), batch=batch)


def load_sticker_areas(
    jobs: Sequence[tuple[int, Path, str]],
    *,
    workers: int = 1,
    log_every: int | None = None,
) -> dict[int, int | None]:
    """Resolve ``{job_idx: sticker_area_px or None}`` for each input job.

    Args:
        jobs: Tuples of ``(arbitrary_idx, mask_path, batch)``. The caller
            chooses the idx scheme — e.g. enumeration over a sample list.
            Mask path can be ``None`` only if the caller pre-filters those
            out; this function expects a real path per job.
        workers: ``1`` → run sequentially in the calling process. ``>1`` →
            use ``ProcessPoolExecutor`` with that many workers.
        log_every: Optional progress-log cadence (number of jobs between
            INFO messages). Defaults to about every 5% of the workload.

    Returns:
        A dict keyed by the caller's idx. Missing or unreadable masks map
        to ``None``; never raises on a single bad file.
    """
    if not jobs:
        return {}

    job_tuples = [(idx, str(p), b) for idx, p, b in jobs]
    if log_every is None:
        log_every = max(1, len(job_tuples) // 20)

    results: dict[int, int | None] = {}

    if workers <= 1:
        for n_done, job in enumerate(job_tuples, start=1):
            idx, area = _load_one(job)
            results[idx] = area
            if n_done % log_every == 0:
                log.info("Mask load progress: %d / %d", n_done, len(job_tuples))
        return results

    from concurrent.futures import ProcessPoolExecutor, as_completed

    n_done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_load_one, j) for j in job_tuples]
        for fut in as_completed(futures):
            idx, area = fut.result()
            results[idx] = area
            n_done += 1
            if n_done % log_every == 0:
                log.info("Mask load progress: %d / %d", n_done, len(job_tuples))
    return results
