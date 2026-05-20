"""Schaeffer's heart-girth-and-body-length cattle weight formula.

This is the canonical smallholder weight estimator (used in the Acme AI BMGF
brief: ``weight_lb = HG² × BL / 300`` with both measurements in inches). We
implement it as a **zero-parameter baseline** that the learned weight head
must beat on the held-out test split — otherwise we haven't earned the
machine-learning complexity. Report MAE / RMSE / MAPE for Schaeffer alongside
every learned model in the run dir.

Two layers:

* :func:`schaeffer_weight_kg` — pure formula on cm-space measurements.
* :func:`schaeffer_from_keypoints` — applied form that converts pixel-space
  canonical side-view keypoints to cm via a supplied ``px_to_cm`` scalar
  and applies the formula.

The pixel→cm scalar is the output of :mod:`cattle_phenotyping.pipeline.scale_calibration`
(Phase 3 task #20). Until that lands, callers can pass any constant for
sanity-checking the function — see the unit tests.
"""

from __future__ import annotations

import math
from typing import Mapping

from cattle_phenotyping.utils.log import get_logger

log = get_logger(__name__)


# Schaeffer's empirical divisor; units in inches → lb on the right-hand side.
_SCHAEFFER_DIVISOR_IN = 300.0
_INCH_PER_CM = 1.0 / 2.54
_KG_PER_LB = 0.45359237

# Composite constant for the cm→kg fused form:
#   weight_kg = (HG_cm² × BL_cm) × K
# where K = (1/2.54)³ × (1/300) × 0.45359237
_SCHAEFFER_K_CM3_TO_KG = (_INCH_PER_CM ** 3) / _SCHAEFFER_DIVISOR_IN * _KG_PER_LB


def schaeffer_weight_kg(heart_girth_cm: float, body_length_cm: float) -> float:
    """Return Schaeffer's body-weight estimate (kg) from cm measurements.

    >>> # Heart girth 150 cm (=59.0551 in), body length 130 cm (=51.1811 in)
    >>> # → (59.0551)² × 51.1811 / 300 lb = 594.98 lb = 269.88 kg
    >>> round(schaeffer_weight_kg(150.0, 130.0), 2)
    269.88
    """
    if heart_girth_cm <= 0 or body_length_cm <= 0:
        raise ValueError(
            f"Schaeffer requires positive cm measurements; got "
            f"HG={heart_girth_cm}, BL={body_length_cm}"
        )
    return heart_girth_cm * heart_girth_cm * body_length_cm * _SCHAEFFER_K_CM3_TO_KG


def schaeffer_weight_lb(heart_girth_in: float, body_length_in: float) -> float:
    """Schaeffer in its original imperial form. Kept for parity with the PDF."""
    if heart_girth_in <= 0 or body_length_in <= 0:
        raise ValueError(
            f"Schaeffer requires positive inch measurements; got "
            f"HG={heart_girth_in}, BL={body_length_in}"
        )
    return heart_girth_in * heart_girth_in * body_length_in / _SCHAEFFER_DIVISOR_IN


# ----------------------------------------------------------- applied form


# Visibility code in the COCO keypoints array; 0 = not labelled, 1 = labelled
# but occluded, 2 = labelled and visible. We accept 1 and 2 (the annotator
# placed a point); 0 means we have nothing.
_VISIBLE_CODES = {1, 2}


Keypoint = tuple[float, float, int]  # (x_px, y_px, visibility)


def _distance_px(a: Keypoint, b: Keypoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _resolve_kp(
    keypoints: Mapping[str, Keypoint | None],
    name: str,
) -> Keypoint | None:
    """Return ``(x, y, v)`` if visible, else ``None``."""
    kp = keypoints.get(name)
    if kp is None or kp[2] not in _VISIBLE_CODES:
        return None
    return kp


def schaeffer_from_keypoints(
    keypoints: Mapping[str, Keypoint | None],
    px_to_cm: float,
    *,
    girth_keypoints: tuple[str, str] = ("front_girth_top", "front_girth_bottom"),
    length_keypoints: tuple[str, str] = ("shoulderbone", "pinbone"),
    girth_chord_to_circumference: float = math.pi,
) -> float | None:
    """Estimate Schaeffer weight from a canonical side-view keypoint dict.

    Args:
        keypoints: Canonical-name → ``(x_px, y_px, visibility)`` mapping
            (matches :class:`cattle_phenotyping.data.kaggle.KaggleSample.keypoints`).
        px_to_cm: Pixels-per-centimetre conversion. Output of the scale
            calibration stage (sticker-based). Must be > 0.
        girth_keypoints: Which keypoint pair to treat as the chest-depth
            chord. Defaults to ``(front_girth_top, front_girth_bottom)``.
            Pass ``(rear_girth_top, rear_girth_bottom)`` if the front pair
            is occluded.
        length_keypoints: Which keypoint pair defines body length. The
            Acme PDF uses "point of shoulder to point of rump"; our
            ``shoulderbone`` and ``pinbone`` keypoints match.
        girth_chord_to_circumference: Multiplier converting the 2D chord
            (a single diameter measurement from the side view) to an
            approximation of the 3D girth circumference. Defaults to
            ``π`` (assumes circular cross-section); for elliptical
            cross-sections a value in ``[2.0, 2.5]`` may be more accurate.

    Returns:
        Weight in kg, or ``None`` if any required keypoint is missing or
        invisible.
    """
    if px_to_cm <= 0:
        raise ValueError(f"px_to_cm must be positive; got {px_to_cm}")

    g_top = _resolve_kp(keypoints, girth_keypoints[0])
    g_bot = _resolve_kp(keypoints, girth_keypoints[1])
    l_a = _resolve_kp(keypoints, length_keypoints[0])
    l_b = _resolve_kp(keypoints, length_keypoints[1])

    if g_top is None or g_bot is None:
        log.debug("Schaeffer skipped: girth keypoints missing (%s)", girth_keypoints)
        return None
    if l_a is None or l_b is None:
        log.debug("Schaeffer skipped: length keypoints missing (%s)", length_keypoints)
        return None

    chord_cm = _distance_px(g_top, g_bot) / px_to_cm
    body_length_cm = _distance_px(l_a, l_b) / px_to_cm
    heart_girth_cm = chord_cm * girth_chord_to_circumference

    return schaeffer_weight_kg(heart_girth_cm, body_length_cm)
