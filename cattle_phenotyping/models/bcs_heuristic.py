"""Heuristic Body Condition Score (BCS) estimator — **demo display only**.

The Kaggle BMGF dataset has **no BCS labels**, so a learned BCS head is not
possible from this corpus. This module exists to give the live demo a BCS
readout for audience comprehension, **not** as a research deliverable. Per
the project plan ([[scope-kaggle-only]] + project_cattle_phenotyping.md),
BCS is explicitly excluded from the IEEE paper.

Anatomy of the heuristic
------------------------

Cattle BCS (1-5 scale; 3 = ideal) correlates visually with how much soft
tissue covers the skeletal frame. The most accessible image proxy is the
**girth-to-length ratio**: a stockier body has a higher ratio at the same
length. We anchor BCS = 3 at the *Bos indicus* zebu cohort's typical
girth/length ≈ 0.48 (estimated from this dataset's training-split median,
documented below in :data:`_RATIO_ANCHOR_BCS3`).

Linear slope: each 0.05 unit of girth/length deviation moves BCS by 0.5 of
a score unit. Clamped to ``[1.5, 4.5]`` because one feature alone can't
reliably resolve the extremes (BCS 1 emaciated / BCS 5 obese visually need
multiple cues — protruding bones, fat cover on tail head, etc.). The
estimator rounds to the nearest 0.5 score, matching how vets report BCS in
practice.

This is **not** a substitute for visual scoring by a trained observer or a
BCS-labelled learned model. Treat the output as an interpretation aid for
the displayed body proportions, not as a clinical assessment.

References:
    AHDB (2018), Body condition scoring of dairy cows. UK Agriculture &
    Horticulture Development Board guide.
    Edmonson et al. (1989), A Body Condition Scoring Chart for Holstein
    Dairy Cows. J. Dairy Sci. 72: 68-78.
"""

from __future__ import annotations

from dataclasses import dataclass


# Anchored at girth/length ratio for "ideal" body condition (BCS 3).
# 0.48 reflects the central tendency of the Acme BMGF zebu training split.
# Adjust if you re-anchor on a different cohort (e.g. Holstein → closer to 0.55).
_RATIO_ANCHOR_BCS3: float = 0.48

# Slope: per 0.05 unit of ratio deviation, BCS changes by 0.5 units → slope = 10.
# Documented as a constant so it's overridable in a sweep.
_BCS_PER_RATIO_UNIT: float = 10.0

# Clamp range. We refuse to claim BCS 1 (emaciated) or 5 (obese) from a
# single image-derived feature; those calls genuinely need a trained eye or
# multiple body-region cues.
_BCS_MIN: float = 1.5
_BCS_MAX: float = 4.5


@dataclass(frozen=True)
class BCSResult:
    """Output of :func:`estimate_bcs_from_ratios`.

    Attributes:
        score: Heuristic BCS on the standard 1-5 scale, rounded to nearest 0.5.
        label: One-word human-readable bucket (``"Thin"``, ``"Ideal"``, etc.).
        raw_score: The pre-clamp, pre-round score for diagnostics / plotting.
    """

    score: float
    label: str
    raw_score: float


def _bcs_label(score: float) -> str:
    """Categorical label for a BCS score (matches AHDB / Edmonson buckets).

    The "Ideal" bucket spans 2.5 < score ≤ 3.5 — the standard vet-practice
    range. 2.0 and below is "Very thin"; 4.0 and above is "Overweight".
    """
    if score <= 2.0:
        return "Very thin"
    if score <= 2.5:
        return "Thin"
    if score <= 3.5:
        return "Ideal"
    if score < 4.5:
        return "Slightly heavy"
    return "Overweight"


def estimate_bcs_from_ratios(
    girth_to_length_ratio_cm: float,
    *,
    anchor_ratio: float = _RATIO_ANCHOR_BCS3,
    slope: float = _BCS_PER_RATIO_UNIT,
    bcs_min: float = _BCS_MIN,
    bcs_max: float = _BCS_MAX,
) -> BCSResult:
    """Heuristic BCS from the girth-to-length ratio.

    Args:
        girth_to_length_ratio_cm: Front-girth-chord ÷ body-length (both cm).
            This is the ``girth_to_length_ratio_cm`` field that the weight-head
            feature builder already computes — pass it through unchanged.
        anchor_ratio: Girth/length value mapped to BCS = 3. Default tuned for
            *Bos indicus* zebu (0.48). Use ~0.55 for Holstein dairy if ever
            reused on that cohort.
        slope: BCS units gained per unit of ratio deviation above the anchor.
            Default 10.0 → 0.05 ratio change ≈ 0.5 BCS change.
        bcs_min, bcs_max: Clamp range. Default ``[1.5, 4.5]`` — see module
            docstring on why we don't claim emaciated/obese from one feature.

    Returns:
        :class:`BCSResult` with rounded score, label, and raw (pre-clamp) value.

    Raises:
        ValueError: if the ratio is non-positive (degenerate body geometry).
    """
    if girth_to_length_ratio_cm <= 0:
        raise ValueError(
            f"girth_to_length_ratio_cm must be positive; got {girth_to_length_ratio_cm}"
        )
    raw = 3.0 + (girth_to_length_ratio_cm - anchor_ratio) * slope
    clamped = max(bcs_min, min(bcs_max, raw))
    rounded = round(clamped * 2) / 2
    return BCSResult(score=rounded, label=_bcs_label(rounded), raw_score=raw)
