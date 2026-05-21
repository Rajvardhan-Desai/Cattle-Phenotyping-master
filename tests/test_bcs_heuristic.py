"""Tests for the heuristic BCS estimator.

These exercise the formula's anchor + slope + clamp behaviour so the demo's
BCS readout stays in line with what the module docstring promises. The
heuristic is **not** a learned model — these tests verify properties of the
rule, not predictive accuracy.
"""

from __future__ import annotations

import pytest

from cattle_phenotyping.models.bcs_heuristic import (
    BCSResult,
    estimate_bcs_from_ratios,
)


# --------------------------- anchor + slope --------------------------------


def test_anchor_ratio_maps_to_bcs_three():
    """girth/length == anchor → BCS exactly 3.0 (and 'Ideal')."""
    r = estimate_bcs_from_ratios(0.48)
    assert r.score == 3.0
    assert r.label == "Ideal"
    # raw equals the rounded value at the anchor
    assert abs(r.raw_score - 3.0) < 1e-9


def test_higher_ratio_increases_bcs():
    """Stockier body (higher ratio) → higher BCS."""
    lean = estimate_bcs_from_ratios(0.40)
    ideal = estimate_bcs_from_ratios(0.48)
    stocky = estimate_bcs_from_ratios(0.55)
    assert lean.score < ideal.score < stocky.score


def test_slope_matches_doc_promise():
    """A 0.05 ratio increment moves the raw (pre-clamp) score by 0.5."""
    base = estimate_bcs_from_ratios(0.45)
    plus = estimate_bcs_from_ratios(0.50)
    assert abs((plus.raw_score - base.raw_score) - 0.5) < 1e-9


# --------------------------------- clamp -----------------------------------


def test_extreme_low_ratio_clamps_to_minimum():
    """A very lean cow can't push BCS below 1.5 from one feature alone."""
    r = estimate_bcs_from_ratios(0.10)
    assert r.score == 1.5
    assert r.label == "Very thin"
    # raw should be below the clamp
    assert r.raw_score < 1.5


def test_extreme_high_ratio_clamps_to_maximum():
    """A very heavy cow can't push BCS above 4.5 from one feature alone."""
    r = estimate_bcs_from_ratios(1.20)
    assert r.score == 4.5
    assert r.label == "Overweight"
    assert r.raw_score > 4.5


# --------------------------------- rounding --------------------------------


def test_score_rounds_to_nearest_half():
    """Score is always reported at 0.5 granularity (matches vet practice)."""
    for ratio in (0.40, 0.42, 0.45, 0.48, 0.50, 0.53, 0.56):
        s = estimate_bcs_from_ratios(ratio).score
        # 0.5-step rounding: s*2 must be an integer
        assert abs((s * 2) - round(s * 2)) < 1e-9


# --------------------------------- labels ----------------------------------


@pytest.mark.parametrize("ratio,expected_label", [
    (0.30, "Very thin"),     # well below anchor (rounds to 1.5)
    (0.42, "Thin"),          # ratio 0.42 → raw 2.4 → rounded 2.5 → "Thin"
    (0.48, "Ideal"),         # anchor → 3.0
    (0.52, "Ideal"),         # ratio 0.52 → raw 3.4 → rounded 3.5 → still "Ideal"
    (0.56, "Slightly heavy"),  # ratio 0.56 → raw 3.8 → rounded 4.0 → "Slightly heavy"
    (0.70, "Overweight"),    # clamped to max
])
def test_label_buckets(ratio: float, expected_label: str):
    assert estimate_bcs_from_ratios(ratio).label == expected_label


# ---------------------------- input validation -----------------------------


def test_zero_ratio_raises():
    with pytest.raises(ValueError, match="must be positive"):
        estimate_bcs_from_ratios(0.0)


def test_negative_ratio_raises():
    with pytest.raises(ValueError, match="must be positive"):
        estimate_bcs_from_ratios(-0.1)


# ----------------------- overridable anchor + slope ------------------------


def test_holstein_anchor_shifts_ideal_point():
    """Re-anchoring at 0.55 (Holstein-like) → zebu's 0.48 ratio reads as thin."""
    zebu_ratio = 0.48
    holstein_result = estimate_bcs_from_ratios(zebu_ratio, anchor_ratio=0.55)
    assert holstein_result.score < 3.0
    assert holstein_result.label in ("Thin", "Very thin")


def test_user_demo_value():
    """The 'cattle, batch B4, predicted 190 kg' sample from the live demo.

    The feature builder reports girth_to_length_ratio_cm = 0.475 for this image.
    Expectation: very close to anchor → BCS ≈ 3.0, label 'Ideal'.
    """
    r = estimate_bcs_from_ratios(0.475)
    assert r.score == 3.0
    assert r.label == "Ideal"


def test_returns_bcsresult_type():
    """Sanity: the public API returns a BCSResult, not a bare float."""
    assert isinstance(estimate_bcs_from_ratios(0.48), BCSResult)
