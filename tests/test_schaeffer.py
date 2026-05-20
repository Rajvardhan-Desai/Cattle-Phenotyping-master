"""Tests for the Schaeffer zero-parameter weight baseline."""

from __future__ import annotations

import math

import pytest

from cattle_phenotyping.models.schaeffer import (
    schaeffer_from_keypoints,
    schaeffer_weight_kg,
    schaeffer_weight_lb,
)


# ------------------------------------------------------- pure formula tests


def test_schaeffer_lb_matches_pdf_example():
    """The Acme PDF cites: HG 76" × HG 76" × BL 66" / 300 = 1,270 lb."""
    assert schaeffer_weight_lb(76.0, 66.0) == pytest.approx(1270.4, abs=0.5)


def test_schaeffer_kg_matches_cm_inputs():
    """Heart girth 150 cm, body length 130 cm → 269.88 kg (cross-checked by hand)."""
    assert schaeffer_weight_kg(150.0, 130.0) == pytest.approx(269.88, abs=0.02)


def test_schaeffer_cm_lb_consistency():
    """The cm-fused form must equal lb form × kg/lb after unit conversion."""
    hg_cm, bl_cm = 175.0, 145.0
    hg_in, bl_in = hg_cm / 2.54, bl_cm / 2.54
    via_cm = schaeffer_weight_kg(hg_cm, bl_cm)
    via_lb = schaeffer_weight_lb(hg_in, bl_in) * 0.45359237
    assert via_cm == pytest.approx(via_lb, rel=1e-9)


def test_schaeffer_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive"):
        schaeffer_weight_kg(0.0, 100.0)
    with pytest.raises(ValueError, match="positive"):
        schaeffer_weight_kg(150.0, -1.0)


@pytest.mark.parametrize(
    "hg_cm,bl_cm,expected_min,expected_max",
    [
        # Small calf: HG 90 cm, BL 70 cm
        (90.0, 70.0, 40.0, 65.0),
        # Mid-size cow: HG 165 cm, BL 140 cm  ≈ 350 kg
        (165.0, 140.0, 300.0, 400.0),
        # Large mature: HG 210 cm, BL 180 cm  ≈ 750 kg
        (210.0, 180.0, 700.0, 800.0),
    ],
)
def test_schaeffer_in_realistic_range(hg_cm, bl_cm, expected_min, expected_max):
    """Sanity check that the formula gives biologically plausible weights."""
    w = schaeffer_weight_kg(hg_cm, bl_cm)
    assert expected_min <= w <= expected_max, (
        f"HG={hg_cm} BL={bl_cm} -> {w} kg outside [{expected_min}, {expected_max}]"
    )


# ----------------------------------------------------- applied (keypoint) tests


def _kp(x: float, y: float, v: int = 2) -> tuple[float, float, int]:
    return (x, y, v)


def test_from_keypoints_simple_geometry():
    """With px_to_cm=1.0 and chord/length set up for clean math, verify the result."""
    # Place keypoints so chord = 50 px = 50 cm and length = 100 px = 100 cm.
    kps = {
        "front_girth_top": _kp(100.0, 100.0),
        "front_girth_bottom": _kp(100.0, 150.0),  # vertical 50 px down
        "shoulderbone": _kp(50.0, 200.0),
        "pinbone": _kp(150.0, 200.0),  # horizontal 100 px right
    }
    # HG = chord × π = 50π cm ≈ 157.08 cm
    # weight_kg = HG² × BL × K = (50π)² × 100 × K
    expected_hg = 50.0 * math.pi
    expected_kg = schaeffer_weight_kg(expected_hg, 100.0)
    got = schaeffer_from_keypoints(kps, px_to_cm=1.0)
    assert got == pytest.approx(expected_kg, rel=1e-9)


def test_from_keypoints_respects_px_to_cm():
    """Doubling px_to_cm halves the cm values → halves² × halves on weight."""
    kps = {
        "front_girth_top": _kp(0.0, 0.0),
        "front_girth_bottom": _kp(0.0, 100.0),  # 100 px chord
        "shoulderbone": _kp(0.0, 0.0),
        "pinbone": _kp(200.0, 0.0),  # 200 px length
    }
    w_unit = schaeffer_from_keypoints(kps, px_to_cm=1.0)
    w_double = schaeffer_from_keypoints(kps, px_to_cm=2.0)
    # Doubling px_to_cm halves every cm measurement; weight scales as cm³.
    assert w_double == pytest.approx(w_unit / 8, rel=1e-9)


def test_from_keypoints_missing_returns_none():
    kps = {
        "front_girth_top": _kp(100.0, 100.0),
        # front_girth_bottom missing entirely
        "shoulderbone": _kp(50.0, 200.0),
        "pinbone": _kp(150.0, 200.0),
    }
    assert schaeffer_from_keypoints(kps, px_to_cm=1.0) is None


def test_from_keypoints_invisible_returns_none():
    """visibility=0 means the annotator did not place the keypoint."""
    kps = {
        "front_girth_top": _kp(100.0, 100.0, v=0),  # not labelled
        "front_girth_bottom": _kp(100.0, 150.0),
        "shoulderbone": _kp(50.0, 200.0),
        "pinbone": _kp(150.0, 200.0),
    }
    assert schaeffer_from_keypoints(kps, px_to_cm=1.0) is None


def test_from_keypoints_rejects_bad_scale():
    kps = {
        "front_girth_top": _kp(0.0, 0.0),
        "front_girth_bottom": _kp(0.0, 50.0),
        "shoulderbone": _kp(0.0, 0.0),
        "pinbone": _kp(100.0, 0.0),
    }
    with pytest.raises(ValueError, match="px_to_cm"):
        schaeffer_from_keypoints(kps, px_to_cm=0.0)
    with pytest.raises(ValueError, match="px_to_cm"):
        schaeffer_from_keypoints(kps, px_to_cm=-1.0)


def test_from_keypoints_rear_girth_fallback():
    """If front girth is missing, callers can swap to rear girth."""
    kps = {
        "rear_girth_top": _kp(0.0, 0.0),
        "rear_girth_bottom": _kp(0.0, 60.0),  # chord 60 px
        "shoulderbone": _kp(0.0, 100.0),
        "pinbone": _kp(120.0, 100.0),  # length 120 px
        "front_girth_top": None,
        "front_girth_bottom": None,
    }
    result = schaeffer_from_keypoints(
        kps,
        px_to_cm=1.0,
        girth_keypoints=("rear_girth_top", "rear_girth_bottom"),
    )
    expected_hg = 60.0 * math.pi
    assert result == pytest.approx(schaeffer_weight_kg(expected_hg, 120.0), rel=1e-9)


def test_from_keypoints_alternate_girth_multiplier():
    """Caller can override the chord→circumference multiplier (elliptical body)."""
    kps = {
        "front_girth_top": _kp(0.0, 0.0),
        "front_girth_bottom": _kp(0.0, 50.0),
        "shoulderbone": _kp(0.0, 100.0),
        "pinbone": _kp(120.0, 100.0),
    }
    w_circle = schaeffer_from_keypoints(kps, px_to_cm=1.0)
    w_ellipse = schaeffer_from_keypoints(
        kps, px_to_cm=1.0, girth_chord_to_circumference=2.0,
    )
    # Smaller multiplier → smaller HG → smaller weight (HG² scaling).
    ratio = (2.0 / math.pi) ** 2  # HG² ratio
    assert w_ellipse == pytest.approx(w_circle * ratio, rel=1e-9)
