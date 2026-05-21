"""Tests for cattle keypoint sigmas + Ultralytics patch.

The Ultralytics-dependent tests are skipped when ``ultralytics`` isn't
importable (e.g. on CI without the GPU stack). The data-only tests run
unconditionally so the sigma table itself can't drift from
``CANONICAL_SIDE_KEYPOINTS`` without CI catching it.
"""

from __future__ import annotations

import pytest

from cattle_phenotyping.data.kaggle import CANONICAL_SIDE_KEYPOINTS
from cattle_phenotyping.training.kpt_sigmas import (
    CATTLE_KEYPOINT_SIGMAS,
    _SIGMA_BY_NAME,
    apply_cattle_pose_sigmas,
    restore_default_pose_sigmas,
)


# ----------------------------------------------------------- data tests


def test_sigma_array_length_matches_canonical_keypoints():
    """The exported sigma tuple must have one value per canonical keypoint."""
    assert len(CATTLE_KEYPOINT_SIGMAS) == len(CANONICAL_SIDE_KEYPOINTS)


def test_sigma_by_name_covers_every_canonical_keypoint():
    """Adding a canonical keypoint without a sigma should be impossible to ship."""
    missing = set(CANONICAL_SIDE_KEYPOINTS) - set(_SIGMA_BY_NAME.keys())
    assert not missing, f"_SIGMA_BY_NAME missing entries for {missing}"


def test_sigmas_are_in_canonical_order():
    """Tuple ordering must match CANONICAL_SIDE_KEYPOINTS, not the dict insertion order."""
    expected = tuple(_SIGMA_BY_NAME[name] for name in CANONICAL_SIDE_KEYPOINTS)
    assert CATTLE_KEYPOINT_SIGMAS == expected


def test_sigmas_are_tighter_than_ultralytics_default():
    """Each sigma must be below the 0.111 (= 1/9) default Ultralytics falls back to.

    If a future contributor loosens a sigma above the default, that defeats the
    point of the patch — call it out in CI.
    """
    DEFAULT = 1.0 / len(CANONICAL_SIDE_KEYPOINTS)  # 0.111
    for name, s in zip(CANONICAL_SIDE_KEYPOINTS, CATTLE_KEYPOINT_SIGMAS):
        assert s < DEFAULT, f"{name}: sigma {s} >= default {DEFAULT}; loosens the metric"


def test_sigmas_are_positive():
    for name, s in zip(CANONICAL_SIDE_KEYPOINTS, CATTLE_KEYPOINT_SIGMAS):
        assert s > 0, f"{name}: sigma must be > 0; got {s}"


def test_skeletal_landmarks_have_tightest_sigma():
    """wither/pinbone/shoulderbone are anatomical bone protrusions; they
    should have the smallest sigma in the table (most precise localization).
    """
    skeletal = ("wither", "pinbone", "shoulderbone")
    outline = ("front_girth_top", "front_girth_bottom",
               "rear_girth_top", "rear_girth_bottom",
               "height_top", "height_bottom")
    max_skeletal = max(_SIGMA_BY_NAME[n] for n in skeletal)
    min_outline = min(_SIGMA_BY_NAME[n] for n in outline)
    assert max_skeletal <= min_outline, (
        f"skeletal max sigma {max_skeletal} should be <= outline min {min_outline}; "
        "otherwise the table inverts anatomical precision"
    )


# ----------------------------------------------- Ultralytics patch tests


pytest.importorskip("ultralytics")


def test_apply_cattle_pose_sigmas_patches_pose_loss():
    """v8PoseLoss.__init__ wrapper should set self.sigmas after patch.

    The wrapper does ``getattr(type(self), _ORIG_LOSS_ATTR)`` to find the saved
    original, so we run it on a *subclass* of ``v8PoseLoss`` where we've set
    the attribute to a no-op. This isolates the test from Ultralytics' real
    init (which needs a full YOLO model graph to run).
    """
    import torch
    from ultralytics.utils.loss import v8PoseLoss

    try:
        apply_cattle_pose_sigmas()

        captured_device = torch.device("cpu")

        class FakePoseLoss(v8PoseLoss):
            # Override the saved-original attribute on the subclass so the
            # wrapper's getattr() picks this up at call time.
            _cattle_orig_init = staticmethod(  # type: ignore[assignment]
                lambda self, model: setattr(self, "device", captured_device)
            )

        instance = FakePoseLoss.__new__(FakePoseLoss)
        FakePoseLoss.__init__(instance, model=None)  # type: ignore[arg-type]
        assert hasattr(instance, "sigmas")
        assert instance.sigmas.shape == (len(CATTLE_KEYPOINT_SIGMAS),)
        for got, want in zip(instance.sigmas.tolist(), CATTLE_KEYPOINT_SIGMAS):
            assert abs(got - want) < 1e-6
    finally:
        restore_default_pose_sigmas()


def test_apply_cattle_pose_sigmas_patches_pose_validator():
    """PoseValidator.init_metrics wrapper should overwrite self.sigma after patch.

    Same isolation pattern: subclass PoseValidator and stub the saved-original
    attribute on the subclass.
    """
    import numpy as np
    from ultralytics.models.yolo.pose.val import PoseValidator

    try:
        apply_cattle_pose_sigmas()

        class FakeValidator(PoseValidator):
            _cattle_orig_init_metrics = staticmethod(  # type: ignore[assignment]
                lambda self, model: None
            )

            def __init__(self):
                # Bypass PoseValidator.__init__ (needs a real dataloader, etc.)
                pass

        stub = FakeValidator()
        FakeValidator.init_metrics(stub, model=None)
        assert hasattr(stub, "sigma")
        assert isinstance(stub.sigma, np.ndarray)
        assert stub.sigma.shape == (len(CATTLE_KEYPOINT_SIGMAS),)
        np.testing.assert_allclose(stub.sigma, CATTLE_KEYPOINT_SIGMAS)
    finally:
        restore_default_pose_sigmas()


def test_apply_then_restore_returns_classes_to_original_state():
    from ultralytics.utils.loss import v8PoseLoss
    from ultralytics.models.yolo.pose.val import PoseValidator

    pre_loss_init = v8PoseLoss.__init__
    pre_val_init = PoseValidator.init_metrics

    apply_cattle_pose_sigmas()
    assert v8PoseLoss.__init__ is not pre_loss_init
    assert PoseValidator.init_metrics is not pre_val_init

    restore_default_pose_sigmas()
    assert v8PoseLoss.__init__ is pre_loss_init
    assert PoseValidator.init_metrics is pre_val_init


def test_apply_is_idempotent():
    """Calling apply twice should not stack wrappers in a way that loses the original."""
    from ultralytics.utils.loss import v8PoseLoss

    pre = v8PoseLoss.__init__
    try:
        apply_cattle_pose_sigmas()
        apply_cattle_pose_sigmas()
        restore_default_pose_sigmas()
        assert v8PoseLoss.__init__ is pre
    finally:
        # If anything went sideways above, force-restore to avoid bleeding state.
        if v8PoseLoss.__init__ is not pre:
            v8PoseLoss.__init__ = pre  # type: ignore[method-assign]


def test_apply_rejects_wrong_length_sigmas():
    with pytest.raises(ValueError, match="Expected"):
        apply_cattle_pose_sigmas(sigmas=(0.05, 0.05))  # only 2 instead of 9
