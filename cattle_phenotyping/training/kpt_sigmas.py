"""OKS sigmas for the 9 canonical cattle side-view keypoints.

Why this module exists
----------------------

Ultralytics 8.4.x's :class:`v8PoseLoss` and :class:`PoseValidator` default to
``sigma = np.ones(N) / N = 0.111`` for every keypoint when ``kpt_shape[0] != 17``
(i.e. anything that isn't COCO human pose). At that sigma the OKS loss gradient
is too soft to push the pose head toward precise anatomical localization — the
2026-05-21 run with these defaults reached training-time mAP50-95(P) = 0.953
**while** predicting ``_top`` keypoints at the bbox top edge (~400 px off) and
producing a forward-Schaeffer val MAE of 166 kg. The metric was numerically
correct but anatomically meaningless.

Tight per-keypoint sigmas fix both ends of that failure:

* **Loss gradient.** The pose loss inside :class:`v8PoseLoss` is
  ``loss = 1 - exp(-d² / (2·area·σ²))``. Smaller σ makes the loss much steeper
  for the same pixel error, so the optimizer is forced to push keypoints toward
  the GT instead of settling on the bbox-edge prior.
* **Metric meaning.** With σ matching the actual anatomical precision of each
  landmark, OKS-50/75/95 read as real "X% of images within Y% body-scale of
  GT" — they stop rewarding the degenerate solution.

The values below are calibrated by analogy to COCO's per-keypoint sigmas:

* COCO uses σ ≈ 0.025 for tight skeletal anchors (eye, ear, ankle), σ ≈ 0.05
  for less precise body landmarks (knee, elbow), σ ≈ 0.08-0.11 for fuzzy
  joints (hip, shoulder).
* Cattle anatomy: wither / pinbone / shoulderbone are skeletal protrusions
  (precise); ``*_girth_top`` and ``*_girth_bottom`` live on the body outline
  where the silhouette varies by a few cm (less precise); ``height_top`` is
  the top of the back and ``height_bottom`` is the hoof / ground line, both
  on outline curves rather than landmark points.

Apply once before ``model.train(...)`` via
:func:`apply_cattle_pose_sigmas`. The patch is idempotent and reversible
through :func:`restore_default_pose_sigmas`.

References
----------

* :data:`cattle_phenotyping.data.kaggle.CANONICAL_SIDE_KEYPOINTS` — the
  9-name order this sigma array aligns to.
* 2026-05-21 keypoint-training post-mortem: see [[progress-2026-05-21]].
"""

from __future__ import annotations

from typing import Callable

from cattle_phenotyping.data.kaggle import CANONICAL_SIDE_KEYPOINTS
from cattle_phenotyping.utils.log import get_logger

log = get_logger(__name__)


# Order MUST match CANONICAL_SIDE_KEYPOINTS. Indexed by name in the dict below
# so a reader can audit each value without counting positions.
_SIGMA_BY_NAME: dict[str, float] = {
    "wither":             0.025,  # top of shoulder, skeletal protrusion
    "pinbone":            0.025,  # hip pinbone, skeletal protrusion
    "shoulderbone":       0.025,  # front shoulder bone
    "front_girth_top":    0.040,  # body outline at front girth, top side
    "front_girth_bottom": 0.040,  # body outline at front girth, bottom side
    "rear_girth_top":     0.040,  # body outline at rear girth, top side
    "rear_girth_bottom":  0.040,  # body outline at rear girth, bottom side
    "height_top":         0.050,  # top-of-back curve for body height
    "height_bottom":      0.050,  # hoof / ground line for body height
}


def _build_sigma_array() -> tuple[float, ...]:
    """Materialize the sigma tuple in :data:`CANONICAL_SIDE_KEYPOINTS` order."""
    missing = [n for n in CANONICAL_SIDE_KEYPOINTS if n not in _SIGMA_BY_NAME]
    if missing:
        raise RuntimeError(
            f"kpt_sigmas missing entries for canonical keypoints: {missing}. "
            "Update _SIGMA_BY_NAME so its keys cover CANONICAL_SIDE_KEYPOINTS."
        )
    return tuple(_SIGMA_BY_NAME[n] for n in CANONICAL_SIDE_KEYPOINTS)


#: Per-keypoint OKS sigmas, ordered to match :data:`CANONICAL_SIDE_KEYPOINTS`.
CATTLE_KEYPOINT_SIGMAS: tuple[float, ...] = _build_sigma_array()


# Ultralytics patch ----------------------------------------------------------
#
# We touch two classes inside Ultralytics:
#
# * ``v8PoseLoss`` — sets ``self.sigmas`` (a torch tensor on the trainer's
#   device) inside ``__init__``. The trainer constructs one of these *after*
#   ``model.train(...)`` starts. We can't set it directly without subclassing,
#   so we monkey-patch ``__init__`` to overwrite ``self.sigmas`` after the
#   parent init runs.
# * ``PoseValidator`` — sets ``self.sigma`` (a numpy array) inside
#   ``init_metrics``. Same pattern: we patch ``init_metrics`` to overwrite it.
#
# Both patches save the original method on the class so :func:`restore_default_pose_sigmas`
# can put it back. Idempotency: re-applying replaces the previous wrapper but
# the saved original chain remains intact.

_ORIG_LOSS_ATTR = "_cattle_orig_init"
_ORIG_VAL_ATTR = "_cattle_orig_init_metrics"


def apply_cattle_pose_sigmas(
    sigmas: tuple[float, ...] | None = None,
    *,
    on_log: Callable[[str], None] | None = None,
) -> None:
    """Monkey-patch Ultralytics so cattle-specific sigmas are used everywhere.

    Call this once after ``import ultralytics`` (or after ``model = YOLO(...)``)
    and **before** ``model.train(...)``. Both the pose loss and the
    val-time OKS metric pick up the override.

    Args:
        sigmas: 9-element tuple, ordered to match
            :data:`CANONICAL_SIDE_KEYPOINTS`. Defaults to
            :data:`CATTLE_KEYPOINT_SIGMAS`.
        on_log: Optional callback to receive the human-readable status line
            (defaults to :mod:`logging`).

    Raises:
        ImportError: if Ultralytics isn't installed.
        ValueError: if ``len(sigmas) != 9``.
    """
    sigmas = tuple(sigmas) if sigmas is not None else CATTLE_KEYPOINT_SIGMAS
    if len(sigmas) != len(CANONICAL_SIDE_KEYPOINTS):
        raise ValueError(
            f"Expected {len(CANONICAL_SIDE_KEYPOINTS)} sigmas; got {len(sigmas)}."
        )

    try:
        import numpy as np
        import torch
        from ultralytics.utils.loss import v8PoseLoss
        from ultralytics.models.yolo.pose.val import PoseValidator
    except ImportError as exc:  # pragma: no cover — only fires when ultralytics not installed
        raise ImportError(
            "apply_cattle_pose_sigmas requires `ultralytics` and `torch` to be installed."
        ) from exc

    sigma_np = np.array(sigmas, dtype=np.float64)
    sigma_list = list(sigmas)

    # --- Patch v8PoseLoss.__init__ ---------------------------------------
    # Look up the saved original dynamically (via getattr on type(self)) so
    # the wrapper survives test-time stubbing of the class attribute and
    # supports subclassing without losing the override.
    if not hasattr(v8PoseLoss, _ORIG_LOSS_ATTR):
        setattr(v8PoseLoss, _ORIG_LOSS_ATTR, v8PoseLoss.__init__)

    def _patched_loss_init(self, model):  # type: ignore[no-untyped-def]
        original = getattr(type(self), _ORIG_LOSS_ATTR)
        original(self, model)
        device = getattr(self, "device", None)
        self.sigmas = torch.tensor(sigma_list, dtype=torch.float32, device=device)

    v8PoseLoss.__init__ = _patched_loss_init  # type: ignore[method-assign]

    # --- Patch PoseValidator.init_metrics --------------------------------
    if not hasattr(PoseValidator, _ORIG_VAL_ATTR):
        setattr(PoseValidator, _ORIG_VAL_ATTR, PoseValidator.init_metrics)

    def _patched_val_init(self, model):  # type: ignore[no-untyped-def]
        original = getattr(type(self), _ORIG_VAL_ATTR)
        original(self, model)
        self.sigma = sigma_np.copy()

    PoseValidator.init_metrics = _patched_val_init  # type: ignore[method-assign]

    msg = (
        f"Applied cattle pose sigmas to Ultralytics: "
        f"{dict(zip(CANONICAL_SIDE_KEYPOINTS, sigmas))}"
    )
    if on_log:
        on_log(msg)
    else:
        log.info(msg)


def restore_default_pose_sigmas() -> None:
    """Revert :func:`apply_cattle_pose_sigmas` — restore Ultralytics' defaults.

    Idempotent: safe to call without a prior ``apply_cattle_pose_sigmas``.
    Intended for tests; production training should not need to call this.
    """
    try:
        from ultralytics.utils.loss import v8PoseLoss
        from ultralytics.models.yolo.pose.val import PoseValidator
    except ImportError:  # pragma: no cover
        return

    if hasattr(v8PoseLoss, _ORIG_LOSS_ATTR):
        v8PoseLoss.__init__ = getattr(v8PoseLoss, _ORIG_LOSS_ATTR)  # type: ignore[method-assign]
        delattr(v8PoseLoss, _ORIG_LOSS_ATTR)

    if hasattr(PoseValidator, _ORIG_VAL_ATTR):
        PoseValidator.init_metrics = getattr(PoseValidator, _ORIG_VAL_ATTR)  # type: ignore[method-assign]
        delattr(PoseValidator, _ORIG_VAL_ATTR)
