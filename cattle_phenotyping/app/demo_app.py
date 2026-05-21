"""Demo app for the trained cattle-phenotyping stack.

User flow (single page):
1. Upload a side-view cattle image.
2. Trained YOLOv8s-pose head detects the cow + 9 anatomical keypoints.
3. User clicks once on the sticker.
4. SAM (ViT-B) segments the sticker from that point → pixel area.
5. Per-batch sticker cm² + predicted keypoints → feature vector.
6. Trained XGBoost ``WeightHead`` predicts body weight (kg).

Each stage's output is shown so the demo audience sees what the model is
actually doing — not just a number.

Run::

    streamlit run cattle_phenotyping/app/demo_app.py

Required artifacts (paths shown are the defaults; override via the sidebar):

* ``weights/pose/best.pt``                       — trained YOLOv8s-pose
* ``weights/weight_head.json``                   — XGBoost dump
* ``weights/weight_head.meta.json``              — feature schema + best_iteration
* ``sam_vit_b_01ec64.pth``                       — SAM ViT-B checkpoint
* ``data/calibration/sticker_area_cm2_by_batch.json``

Notes for the demo:
* The "batch" selector tells the head which sticker cm² to use. In a real
  deployment we'd also classify the batch automatically; for the demo we
  let the user pick because both B3 and B4 are present in the training set.
* When ``streamlit-image-coordinates`` is installed, sticker click is
  point-and-shoot. When it isn't, manual x/y number inputs are used as a
  fallback so the demo still works on any plain Streamlit install.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from cattle_phenotyping.data.kaggle import CANONICAL_SIDE_KEYPOINTS
from cattle_phenotyping.models.schaeffer import schaeffer_from_keypoints
from cattle_phenotyping.models.segmenter_sam import CowSegmenter
from cattle_phenotyping.models.weight_head import WeightHead
from cattle_phenotyping.pipeline.weight_head_features import (
    FeatureSkip,
    WEIGHT_HEAD_FEATURE_NAMES,
    build_features,
)
from cattle_phenotyping.utils.log import setup_logging

setup_logging(level="INFO")


# Optional click-on-image input. Falls back to number inputs if missing.
try:
    from streamlit_image_coordinates import streamlit_image_coordinates  # type: ignore[import-not-found]
    HAS_CLICK_INPUT = True
except ImportError:  # pragma: no cover
    HAS_CLICK_INPUT = False


# Lightweight KaggleSample stand-in for feature building from an upload.
# The feature builder only reads ``batch``, ``image_width``, and
# ``image_height`` from the sample, so we can avoid the full COCO scaffolding.
from dataclasses import dataclass


@dataclass
class _DemoSample:
    """Minimal sample object for feature building from an uploaded image."""
    batch: str
    image_width: int
    image_height: int


# ── Page config ────────────────────────────────────────────────────────────


st.set_page_config(page_title="Cattle Phenotyping — Demo", page_icon="🐄", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background: #f0f7f0;
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 18px 22px;
        margin: 6px 0;
    }
    .metric-card .label { font-size: 13px; color: #555; text-transform: uppercase; letter-spacing: .05em; }
    .metric-card .value { font-size: 36px; font-weight: 700; color: #1b5e20; line-height: 1.1; }
    .metric-card .sub   { font-size: 12px; color: #777; margin-top: 4px; }
    .stage {
        background: #fafafa; border: 1px solid #e0e0e0; border-radius: 6px;
        padding: 8px 12px; margin: 4px 0; font-size: 13px;
    }
    .stage.ok { border-left: 4px solid #43a047; }
    .stage.warn { border-left: 4px solid #fb8c00; }
    .stage.err { border-left: 4px solid #e53935; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: paths + batch ────────────────────────────────────────────────


with st.sidebar:
    st.header("Models")
    pose_weights_default = "weights/pose/best.pt"
    weight_head_default = "weights/weight_head"
    sam_ckpt_default = "sam_vit_b_01ec64.pth"
    sticker_cm2_default = "data/calibration/sticker_area_cm2_by_batch.json"

    pose_weights_path = st.text_input("Pose weights (best.pt)", pose_weights_default)
    weight_head_path = st.text_input("Weight head stem (no extension)", weight_head_default)
    sam_ckpt_path = st.text_input("SAM ViT-B checkpoint", sam_ckpt_default)
    sticker_cm2_path = st.text_input("Per-batch sticker cm² JSON", sticker_cm2_default)

    st.divider()
    st.header("Inference")
    batch_choice = st.selectbox(
        "Batch (sticker calibration)",
        options=("B3", "B4"),
        index=1,  # default B4 — larger sticker, easier to click and segment
        help="Determines which sticker cm² and one-hot the weight head uses. "
             "B3 ≈ 15 cm² (smaller sticker), B4 ≈ 79 cm² (larger sticker).",
    )
    pose_conf = st.slider("Pose confidence threshold", 0.0, 0.9, 0.25, 0.05)
    sticker_max_area_pct = st.slider(
        "SAM sticker cap (% of image area)", 0.5, 10.0, 5.0, 0.5,
        help="Reject SAM masks larger than this — prevents 'segmented the whole cow' when the click misses.",
    )


# ── Cached model loaders ──────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading YOLOv8 pose model...")
def load_pose(path: str):
    from ultralytics import YOLO
    if not Path(path).exists():
        raise FileNotFoundError(f"Pose weights not found at {path}")
    return YOLO(path)


@st.cache_resource(show_spinner="Loading SAM ViT-B (~375 MB)...")
def load_sam(path: str):
    return CowSegmenter(checkpoint_path=path, allow_auto_download=False)


@st.cache_resource(show_spinner="Loading weight head...")
def load_weight_head(stem: str) -> WeightHead:
    return WeightHead.load(stem)


@st.cache_data
def load_sticker_cm2(path: str) -> dict[str, float]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ── Header + load gate ────────────────────────────────────────────────────


st.title("🐄 Cattle Phenotyping — Live Demo")
st.markdown(
    "Trained on the [Kaggle BMGF cattle-weight dataset](https://www.kaggle.com/datasets/sadhliroomyprime/cattle-weight-detection-model-dataset-12k). "
    "Test-set MAE: **24.47 kg** (vs Schaeffer baseline 30.25 kg)."
)

load_errors: list[str] = []
pose = sam_seg = weight_head = sticker_cm2_by_batch = None
try:
    pose = load_pose(pose_weights_path)
except Exception as e:
    load_errors.append(f"Pose: {e}")
try:
    sam_seg = load_sam(sam_ckpt_path)
except Exception as e:
    load_errors.append(f"SAM: {e}")
try:
    weight_head = load_weight_head(weight_head_path)
except Exception as e:
    load_errors.append(f"WeightHead: {e}")
try:
    sticker_cm2_by_batch = load_sticker_cm2(sticker_cm2_path)
except Exception as e:
    load_errors.append(f"Sticker cm² JSON: {e}")

if load_errors:
    st.error("**Model loading failed.** Fix the sidebar paths and refresh:\n\n" +
             "\n".join(f"- {e}" for e in load_errors))
    st.stop()

assert pose is not None and sam_seg is not None and weight_head is not None and sticker_cm2_by_batch is not None

with st.sidebar:
    st.divider()
    st.markdown("**Loaded:** ✓ pose ✓ SAM ✓ weight head")
    st.caption(f"Pose classes: {pose.names}")
    st.caption(f"Pose kpts: {pose.model.model[-1].kpt_shape}")
    st.caption(f"WeightHead best_iter: {weight_head.best_iteration}")
    st.caption(f"Sticker cm² (per batch): {sticker_cm2_by_batch}")


# ── Upload ────────────────────────────────────────────────────────────────


uploaded = st.file_uploader(
    "Upload a side-view cattle image",
    type=("jpg", "jpeg", "png", "bmp", "webp"),
)
if uploaded is None:
    st.info("👆 Upload an image to begin.")
    st.stop()

# Decode upload to numpy. Keep both BGR (for cv2) and RGB (for display).
file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
if image_bgr is None:
    st.error("Could not decode this image.")
    st.stop()
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
H, W = image_rgb.shape[:2]


# ── Stage 1: pose ─────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Running keypoint detector...")
def run_pose(image_rgb_bytes: bytes, conf: float, model_path: str):
    """Cached so repeated reruns (e.g. after a new click) don't repredict."""
    # We hash on `image_rgb_bytes` so the cache keys on image content.
    # Re-decode into the cached function so caching works.
    arr = np.frombuffer(image_rgb_bytes, dtype=np.uint8).reshape((H, W, 3))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    pose_local = load_pose(model_path)
    result = pose_local.predict(
        source=bgr, imgsz=640, conf=conf, device=None,
        save=False, save_txt=False, verbose=False,
    )[0]
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None
    best_idx = int(result.boxes.conf.argmax())
    kxy = result.keypoints.xy[best_idx].cpu().numpy()
    kconf = result.keypoints.conf[best_idx].cpu().numpy()
    box = result.boxes.xyxy[best_idx].cpu().numpy()
    box_conf = float(result.boxes.conf[best_idx])
    return {
        "keypoints": {
            kp_name: (float(kxy[i, 0]), float(kxy[i, 1]), float(kconf[i]))
            for i, kp_name in enumerate(CANONICAL_SIDE_KEYPOINTS)
        },
        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
        "bbox_conf": box_conf,
    }


pose_out = run_pose(image_rgb.tobytes(), pose_conf, pose_weights_path)
if pose_out is None:
    st.markdown('<div class="stage err">Stage 1 — pose: no cow detected. Try a clearer side-view image or lower the confidence threshold.</div>', unsafe_allow_html=True)
    st.image(image_rgb, caption=uploaded.name, use_container_width=True)
    st.stop()
st.markdown(
    f'<div class="stage ok">Stage 1 — pose: cow detected at conf {pose_out["bbox_conf"]:.2f}, '
    f'{sum(1 for v in pose_out["keypoints"].values() if v[2] > 0.5)}/9 keypoints above conf 0.5</div>',
    unsafe_allow_html=True,
)


# Overlay keypoints + bbox on a copy of the image for display.
overlay = image_rgb.copy()
box = pose_out["bbox"]
cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 0, 0), 3)
KP_COLORS = {
    "wither": (255, 0, 0), "pinbone": (255, 0, 0), "shoulderbone": (255, 0, 0),
    "front_girth_top": (0, 200, 0), "front_girth_bottom": (0, 200, 0),
    "rear_girth_top": (0, 200, 0), "rear_girth_bottom": (0, 200, 0),
    "height_top": (0, 0, 200), "height_bottom": (0, 0, 200),
}
for name, (x, y, c) in pose_out["keypoints"].items():
    if c < 0.3:
        continue
    cv2.circle(overlay, (int(x), int(y)), max(4, W // 600), KP_COLORS.get(name, (255, 200, 0)), -1)


# ── Stage 2: sticker click + SAM ─────────────────────────────────────────


st.divider()
st.subheader("👆 Click the sticker")

left_col, right_col = st.columns([1, 1], gap="medium")

with left_col:
    if HAS_CLICK_INPUT:
        # Show overlay with keypoints; user clicks on the sticker.
        # The component scales the displayed image; coords come back in
        # display-pixel space, so we rescale below.
        display_max_dim = 900
        scale = min(1.0, display_max_dim / max(H, W))
        disp_w, disp_h = int(W * scale), int(H * scale)
        pil = Image.fromarray(overlay).resize((disp_w, disp_h))
        click = streamlit_image_coordinates(pil, key="sticker_click")
        st.caption("Tip: click the centre of the sticker. Red=spine • Green=girth • Blue=height keypoints.")
        if click is None:
            st.info("Waiting for click...")
            st.stop()
        click_x = int(click["x"] / scale)
        click_y = int(click["y"] / scale)
    else:
        st.warning(
            "`streamlit-image-coordinates` isn't installed — using manual coordinate input. "
            "Install with `pip install streamlit-image-coordinates` for click-to-segment."
        )
        st.image(overlay, caption="Keypoints + bbox overlay (use this to read sticker coords)", use_container_width=True)
        cc1, cc2 = st.columns(2)
        click_x = cc1.number_input("Sticker x (px)", 0, W - 1, W // 2)
        click_y = cc2.number_input("Sticker y (px)", 0, H - 1, H // 2)
        if not st.button("Segment sticker at this point"):
            st.stop()

with right_col:
    with st.spinner("SAM segmenting sticker..."):
        sticker_mask = sam_seg.segment_from_point(
            image_bgr, (click_x, click_y),
            max_area_fraction=sticker_max_area_pct / 100.0,
        )
    sticker_area_px = int((sticker_mask > 0).sum())
    if sticker_area_px == 0:
        st.markdown(
            '<div class="stage err">Stage 2 — SAM: no mask under the size cap. '
            'Click more precisely on the sticker, or raise the SAM cap in the sidebar.</div>',
            unsafe_allow_html=True,
        )
        st.image(image_rgb, caption=f"Click was at ({click_x}, {click_y})", use_container_width=True)
        st.stop()
    st.markdown(
        f'<div class="stage ok">Stage 2 — SAM: sticker segmented, area = {sticker_area_px:,} px '
        f'({100*sticker_area_px/(H*W):.2f}% of image)</div>', unsafe_allow_html=True,
    )

    # Render the sticker mask as a red translucent overlay.
    mask_vis = overlay.copy()
    red = np.zeros_like(mask_vis); red[..., 0] = 255
    alpha = (sticker_mask > 0).astype(np.float32)[..., None] * 0.6
    mask_vis = (mask_vis * (1 - alpha) + red * alpha).astype(np.uint8)
    cv2.drawMarker(mask_vis, (click_x, click_y), (255, 255, 0), cv2.MARKER_CROSS, 30, 3)
    st.image(mask_vis, caption=f"Sticker mask overlay + click marker", use_container_width=True)


# ── Stage 3: features + weight head ──────────────────────────────────────


st.divider()
st.subheader("🧮 Weight prediction")

sample = _DemoSample(batch=batch_choice, image_width=W, image_height=H)
feat_result = build_features(
    sample, pose_out["keypoints"],
    sticker_area_px=sticker_area_px,
    sticker_cm2_by_batch=sticker_cm2_by_batch,
)
if isinstance(feat_result, FeatureSkip):
    st.markdown(
        f'<div class="stage err">Stage 3 — features: {feat_result.reason}. '
        'Try a clearer side-view image where all 9 keypoints are visible.</div>',
        unsafe_allow_html=True,
    )
    st.stop()
st.markdown(
    f'<div class="stage ok">Stage 3 — features: built {len(feat_result)} features, '
    f'Schaeffer prior = {feat_result["schaeffer_kg"]:.1f} kg</div>', unsafe_allow_html=True,
)

import pandas as pd
X = pd.DataFrame([feat_result], columns=list(WEIGHT_HEAD_FEATURE_NAMES))
pred_kg = float(weight_head.predict(X)[0])

# Recompute Schaeffer manually from the same features for side-by-side display.
px_per_cm = feat_result["px_per_cm"]
schaeffer_kg = feat_result["schaeffer_kg"]

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Predicted weight</div>
        <div class="value">{pred_kg:.1f} kg</div>
        <div class="sub">{pred_kg * 2.205:.0f} lb · test MAE ≈ 24.5 kg</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:#888;background:#fafafa;">
        <div class="label">Schaeffer baseline</div>
        <div class="value" style="color:#555;">{schaeffer_kg:.1f} kg</div>
        <div class="sub">HG²·BL/300 with the same keypoints</div>
    </div>""", unsafe_allow_html=True)
with m3:
    diff = pred_kg - schaeffer_kg
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:#1976d2;">
        <div class="label">Learned correction</div>
        <div class="value" style="color:#0d47a1;">{diff:+.1f} kg</div>
        <div class="sub">Learned head − Schaeffer</div>
    </div>""", unsafe_allow_html=True)

st.divider()
st.subheader("📐 Derived body measurements (cm)")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Body length", f"{feat_result['body_length_cm']:.1f} cm")
b2.metric("Front girth chord", f"{feat_result['front_girth_chord_cm']:.1f} cm")
b3.metric("Rear girth chord", f"{feat_result['rear_girth_chord_cm']:.1f} cm")
b4.metric("Body height", f"{feat_result['body_height_cm']:.1f} cm")
b5, b6, b7, b8 = st.columns(4)
b5.metric("px per cm", f"{px_per_cm:.2f}")
b6.metric("Sticker area", f"{sticker_area_px:,} px²")
b7.metric("Sticker cm²", f"{feat_result['sticker_cm2']:.2f}")
b8.metric("Keypoint mean conf", f"{feat_result['kp_conf_mean']:.2f}")

with st.expander("📄 Full feature vector + raw prediction JSON"):
    st.json({
        "click_xy": [click_x, click_y],
        "batch": batch_choice,
        "predicted_weight_kg": pred_kg,
        "schaeffer_kg": schaeffer_kg,
        "features": feat_result,
        "pose": {
            "bbox": pose_out["bbox"],
            "bbox_conf": pose_out["bbox_conf"],
            "keypoints": {k: {"x": v[0], "y": v[1], "conf": v[2]} for k, v in pose_out["keypoints"].items()},
        },
    })
