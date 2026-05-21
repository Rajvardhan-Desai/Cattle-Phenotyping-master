"""Tests for :class:`cattle_phenotyping.models.weight_head.WeightHead`.

These exercise the wrapper's contract (schema validation, save/load roundtrip,
metrics shape) rather than XGBoost's own learning behavior — that's covered
by upstream tests in the xgboost package.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

# xgboost is a hard dep; skip the whole file if it's missing in the env.
pytest.importorskip("xgboost")
pd = pytest.importorskip("pandas")

from cattle_phenotyping.models.weight_head import (  # noqa: E402
    DEFAULT_PARAMS,
    WeightHead,
    WeightHeadMetrics,
    _compute_metrics,
    metrics_to_dict,
)
from cattle_phenotyping.pipeline.weight_head_features import (  # noqa: E402
    WEIGHT_HEAD_FEATURE_NAMES,
)


# Synthetic data ------------------------------------------------------------
#
# Build a small synthetic dataset where the target is a known linear
# combination of two features + noise. XGBoost should easily learn this and
# the fit/evaluate path should report finite, reasonable metrics.


def _synth_dataset(n_train: int = 200, n_val: int = 80, seed: int = 0):
    rng = random.Random(seed)

    rows = []
    targets = []
    for _ in range(n_train + n_val):
        # Realistic-ish ranges for our features.
        chord_cm = rng.uniform(80, 200)
        length_cm = rng.uniform(80, 200)
        height_cm = rng.uniform(60, 130)
        sticker_area_px = rng.uniform(200, 4000)
        sticker_cm2 = rng.choice([15.27, 79.33])
        px_per_cm = math.sqrt(sticker_area_px / sticker_cm2)
        chord_px = chord_cm * px_per_cm
        length_px = length_cm * px_per_cm
        # Schaeffer-like target with noise — model should learn the shape.
        schaeffer_kg = chord_cm * chord_cm * length_cm * math.pi * math.pi * 1.27e-5
        true_kg = 0.6 * schaeffer_kg + 0.35 * chord_cm + rng.gauss(0, 5)
        batch = "B3" if sticker_cm2 < 50 else "B4"
        rows.append({
            "schaeffer_kg": schaeffer_kg,
            "front_girth_chord_cm": chord_cm,
            "rear_girth_chord_cm": chord_cm * rng.uniform(0.95, 1.05),
            "body_length_cm": length_cm,
            "body_height_cm": height_cm,
            "front_girth_chord_px": chord_px,
            "body_length_px": length_px,
            "sticker_area_px": sticker_area_px,
            "sticker_cm2": sticker_cm2,
            "px_per_cm": px_per_cm,
            "front_to_rear_girth_ratio_cm": rng.uniform(0.9, 1.1),
            "girth_to_length_ratio_cm": chord_cm / length_cm,
            "length_to_height_ratio_cm": length_cm / height_cm,
            "kp_conf_mean": rng.uniform(0.5, 0.95),
            "kp_conf_min": rng.uniform(0.2, 0.7),
            "batch_B3": 1.0 if batch == "B3" else 0.0,
            "batch_B4": 1.0 if batch == "B4" else 0.0,
        })
        targets.append(true_kg)

    df = pd.DataFrame(rows, columns=list(WEIGHT_HEAD_FEATURE_NAMES))
    return df.iloc[:n_train].reset_index(drop=True), targets[:n_train], \
        df.iloc[n_train:].reset_index(drop=True), targets[n_train:]


# Metric layer --------------------------------------------------------------


def test_compute_metrics_basic():
    m = _compute_metrics([100.0, 200.0, 150.0], [110.0, 190.0, 160.0])
    assert m.n == 3
    assert abs(m.mae_kg - 10.0) < 1e-9
    assert abs(m.bias_kg - (10 + -10 + 10) / 3) < 1e-9
    assert math.isfinite(m.r2)
    assert math.isfinite(m.rmse_kg)


def test_compute_metrics_empty_raises():
    with pytest.raises(ValueError):
        _compute_metrics([], [])


def test_metrics_to_dict_keys():
    m = _compute_metrics([100.0, 200.0], [105.0, 195.0])
    d = metrics_to_dict(m)
    assert set(d.keys()) == {"n", "mae_kg", "rmse_kg", "mape_pct", "r2", "bias_kg", "mean_true_kg"}


# Schema enforcement --------------------------------------------------------


def test_predict_before_fit_raises():
    head = WeightHead()
    with pytest.raises(RuntimeError, match="before fit"):
        head.predict(pd.DataFrame())


def test_save_before_fit_raises():
    head = WeightHead()
    with pytest.raises(RuntimeError, match="before fit"):
        head.save("/tmp/never.json")


def test_predict_rejects_misordered_columns(tmp_path: Path):
    X_train, y_train, X_val, y_val = _synth_dataset(n_train=80, n_val=20)
    head = WeightHead(params={"n_estimators": 50})
    head.fit_with_validation(X_train, y_train, X_val, y_val)

    # Permute the columns of X_val — predict() should catch it.
    shuffled = X_val[list(reversed(WEIGHT_HEAD_FEATURE_NAMES))]
    with pytest.raises(ValueError, match="columns don't match"):
        head.predict(shuffled)


def test_predict_rejects_missing_column(tmp_path: Path):
    X_train, y_train, X_val, y_val = _synth_dataset(n_train=80, n_val=20)
    head = WeightHead(params={"n_estimators": 50})
    head.fit_with_validation(X_train, y_train, X_val, y_val)

    truncated = X_val.drop(columns=["schaeffer_kg"])
    with pytest.raises(ValueError, match="missing"):
        head.predict(truncated)


# Train / predict / persistence -------------------------------------------


def test_fit_with_validation_returns_finite_metrics():
    X_train, y_train, X_val, y_val = _synth_dataset()
    head = WeightHead(params={"n_estimators": 100})
    metrics = head.fit_with_validation(X_train, y_train, X_val, y_val)
    assert isinstance(metrics, WeightHeadMetrics)
    assert metrics.n == len(y_val)
    assert math.isfinite(metrics.mae_kg)
    assert math.isfinite(metrics.rmse_kg)
    # Model should fit synthetic linear-ish data well — MAE under 20% of mean.
    assert metrics.mae_kg < 0.20 * metrics.mean_true_kg


def test_save_load_roundtrip_preserves_predictions(tmp_path: Path):
    X_train, y_train, X_val, y_val = _synth_dataset()
    head = WeightHead(params={"n_estimators": 100})
    head.fit_with_validation(X_train, y_train, X_val, y_val)

    preds_before = head.predict(X_val)
    saved = head.save(tmp_path / "weight_head")
    assert saved.exists()
    assert (tmp_path / "weight_head.meta.json").exists()

    restored = WeightHead.load(tmp_path / "weight_head")
    preds_after = restored.predict(X_val)

    # XGBoost JSON round-trip is bitwise-stable.
    for a, b in zip(preds_before, preds_after):
        assert abs(float(a) - float(b)) < 1e-6


def test_load_with_missing_files_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        WeightHead.load(tmp_path / "does_not_exist")


def test_feature_importance_keys_match_schema():
    X_train, y_train, X_val, y_val = _synth_dataset()
    head = WeightHead(params={"n_estimators": 50})
    head.fit_with_validation(X_train, y_train, X_val, y_val)
    imp = head.feature_importance
    assert imp is not None
    assert set(imp.keys()) == set(WEIGHT_HEAD_FEATURE_NAMES)
    # Schaeffer should be one of the most important features given how the
    # synthetic target is built.
    assert imp["schaeffer_kg"] > 0


def test_default_params_are_inherited_then_overridable():
    head = WeightHead(params={"max_depth": 9, "learning_rate": 0.1})
    assert head.params["max_depth"] == 9
    assert head.params["learning_rate"] == 0.1
    # Untouched defaults survive.
    assert head.params["objective"] == DEFAULT_PARAMS["objective"]
