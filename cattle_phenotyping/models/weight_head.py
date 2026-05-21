"""Learned weight head — XGBoost on top of predicted-keypoint features.

This sits one stage downstream of the trained pose head (and, later, the
trained segmenter). At inference time the chain is:

    image
      → trained pose head → {kp_name: (x, y, conf)} per image
      → (predicted) sticker mask → sticker pixel area
      → cattle_phenotyping.pipeline.weight_head_features.build_features(...)
      → WeightHead.predict(...) → estimated weight_kg

The model is a thin :class:`xgboost.XGBRegressor` wrapper that:

* persists ``feature_names`` so loaded models reject DataFrames with the
  wrong columns or order — there's no silent column-shuffle bug;
* exposes ``fit_with_validation`` which uses XGBoost's early-stopping on a
  val split (so the per-epoch ``results.csv``-equivalent isn't required);
* returns the same dict-of-metrics shape as
  :mod:`cattle_phenotyping.eval.baseline_schaeffer` so the test-set
  comparison table is symmetric across "Schaeffer-only" and "learned head"
  rows.

Why XGBoost and not a small MLP?

* The feature space is tabular (~17 cols), each feature is interpretable,
  and the dataset is small (~3000 train rows). Gradient-boosted trees
  generalize better than an MLP at this scale and need no
  feature-normalization / dropout / early-stopping plumbing.
* Per-feature importance plots are trivial to ship as a debug artifact.
* If/when the dataset grows >100k rows, swap in a small MLP — the
  ``fit``/``predict`` interface here is what
  :mod:`cattle_phenotyping.training.train_weight_head` calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cattle_phenotyping.pipeline.weight_head_features import (
    WEIGHT_HEAD_FEATURE_NAMES,
)
from cattle_phenotyping.utils.log import get_logger

log = get_logger(__name__)


# Default hyperparams chosen for the ~3000-row Kaggle train split. They
# match the conservative end of XGBoost defaults: shallow trees, mild
# learning rate, generous early-stopping budget. Tune on val.csv if
# warranted; never re-tune against test.csv.

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 800,        # ceiling — early stopping decides the real count
    "max_depth": 5,             # mild interactions; deeper overfits on N≈3000
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 4.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "verbosity": 0,
    "random_state": 42,
}

# Number of rounds without val-MAE improvement before early-stopping fires.
DEFAULT_EARLY_STOPPING_ROUNDS = 40


# Lightweight metric struct (mirrors :class:`baseline_schaeffer.Metrics` so
# downstream code can serialize it the same way).
@dataclass(frozen=True)
class WeightHeadMetrics:
    n: int
    mae_kg: float
    rmse_kg: float
    mape_pct: float
    r2: float
    bias_kg: float
    mean_true_kg: float


def _compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> WeightHeadMetrics:
    """Compute MAE / RMSE / MAPE / R² / bias on aligned sequences."""
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        raise ValueError(f"y_true and y_pred must be equal-length non-empty; got {n} and {len(y_pred)}")
    import math
    res = [yp - yt for yt, yp in zip(y_true, y_pred)]
    abs_err = [abs(e) for e in res]
    sq_err = [e * e for e in res]
    mean_true = sum(y_true) / n
    ss_res = sum(sq_err)
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pct_err = [abs(e) / yt * 100 for e, yt in zip(res, y_true) if yt > 0]
    return WeightHeadMetrics(
        n=n,
        mae_kg=sum(abs_err) / n,
        rmse_kg=math.sqrt(ss_res / n),
        mape_pct=sum(pct_err) / len(pct_err) if pct_err else float("nan"),
        r2=r2,
        bias_kg=sum(res) / n,
        mean_true_kg=mean_true,
    )


def metrics_to_dict(m: WeightHeadMetrics) -> dict[str, float | int]:
    """JSON-friendly dict for inclusion in run reports."""
    return {
        "n": m.n,
        "mae_kg": m.mae_kg,
        "rmse_kg": m.rmse_kg,
        "mape_pct": m.mape_pct,
        "r2": m.r2,
        "bias_kg": m.bias_kg,
        "mean_true_kg": m.mean_true_kg,
    }


# Model wrapper -------------------------------------------------------------


class WeightHead:
    """XGBoost regressor for cattle weight (kg) from tabular keypoint features.

    The ``feature_names`` it stores at fit time are checked on every
    :meth:`predict` call. Loading a model into a process whose
    :data:`WEIGHT_HEAD_FEATURE_NAMES` has drifted raises immediately rather
    than silently scoring against the wrong columns.
    """

    def __init__(
        self,
        params: Mapping[str, Any] | None = None,
        *,
        feature_names: Sequence[str] = WEIGHT_HEAD_FEATURE_NAMES,
    ) -> None:
        self.params: dict[str, Any] = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self._model: Any = None  # xgb.XGBRegressor instance after fit/load
        self.best_iteration: int | None = None

    # ------------------------------------------------------------ training

    def fit_with_validation(
        self,
        X_train,                # pandas.DataFrame
        y_train: Sequence[float],
        X_val,
        y_val: Sequence[float],
        *,
        early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    ) -> WeightHeadMetrics:
        """Fit on train, use val for early stopping. Returns val metrics."""
        self._require_columns(X_train, "X_train")
        self._require_columns(X_val, "X_val")

        # Imports are lazy so the rest of this module remains importable
        # in environments without xgboost (e.g. CI on Python-only test runs).
        import xgboost as xgb

        # XGBoost moved early stopping into the constructor in v2.0.
        # callbacks=[EarlyStopping(...)] keeps the API stable across the
        # 2.x line.
        callbacks = [xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name="mae",
            data_name="validation_0",
            save_best=True,
        )]
        self._model = xgb.XGBRegressor(
            **self.params,
            eval_metric="mae",
            callbacks=callbacks,
        )
        log.info(
            "Fitting WeightHead | train n=%d | val n=%d | params=%s",
            len(X_train), len(X_val),
            {k: v for k, v in self.params.items() if k not in {"objective", "tree_method", "verbosity"}},
        )
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self.best_iteration = int(getattr(self._model, "best_iteration", -1)) or None
        log.info("WeightHead fit done; best_iteration=%s", self.best_iteration)

        val_preds = self.predict(X_val)
        return _compute_metrics(list(y_val), list(val_preds))

    def fit(self, X_train, y_train) -> None:
        """Fit without early stopping — used by tests or when no val set is available."""
        self._require_columns(X_train, "X_train")
        import xgboost as xgb
        self._model = xgb.XGBRegressor(**self.params, eval_metric="mae")
        self._model.fit(X_train, y_train, verbose=False)
        self.best_iteration = int(getattr(self._model, "best_iteration", -1)) or None

    # ------------------------------------------------------------ inference

    def predict(self, X):
        """Predict weight (kg). Returns a 1-D numpy array."""
        if self._model is None:
            raise RuntimeError("WeightHead.predict called before fit/load")
        self._require_columns(X, "X")
        return self._model.predict(X)

    def evaluate(self, X, y_true: Sequence[float]) -> WeightHeadMetrics:
        """Convenience: predict + compute_metrics in one call."""
        preds = self.predict(X)
        return _compute_metrics(list(y_true), list(preds))

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> Path:
        """Persist model + feature schema as a two-file pair.

        Layout::

            <path>.json   — XGBoost native dump (portable across versions)
            <path>.meta.json — feature_names + best_iteration + params

        ``path`` may have any extension or none; the two output files are
        derived from its stem.
        """
        if self._model is None:
            raise RuntimeError("WeightHead.save called before fit")

        out = Path(path)
        model_path = out.with_suffix(".json")
        meta_path = out.with_suffix(".meta.json")
        model_path.parent.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(model_path))
        meta = {
            "feature_names": list(self.feature_names),
            "best_iteration": self.best_iteration,
            "params": self.params,
        }
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> "WeightHead":
        """Restore a WeightHead saved with :meth:`save`."""
        in_path = Path(path)
        model_path = in_path.with_suffix(".json")
        meta_path = in_path.with_suffix(".meta.json")
        if not model_path.exists():
            raise FileNotFoundError(f"WeightHead model file missing: {model_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"WeightHead meta file missing: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        instance = cls(params=meta.get("params"), feature_names=tuple(meta["feature_names"]))
        import xgboost as xgb
        instance._model = xgb.XGBRegressor()
        instance._model.load_model(str(model_path))
        instance.best_iteration = meta.get("best_iteration")
        return instance

    # ------------------------------------------------------------ internal

    def _require_columns(self, X, label: str) -> None:
        """Raise if X's columns don't match feature_names exactly (incl. order)."""
        cols = getattr(X, "columns", None)
        if cols is None:
            return  # arrays / non-DataFrame inputs skip the check
        got = tuple(cols)
        if got != self.feature_names:
            missing = [c for c in self.feature_names if c not in got]
            extra = [c for c in got if c not in self.feature_names]
            misordered = (
                missing == [] and extra == [] and got != self.feature_names
            )
            raise ValueError(
                f"WeightHead {label} columns don't match feature_names. "
                f"missing={missing} extra={extra} misordered={misordered}"
            )

    @property
    def feature_importance(self) -> dict[str, float] | None:
        """Per-feature importance ('gain') if the model has been fit."""
        if self._model is None:
            return None
        booster = self._model.get_booster()
        raw = booster.get_score(importance_type="gain")
        # XGBoost may name features 'f0', 'f1', ... if a DataFrame wasn't passed
        # at fit time. Our fit always passes a DataFrame, so the keys come back
        # as actual column names — but be defensive in case someone overrides.
        importance: dict[str, float] = {name: 0.0 for name in self.feature_names}
        for k, v in raw.items():
            if k in importance:
                importance[k] = float(v)
            elif k.startswith("f") and k[1:].isdigit():
                idx = int(k[1:])
                if 0 <= idx < len(self.feature_names):
                    importance[self.feature_names[idx]] = float(v)
        return importance
