"""LightGBM RUL model: a self-contained, serializable predictor.

:class:`RULModel` bundles the fitted :class:`~rul.features.FeatureBuilder` with
a LightGBM regressor so that a *single* artifact maps raw C-MAPSS cycles to a
RUL prediction. Keeping the feature builder inside the artifact is what lets the
API and tests reproduce training-time features exactly.

Design notes
------------
* **Why LightGBM** — a strong, fast tabular baseline that handles the
  engineered features natively; a sensible reference before any sequence model.
* **Grouped early-stopping split** — the validation set is chosen by *engine*
  (``GroupShuffleSplit``) so no engine's cycles appear in both fit and
  validation. We then refit on *all* engines using the best iteration found.
* **No feature scaling** — tree ensembles are invariant to monotone transforms.
* **Prediction clipping** — RUL cannot be negative and the capped target means
  the model never learned values above ``rul_cap``; predictions are clipped to
  ``[0, rul_cap]``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import (
    MODEL_FILENAME,
    MODELS_DIR,
    RANDOM_SEED,
    ROLLING_WINDOW,
    RUL_CAP,
    VALIDATION_FRACTION,
)
from .data import compute_rul, last_cycle_rows
from .features import FeatureBuilder, make_xy

# LightGBM hyperparameters. Modest, reproducible defaults tuned for FD001's
# size; documented in the README. objective="regression" (L2) aligns the
# training loss with the reported RMSE.
DEFAULT_PARAMS: dict = {
    "objective": "regression",
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 75


@dataclass
class RULModel:
    """Feature builder + LightGBM regressor, trained and served as one unit."""

    rul_cap: int = RUL_CAP
    window: int = ROLLING_WINDOW
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    regressor: lgb.LGBMRegressor | None = None
    # Populated by rul.conformal in milestone 5 (kept here so the whole
    # predictor persists as one artifact).
    conformal: object | None = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, train_df: pd.DataFrame, subset: str = "") -> "RULModel":
        """Train on a raw run-to-failure frame (RUL is derived internally)."""
        labeled = compute_rul(train_df, cap=self.rul_cap)

        # 1) Grouped split for early stopping (no engine crosses the boundary).
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=VALIDATION_FRACTION, random_state=RANDOM_SEED
        )
        train_idx, val_idx = next(splitter.split(labeled, groups=labeled["unit"]))
        df_tr = labeled.iloc[train_idx]
        df_va = labeled.iloc[val_idx]

        tuning_builder = FeatureBuilder(window=self.window)
        X_tr, y_tr = make_xy(df_tr, tuning_builder, fit=True)
        X_va, y_va = make_xy(df_va, tuning_builder, fit=False)

        tuner = lgb.LGBMRegressor(**self.params)
        tuner.fit(
            X_tr,
            y_tr,
            eval_X=X_va,  # lightgbm>=4.7 replaced eval_set with eval_X/eval_y
            eval_y=y_va,
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        best_iter = int(tuner.best_iteration_ or self.params["n_estimators"])

        # 2) Refit on ALL engines with the discovered iteration count.
        self.feature_builder = FeatureBuilder(window=self.window)
        X_all, y_all = make_xy(labeled, self.feature_builder, fit=True)
        final_params = {**self.params, "n_estimators": best_iter}
        self.regressor = lgb.LGBMRegressor(**final_params)
        self.regressor.fit(X_all, y_all)

        self.metadata = {
            "subset": subset,
            "rul_cap": self.rul_cap,
            "window": self.window,
            "best_iteration": best_iter,
            "n_features": len(self.feature_builder.feature_names_),
            "kept_columns": list(self.feature_builder.kept_columns_),
            "n_train_rows": int(len(labeled)),
            "n_train_engines": int(labeled["unit"].nunique()),
        }
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self.regressor is None:
            raise RuntimeError("RULModel is not fitted; call fit() or load().")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Point RUL prediction per row of *df* (raw C-MAPSS columns)."""
        self._check_fitted()
        X = self.feature_builder.transform(df)
        preds = self.regressor.predict(X)
        return np.clip(preds, 0.0, float(self.rul_cap))

    def predict_last_cycle(self, df: pd.DataFrame) -> pd.Series:
        """One prediction per engine, at its last observed cycle.

        Rolling features are computed over each engine's full trajectory before
        the final cycle is selected — the standard C-MAPSS test protocol.
        """
        self._check_fitted()
        frame = df.reset_index(drop=True)
        preds = self.predict(frame)
        tagged = frame[["unit", "cycle"]].copy()
        tagged["pred"] = preds
        last = last_cycle_rows(tagged)
        return pd.Series(
            last["pred"].to_numpy(),
            index=pd.Index(last["unit"], name="unit"),
            name="pred",
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path | None = None) -> Path:
        """Persist the whole predictor with joblib. Returns the path."""
        self._check_fitted()
        path = Path(path) if path is not None else MODELS_DIR / MODEL_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path | None = None) -> "RULModel":
        path = Path(path) if path is not None else MODELS_DIR / MODEL_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No model artifact at {path}. Train one with `python -m rul.train`."
            )
        model = joblib.load(path)
        if not isinstance(model, RULModel):
            raise TypeError(f"{path} does not contain a RULModel.")
        return model

    def save_metadata(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
