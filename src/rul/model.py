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
    CALIBRATION_FRACTION,
    CONFIDENCE_LEVEL,
    MODEL_FILENAME,
    MODELS_DIR,
    RANDOM_SEED,
    ROLLING_WINDOW,
    RUL_CAP,
    VALIDATION_FRACTION,
)
from .conformal import fit_split_conformal, intervals_from_mapie
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
    confidence_level: float = CONFIDENCE_LEVEL
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    feature_builder: FeatureBuilder = field(default_factory=FeatureBuilder)
    regressor: lgb.LGBMRegressor | None = None
    # MAPIE SplitConformalRegressor, calibrated on held-out engines (see fit).
    conformal: object | None = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, train_df: pd.DataFrame, subset: str = "", conformalize: bool = True) -> RULModel:
        """Train on a raw run-to-failure frame (RUL is derived internally).

        Engines are partitioned into a **fit pool** (for the point model) and a
        disjoint **calibration** set (for conformal intervals) so the coverage
        guarantee is not compromised by reusing training data. Pass
        ``conformalize=False`` to skip interval calibration (faster; tests).
        """
        labeled = compute_rul(train_df, cap=self.rul_cap)

        # 1) Reserve calibration engines, unseen by the point model.
        if conformalize:
            calib_splitter = GroupShuffleSplit(
                n_splits=1, test_size=CALIBRATION_FRACTION, random_state=RANDOM_SEED
            )
            pool_idx, calib_idx = next(
                calib_splitter.split(labeled, groups=labeled["unit"])
            )
            df_pool = labeled.iloc[pool_idx].reset_index(drop=True)
            df_calib = labeled.iloc[calib_idx].reset_index(drop=True)
        else:
            df_pool, df_calib = labeled, None

        # 2) Grouped split inside the pool for early stopping (no engine crosses).
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=VALIDATION_FRACTION, random_state=RANDOM_SEED
        )
        train_idx, val_idx = next(splitter.split(df_pool, groups=df_pool["unit"]))
        df_tr = df_pool.iloc[train_idx]
        df_va = df_pool.iloc[val_idx]

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

        # 3) Refit the point model on the whole fit pool at the best iteration.
        self.feature_builder = FeatureBuilder(window=self.window)
        X_pool, y_pool = make_xy(df_pool, self.feature_builder, fit=True)
        final_params = {**self.params, "n_estimators": best_iter}
        self.regressor = lgb.LGBMRegressor(**final_params)
        # Fit on ndarray (no column names): MAPIE feeds the estimator numpy
        # arrays internally, so training the same way avoids sklearn's
        # "X has no valid feature names" warning at conformalization time.
        self.regressor.fit(X_pool.to_numpy(), y_pool.to_numpy())

        # 4) Calibrate conformal intervals on the held-out calibration engines.
        self.conformal = None
        if conformalize and df_calib is not None:
            self.conformal = fit_split_conformal(
                self.regressor, self.feature_builder, df_calib, self.confidence_level
            )

        self.metadata = {
            "subset": subset,
            "rul_cap": self.rul_cap,
            "window": self.window,
            "confidence_level": self.confidence_level,
            "best_iteration": best_iter,
            "n_features": len(self.feature_builder.feature_names_),
            "kept_columns": list(self.feature_builder.kept_columns_),
            "n_train_rows": int(len(labeled)),
            "n_train_engines": int(labeled["unit"].nunique()),
            "n_pool_engines": int(df_pool["unit"].nunique()),
            "n_calib_engines": int(df_calib["unit"].nunique()) if df_calib is not None else 0,
            "conformalized": bool(self.conformal is not None),
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
        preds = self.regressor.predict(X.to_numpy())  # numpy: see fit() note
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

    def _check_conformal(self) -> None:
        if self.conformal is None:
            raise RuntimeError(
                "This model has no conformal calibrator; fit with conformalize=True."
            )

    def predict_interval(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(point, lower, upper)`` per row at ``confidence_level``."""
        self._check_fitted()
        self._check_conformal()
        X = self.feature_builder.transform(df)
        return intervals_from_mapie(self.conformal, X, float(self.rul_cap))

    def predict_interval_last_cycle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Point + interval per engine at its last cycle (index = unit)."""
        self._check_fitted()
        self._check_conformal()
        frame = df.reset_index(drop=True)
        point, lower, upper = self.predict_interval(frame)
        tagged = frame[["unit", "cycle"]].copy()
        tagged["point"] = point
        tagged["lower"] = lower
        tagged["upper"] = upper
        last = last_cycle_rows(tagged)
        return last.set_index("unit")[["point", "lower", "upper"]]

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
    def load(path: str | Path | None = None) -> RULModel:
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
