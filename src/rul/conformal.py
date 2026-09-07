"""Split-conformal prediction intervals via MAPIE.

A point RUL is not enough for maintenance decisions — we need calibrated
*uncertainty*. This module wraps the trained LightGBM regressor in MAPIE's
:class:`SplitConformalRegressor` to produce intervals with a finite-sample,
distribution-free **marginal coverage** guarantee at ``confidence_level``.

How it works (split / inductive conformal):

1. The point model is trained on a *fit pool* of engines (see
   :meth:`rul.model.RULModel.fit`).
2. A disjoint set of **calibration engines** — never seen by the regressor —
   is passed here. MAPIE computes absolute conformity scores ``|y - ŷ|`` on it.
3. The ``(1 - α)`` empirical quantile of those scores becomes the half-width:
   an interval ``[ŷ - q, ŷ + q]``.

**Exchangeability caveat.** Conformal guarantees assume the calibration and
test points are exchangeable. Here the calibration points are cycles drawn from
full training trajectories, whereas each *test* query is a single truncated
snapshot per engine, and degradation series are autocorrelated. Coverage is
therefore *approximate* on the C-MAPSS test protocol — we measure and report the
empirical coverage rather than trusting the nominal level blindly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor

from .config import CONFIDENCE_LEVEL, RUL_COL
from .features import FeatureBuilder


def fit_split_conformal(
    regressor,
    feature_builder: FeatureBuilder,
    calib_labeled: pd.DataFrame,
    confidence_level: float = CONFIDENCE_LEVEL,
    target_col: str = RUL_COL,
) -> SplitConformalRegressor:
    """Conformalize an already-fitted regressor on held-out calibration data.

    Parameters
    ----------
    regressor:
        A fitted, sklearn-compatible point regressor (LightGBM here).
    feature_builder:
        The *same* fitted builder used to train ``regressor``.
    calib_labeled:
        Calibration rows (raw C-MAPSS columns + a RUL target column) from
        engines the regressor never saw.
    """
    if target_col not in calib_labeled.columns:
        raise ValueError(f"Calibration frame must contain a {target_col!r} column.")
    X_calib = feature_builder.transform(calib_labeled).to_numpy()
    y_calib = calib_labeled[target_col].to_numpy(dtype=float)

    conformal = SplitConformalRegressor(
        estimator=regressor,
        confidence_level=confidence_level,
        conformity_score="absolute",  # symmetric |y - ŷ| residual bands
        prefit=True,  # regressor is already trained; only calibrate here
    )
    conformal.conformalize(X_calib, y_calib)
    return conformal


def intervals_from_mapie(
    conformal: SplitConformalRegressor,
    X: pd.DataFrame,
    rul_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(point, lower, upper)`` arrays with physically-valid bounds.

    Clipping rules that *preserve* coverage:

    * point   -> clipped to ``[0, rul_cap]`` (the model's valid output range);
    * lower   -> clipped up to ``0`` (RUL can't be negative, and true >= 0 so
      raising a negative bound to 0 never drops a covered point);
    * upper   -> left unclipped (clamping it at the cap would shrink intervals
      and break coverage for engines whose true RUL exceeds the cap).
    """
    point, iv = conformal.predict_interval(np.asarray(X))
    point = np.asarray(point, dtype=float)
    lower = np.asarray(iv[:, 0, 0], dtype=float)
    upper = np.asarray(iv[:, 1, 0], dtype=float)

    point_c = np.clip(point, 0.0, rul_cap)
    lower_c = np.clip(lower, 0.0, None)
    upper_c = np.maximum(upper, point_c)  # keep lower <= point <= upper
    lower_c = np.minimum(lower_c, point_c)
    return point_c, lower_c, upper_c
