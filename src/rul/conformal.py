"""Split-conformal prediction intervals via MAPIE.

A point RUL is not enough for a maintenance decision, so every prediction comes
with an interval. This wraps the fitted LightGBM regressor in MAPIE's
:class:`SplitConformalRegressor`, which turns the model's own errors on a
held-out calibration set into a band with a distribution-free coverage
guarantee at ``confidence_level``.

The calibration set matters. Conformal coverage holds when the calibration
points look like the query points. Our queries are single mid-life snapshots
(one truncated cycle per test engine), so we calibrate on mid-life snapshots
too: :func:`operational_snapshot_indices` samples a spread of cycles per
calibration engine across the operating range rather than using every cycle,
which would over-represent the flat early-life region and make the intervals too
narrow. Because the calibration engines are held out from the point model, the
guarantee is not compromised by reusing training data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor

from .config import CONFIDENCE_LEVEL, RANDOM_SEED, RUL_COL


def operational_snapshot_indices(
    labeled_df: pd.DataFrame,
    per_engine: int = 12,
    rul_min: float = 5.0,
    rul_max: float = 150.0,
    seed: int = RANDOM_SEED,
    target_col: str = RUL_COL,
) -> list[int]:
    """Positions of representative mid-life cycles, spread over the RUL range.

    For each engine we draw ``per_engine`` target RUL values uniformly in
    ``[rul_min, rul_max]`` and take the cycle closest to each, so the sample
    mimics how the test set truncates engines rather than following the cycle
    distribution (which is dominated by the healthy plateau).
    """
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for _unit, group in labeled_df.groupby("unit", sort=False):
        rul = group[target_col].to_numpy(dtype=float)
        positions = group.index.to_numpy()
        hi = min(rul_max, float(rul.max()))
        if hi <= rul_min:
            chosen.extend(int(p) for p in positions)
            continue
        for target in rng.uniform(rul_min, hi, size=per_engine):
            chosen.append(int(positions[int(np.argmin(np.abs(rul - target)))]))
    # De-duplicate while preserving order.
    seen: set[int] = set()
    return [i for i in chosen if not (i in seen or seen.add(i))]


def fit_split_conformal(
    regressor,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> SplitConformalRegressor:
    """Calibrate an already-fitted regressor on prepared calibration arrays.

    ``X_calib`` is the feature matrix for the calibration snapshots and
    ``y_calib`` their true RUL. The regressor is passed with ``prefit=True`` so
    MAPIE only measures conformity scores, it does not retrain.
    """
    conformal = SplitConformalRegressor(
        estimator=regressor,
        confidence_level=confidence_level,
        conformity_score="absolute",  # symmetric |y - yhat| residual bands
        prefit=True,
    )
    conformal.conformalize(np.asarray(X_calib), np.asarray(y_calib, dtype=float))
    return conformal


def intervals_from_mapie(
    conformal: SplitConformalRegressor,
    X: pd.DataFrame,
    rul_cap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(point, lower, upper)`` arrays with physically valid bounds.

    The clipping keeps coverage intact: the point estimate is clipped to
    ``[0, rul_cap]`` (the model's valid range), the lower bound is raised to 0
    (RUL is never negative, and since true RUL is also non-negative this cannot
    drop a covered point), and the upper bound is left alone so engines with a
    lot of life left are still covered.
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
