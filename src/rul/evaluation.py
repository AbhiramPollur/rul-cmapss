"""Evaluation metrics for RUL prediction.

Two metrics are reported together on purpose:

* **RMSE** is symmetric and easy to interpret (cycles of error).
* **NASA score** (PHM08 challenge) is *asymmetric*: it penalizes **late**
  predictions — estimating more remaining life than the engine actually has,
  the unsafe direction — more heavily than early ones. A model can improve
  RMSE while getting worse on the metric that matters for safety, so we never
  look at one without the other.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

# Asymmetry constants from the original PHM08 scoring function.
_EARLY_SCALE = 13.0  # d < 0  (predicted too little life)
_LATE_SCALE = 10.0   # d >= 0 (predicted too much life -> punished harder)


def _as_1d(a: ArrayLike) -> np.ndarray:
    return np.asarray(a, dtype=float).ravel()


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error in cycles."""
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    return float(np.sqrt(np.mean((yp - yt) ** 2)))


def nasa_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """PHM08 asymmetric score (lower is better; 0 is perfect).

    ``d = y_pred - y_true``. For early predictions (``d < 0``) the penalty is
    ``exp(-d/13) - 1``; for late predictions (``d >= 0``) it is ``exp(d/10) - 1``.
    Summed over all engines.
    """
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: {yt.shape} vs {yp.shape}")
    d = yp - yt
    penalties = np.where(d < 0, np.expm1(-d / _EARLY_SCALE), np.expm1(d / _LATE_SCALE))
    return float(np.sum(penalties))


def regression_report(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Bundle the headline metrics into a JSON-serializable dict."""
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    return {
        "n": int(yt.size),
        "rmse": rmse(yt, yp),
        "nasa_score": nasa_score(yt, yp),
        "mean_abs_error": float(np.mean(np.abs(yp - yt))),
        "mean_error": float(np.mean(yp - yt)),  # +ve => predicts late on average
    }


def interval_report(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    confidence_level: float,
) -> dict[str, float]:
    """Empirical coverage and width of prediction intervals.

    ``coverage`` is the fraction of true values inside ``[lower, upper]``; it
    should be close to ``confidence_level`` if the conformal calibration holds
    on this data.
    """
    yt, lo, hi = _as_1d(y_true), _as_1d(lower), _as_1d(upper)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError("y_true, lower and upper must share a shape.")
    covered = (yt >= lo) & (yt <= hi)
    return {
        "confidence_level": float(confidence_level),
        "coverage": float(np.mean(covered)),
        "avg_interval_width": float(np.mean(hi - lo)),
        "median_interval_width": float(np.median(hi - lo)),
    }
