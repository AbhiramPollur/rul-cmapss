"""Tests for rul.evaluation: RMSE and the NASA asymmetric score."""
from __future__ import annotations

import math

import numpy as np
import pytest

from rul.evaluation import nasa_score, regression_report, rmse


def test_perfect_prediction_is_zero():
    y = [10.0, 50.0, 125.0]
    assert rmse(y, y) == 0.0
    assert nasa_score(y, y) == 0.0


def test_rmse_known_value():
    assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(math.sqrt(12.5))


def test_nasa_score_single_late_value():
    # d = +10 (late)  -> exp(10/10) - 1 = e - 1
    assert nasa_score([50.0], [60.0]) == pytest.approx(math.e - 1.0)


def test_nasa_score_single_early_value():
    # d = -10 (early) -> exp(10/13) - 1
    assert nasa_score([50.0], [40.0]) == pytest.approx(math.expm1(10.0 / 13.0))


def test_nasa_penalizes_late_more_than_early():
    """Same-magnitude error is worse when predicting too much life."""
    late = nasa_score([50.0], [60.0])   # +10
    early = nasa_score([50.0], [40.0])  # -10
    assert late > early


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        nasa_score([1.0, 2.0], [1.0])


def test_regression_report_keys_and_bias_sign():
    rep = regression_report([50.0, 50.0], [60.0, 55.0])  # both late
    assert set(rep) == {"n", "rmse", "nasa_score", "mean_abs_error", "mean_error"}
    assert rep["n"] == 2
    assert rep["mean_error"] > 0  # predicts late on average
    assert np.isclose(rep["mean_abs_error"], 7.5)
