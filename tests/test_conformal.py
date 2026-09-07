"""Tests for split-conformal intervals (rul.conformal + RULModel wiring)."""
from __future__ import annotations

import numpy as np
import pytest

from rul.data import compute_rul
from rul.evaluation import interval_report
from rul.model import DEFAULT_PARAMS, RULModel
from synth import make_cmapss_frame

FAST_PARAMS = {**DEFAULT_PARAMS, "n_estimators": 80, "learning_rate": 0.1}


@pytest.fixture(scope="module")
def conformal_model():
    """Train once per module: a conformalized model plus its training frame."""
    df = make_cmapss_frame(n_units=30, min_cycles=40, max_cycles=90, seed=2)
    model = RULModel(params=FAST_PARAMS).fit(df, subset="SYNTH", conformalize=True)
    return model, df


def test_calibration_split_is_disjoint(conformal_model):
    model, _ = conformal_model
    md = model.metadata
    assert model.conformal is not None
    assert md["conformalized"] is True
    assert md["n_calib_engines"] > 0
    # Pool and calibration engines partition the training fleet exactly.
    assert md["n_pool_engines"] + md["n_calib_engines"] == md["n_train_engines"] == 30


def test_predict_interval_ordering_and_bounds(conformal_model):
    model, df = conformal_model
    point, lower, upper = model.predict_interval(df)
    n = len(df)
    assert point.shape == lower.shape == upper.shape == (n,)
    assert np.all(lower >= 0.0)              # RUL can't be negative
    assert np.all(lower <= point + 1e-9)
    assert np.all(point <= upper + 1e-9)
    assert np.all(upper - lower > 0)         # non-degenerate intervals


def test_predict_interval_last_cycle_shape(conformal_model):
    model, df = conformal_model
    out = model.predict_interval_last_cycle(df)
    assert list(out.columns) == ["point", "lower", "upper"]
    assert set(out.index) == set(df["unit"].unique())
    assert out.index.name == "unit"
    assert (out["upper"] >= out["lower"]).all()


def test_coverage_is_reasonable(conformal_model):
    """In-distribution coverage should be near/above the target level."""
    model, df = conformal_model
    labeled = compute_rul(df, cap=model.rul_cap)
    _, lower, upper = model.predict_interval(df)
    rep = interval_report(labeled["RUL"], lower, upper, model.confidence_level)
    # Loose sanity bound (synthetic, clear signal): most points are covered.
    assert rep["coverage"] >= 0.8
    assert rep["avg_interval_width"] > 0


def test_predict_interval_requires_conformal():
    df = make_cmapss_frame(n_units=12, seed=7)
    model = RULModel(params=FAST_PARAMS).fit(df, conformalize=False)
    assert model.conformal is None
    with pytest.raises(RuntimeError):
        model.predict_interval(df)


def test_conformal_survives_save_load(conformal_model, tmp_path):
    model, df = conformal_model
    path = tmp_path / "cm.joblib"
    model.save(path)
    loaded = RULModel.load(path)
    a = np.column_stack(model.predict_interval(df))
    b = np.column_stack(loaded.predict_interval(df))
    np.testing.assert_allclose(a, b)
