"""Tests for rul.model.RULModel: training, prediction, persistence."""
from __future__ import annotations

import numpy as np
import pytest

from rul.config import RUL_CAP
from rul.model import DEFAULT_PARAMS, RULModel


@pytest.fixture
def fast_model(synthetic_train_large) -> RULModel:
    """A quickly-trained model for tests (few trees)."""
    params = {**DEFAULT_PARAMS, "n_estimators": 60, "learning_rate": 0.1}
    return RULModel(params=params).fit(synthetic_train_large, subset="SYNTH")


def test_predict_before_fit_raises(synthetic_train_large):
    with pytest.raises(RuntimeError):
        RULModel().predict(synthetic_train_large)


def test_predict_shape_and_range(fast_model, synthetic_train_large):
    preds = fast_model.predict(synthetic_train_large)
    assert preds.shape == (len(synthetic_train_large),)
    assert preds.min() >= 0.0
    assert preds.max() <= RUL_CAP


def test_predict_last_cycle_one_per_engine(fast_model, synthetic_train_large):
    per_engine = fast_model.predict_last_cycle(synthetic_train_large)
    units = synthetic_train_large["unit"].unique()
    assert set(per_engine.index) == set(units)
    assert per_engine.index.name == "unit"
    assert len(per_engine) == len(units)


def test_metadata_populated(fast_model):
    md = fast_model.metadata
    assert md["subset"] == "SYNTH"
    assert md["rul_cap"] == RUL_CAP
    assert md["n_features"] == len(fast_model.feature_builder.feature_names_)
    assert md["n_train_engines"] == 24


def test_save_load_roundtrip(fast_model, synthetic_train_large, tmp_path):
    path = tmp_path / "m.joblib"
    fast_model.save(path)
    assert path.exists()
    loaded = RULModel.load(path)
    before = fast_model.predict(synthetic_train_large)
    after = loaded.predict(synthetic_train_large)
    np.testing.assert_allclose(before, after)


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        RULModel.load(tmp_path / "nope.joblib")


def test_model_learns_signal(fast_model, synthetic_train_large):
    """On the synthetic data (monotone degradation) the model should beat a
    naive mean predictor on RMSE, a sanity check that training works."""
    from rul.data import compute_rul
    from rul.evaluation import rmse

    labeled = compute_rul(synthetic_train_large)
    preds = fast_model.predict(synthetic_train_large)
    naive = np.full_like(preds, labeled["RUL"].mean())
    assert rmse(labeled["RUL"], preds) < rmse(labeled["RUL"], naive)
