"""Tests for rul.deep (the CNN sequence model). Skipped if torch is absent."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from rul.config import RUL_CAP  # noqa: E402
from rul.deep import DeepRULModel  # noqa: E402
from synth import make_cmapss_frame, make_multiregime_frame  # noqa: E402


@pytest.fixture(scope="module")
def deep_model():
    df = make_cmapss_frame(n_units=20, min_cycles=25, max_cycles=45, seed=0)
    model = DeepRULModel(
        window=10, rul_cap=RUL_CAP, n_models=1, epochs=8, patience=5, calibration_fraction=0.2
    ).fit(df, subset="SYNTH")
    return model, df


def test_predict_last_cycle_one_per_engine(deep_model):
    model, df = deep_model
    preds = model.predict_last_cycle(df)
    assert set(preds.index) == set(df["unit"].unique())
    assert preds.index.name == "unit"
    assert preds.min() >= 0.0 and preds.max() <= RUL_CAP


def test_intervals_ordered(deep_model):
    model, df = deep_model
    out = model.predict_interval_last_cycle(df)
    assert list(out.columns) == ["point", "lower", "upper"]
    assert (out["lower"] <= out["point"] + 1e-6).all()
    assert (out["point"] <= out["upper"] + 1e-6).all()
    assert (out["lower"] >= 0.0).all()
    assert model.interval_half_width_ > 0


def test_save_load_roundtrip(deep_model, tmp_path):
    model, df = deep_model
    path = tmp_path / "deep.joblib"
    model.save(path)
    loaded = DeepRULModel.load(path)
    np.testing.assert_allclose(
        model.predict_last_cycle(df).to_numpy(),
        loaded.predict_last_cycle(df).to_numpy(),
    )


def test_features_are_causal(deep_model):
    """Truncation invariance: the window at cycle t must not use cycles > t."""
    model, df = deep_model
    eng = df[df["unit"] == df["unit"].unique()[0]].sort_values("cycle").reset_index(drop=True)
    unit = int(eng["unit"].iloc[0])
    mat_full = model._sequences(eng)[0][unit]
    length = len(eng)
    for t in (3, 8, length):
        mat_trunc = model._sequences(eng.iloc[:t])[0][unit]
        np.testing.assert_allclose(mat_full[t - 1], mat_trunc[t - 1], atol=1e-6)


def test_regime_variant_trains():
    df = make_multiregime_frame(n_units=18, n_regimes=3, min_cycles=30, max_cycles=50, seed=1)
    model = DeepRULModel(
        window=10, rul_cap=RUL_CAP, regime_normalize=True, n_regimes=3,
        n_models=1, epochs=6, patience=4, calibration_fraction=0.2,
    ).fit(df, subset="SYNTH")
    preds = model.predict_last_cycle(df)
    assert len(preds) == df["unit"].nunique()
    assert model.metadata["regime_normalize"] is True
