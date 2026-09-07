"""Tests for rul.regime: regime normalization + regime feature builder.

Covers the audit concerns for the multi-condition pipeline: no test leakage
(train-only normalization stats), causality (truncation invariance), correct
operating-condition normalization, and end-to-end learning.
"""
from __future__ import annotations

import joblib
import numpy as np
import pytest

from rul.config import SENSOR_COLS
from rul.data import compute_rul
from rul.evaluation import rmse
from rul.model import DEFAULT_PARAMS, RULModel
from rul.regime import RegimeFeatureBuilder, RegimeNormalizer
from synth import make_multiregime_frame

FAST = {**DEFAULT_PARAMS, "n_estimators": 80, "learning_rate": 0.1}


# --------------------------------------------------------------------------- #
# RegimeNormalizer
# --------------------------------------------------------------------------- #
def test_normalizer_recovers_regimes_and_standardizes():
    df = make_multiregime_frame(n_units=12, n_regimes=3, seed=0)
    out = RegimeNormalizer(n_regimes=3).fit(df).transform(df)
    assert out["regime"].nunique() == 3
    # Within each recovered regime, sensors are ~zero-mean / ~unit-std on train.
    per_regime = out.groupby("regime")[SENSOR_COLS]
    assert per_regime.mean().abs().to_numpy().max() < 0.05
    assert abs(per_regime.std().to_numpy().mean() - 1.0) < 0.15


def test_normalizer_uses_train_stats_only():
    """No leakage: a shifted (unseen) frame is NOT recentered to zero."""
    train = make_multiregime_frame(n_units=12, n_regimes=3, seed=0)
    norm = RegimeNormalizer(n_regimes=3).fit(train)
    shifted = train.copy()
    shifted[SENSOR_COLS] = shifted[SENSOR_COLS] + 25.0
    out = norm.transform(shifted)
    # Using train means (not the shifted frame's own), the +25 shift survives.
    assert out[SENSOR_COLS].mean().abs().mean() > 0.5


def test_normalizer_roundtrip(tmp_path):
    df = make_multiregime_frame(n_units=8, n_regimes=3, seed=1)
    norm = RegimeNormalizer(n_regimes=3).fit(df)
    path = tmp_path / "norm.joblib"
    joblib.dump(norm, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(
        norm.transform(df)[SENSOR_COLS].to_numpy(),
        loaded.transform(df)[SENSOR_COLS].to_numpy(),
    )


# --------------------------------------------------------------------------- #
# RegimeFeatureBuilder
# --------------------------------------------------------------------------- #
def test_builder_layout_shape_no_nans():
    df = make_multiregime_frame(n_units=8, n_regimes=3, seed=2)
    b = RegimeFeatureBuilder(window=20, ewma_span=10, trend=True, n_regimes=3).fit(df)
    X = b.transform(df)
    assert list(X.columns) == b.feature_names_
    assert X.shape == (len(df), len(b.feature_names_))
    assert X.columns[0] == "cycle" and X.columns[1] == "regime"
    assert not X.isna().any().any()


def test_builder_preserves_row_order():
    df = make_multiregime_frame(n_units=6, n_regimes=3, seed=3)
    b = RegimeFeatureBuilder(window=15, ewma_span=8, n_regimes=3).fit(df)
    X = b.transform(df)
    assert np.allclose(X["cycle"].to_numpy(), df["cycle"].to_numpy())


def test_builder_is_causal_truncation_invariant():
    """Feature vector at cycle t must not depend on cycles > t (no leakage)."""
    df = make_multiregime_frame(n_units=4, n_regimes=3, min_cycles=60, max_cycles=80, seed=4)
    b = RegimeFeatureBuilder(window=20, ewma_span=10, trend=True, n_regimes=3).fit(df)
    eng = df[df["unit"] == 1].sort_values("cycle").reset_index(drop=True)
    full = b.transform(eng)
    T = len(eng)
    for t in (5, 20, 40, T):
        trunc = b.transform(eng.iloc[:t])
        np.testing.assert_allclose(
            full.iloc[t - 1].to_numpy(), trunc.iloc[t - 1].to_numpy(), atol=1e-9
        )


def test_builder_before_fit_raises():
    df = make_multiregime_frame(n_units=3, n_regimes=3, seed=5)
    with pytest.raises(RuntimeError):
        RegimeFeatureBuilder(n_regimes=3).transform(df)


# --------------------------------------------------------------------------- #
# End-to-end through RULModel
# --------------------------------------------------------------------------- #
def test_regime_model_trains_and_beats_naive():
    df = make_multiregime_frame(n_units=24, n_regimes=3, min_cycles=50, max_cycles=90, seed=6)
    model = RULModel(
        regime_normalize=True, n_regimes=3, window=20, ewma_span=10, trend=True,
        rul_cap=125, params=FAST,
    ).fit(df, subset="SYNTH", conformalize=True)
    assert model.metadata["regime_normalize"] is True
    assert model.conformal is not None
    labeled = compute_rul(df, cap=125)
    preds = model.predict(df)
    naive = np.full_like(preds, labeled["RUL"].mean())
    assert rmse(labeled["RUL"], preds) < rmse(labeled["RUL"], naive)
    point, lower, upper = model.predict_interval(df)
    assert np.all(lower <= point + 1e-9) and np.all(point <= upper + 1e-9)
