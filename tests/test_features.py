"""Tests for rul.features: constant-column dropping, layout, no leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rul.config import ALL_COLS, ROLLING_STATS
from rul.features import FeatureBuilder, make_xy
from synth import CONSTANT_SENSOR_IDS, make_cmapss_frame


def test_fit_drops_constant_columns(synthetic_train):
    builder = FeatureBuilder().fit(synthetic_train)
    # Constant settings and the synthetic constant sensors are removed.
    for sid in CONSTANT_SENSOR_IDS:
        assert f"sensor_{sid}" not in builder.kept_columns_
    assert "op_setting_1" not in builder.kept_columns_  # constant in synthetic data
    # Varying sensors survive.
    assert "sensor_2" in builder.kept_columns_
    assert len(builder.kept_columns_) == 21 - len(CONSTANT_SENSOR_IDS)


def test_feature_layout_length(synthetic_train):
    builder = FeatureBuilder().fit(synthetic_train)
    n_kept = len(builder.kept_columns_)
    # cycle + raw kept cols + one block per rolling statistic.
    expected = 1 + n_kept + n_kept * len(ROLLING_STATS)
    assert len(builder.feature_names_) == expected
    assert builder.feature_names_[0] == "cycle"


def test_transform_shape_and_no_nans(synthetic_train):
    builder = FeatureBuilder()
    X = builder.fit_transform(synthetic_train)
    assert X.shape == (len(synthetic_train), len(builder.feature_names_))
    assert list(X.columns) == builder.feature_names_
    assert not X.isna().any().any()  # std NaNs are filled with 0


def test_transform_before_fit_raises(synthetic_train):
    with pytest.raises(RuntimeError):
        FeatureBuilder().transform(synthetic_train)


def test_transform_missing_columns_raises(synthetic_train):
    builder = FeatureBuilder().fit(synthetic_train)
    with pytest.raises(ValueError):
        builder.transform(synthetic_train.drop(columns=["sensor_2"]))


def test_no_cross_engine_leakage():
    """Rolling window must reset at each engine boundary."""
    df = make_cmapss_frame(n_units=3, min_cycles=15, max_cycles=25, seed=5)
    builder = FeatureBuilder().fit(df)
    X = builder.transform(df)
    tagged = X.assign(_unit=df["unit"].to_numpy(), _cycle=df["cycle"].to_numpy())
    w = builder.window
    for _unit, grp in tagged.groupby("_unit"):
        first = grp.sort_values("_cycle").iloc[0]
        # At an engine's first cycle the window contains only that row, so every
        # rolling stat equals the raw value, impossible if a prior engine leaked.
        for col in builder.kept_columns_:
            assert first[f"{col}_roll{w}_mean"] == pytest.approx(first[col])
            assert first[f"{col}_roll{w}_min"] == pytest.approx(first[col])
            assert first[f"{col}_roll{w}_max"] == pytest.approx(first[col])


def test_rolling_mean_matches_manual():
    """Explicit numeric check of the rolling mean on a hand-built engine."""
    n = 8
    df = pd.DataFrame(
        {
            "unit": 1,
            "cycle": np.arange(1, n + 1),
            **{c: 0.0 for c in ALL_COLS if c not in ("unit", "cycle", "sensor_2")},
            "sensor_2": np.arange(1.0, n + 1.0),  # 1,2,...,8
        }
    )[ALL_COLS]
    builder = FeatureBuilder(window=3).fit(df)
    X = builder.transform(df)
    # Last row's 3-window mean over sensor_2 = mean(6,7,8) = 7.
    assert X["sensor_2_roll3_mean"].iloc[-1] == pytest.approx(7.0)
    # Second row = mean(1,2) = 1.5 (min_periods=1).
    assert X["sensor_2_roll3_mean"].iloc[1] == pytest.approx(1.5)


def test_make_xy_fit_and_align(synthetic_train):
    from rul.data import compute_rul

    labeled = compute_rul(synthetic_train)
    builder = FeatureBuilder()
    X, y = make_xy(labeled, builder, fit=True)
    assert len(X) == len(y) == len(labeled)
    assert y.max() <= 125  # capped RUL
    assert list(X.columns) == builder.feature_names_
