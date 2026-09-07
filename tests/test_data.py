"""Tests for rul.data: loading, RUL labeling, and helpers."""
from __future__ import annotations

import numpy as np
import pytest

from rul import data
from rul.config import ALL_COLS, DEFAULT_SUBSET, RUL_CAP


# --------------------------------------------------------------------------- #
# raw_path validation
# --------------------------------------------------------------------------- #
def test_raw_path_rejects_bad_subset():
    with pytest.raises(ValueError):
        data.raw_path("FD999", "train")


def test_raw_path_rejects_bad_split():
    with pytest.raises(ValueError):
        data.raw_path("FD001", "validation")


# --------------------------------------------------------------------------- #
# compute_rul
# --------------------------------------------------------------------------- #
def test_compute_rul_uncapped_is_linear(synthetic_train):
    labeled = data.compute_rul(synthetic_train, cap=None)
    for _unit, grp in labeled.groupby("unit"):
        grp = grp.sort_values("cycle")
        # RUL at the final cycle is 0 and decreases by exactly 1 each cycle.
        assert grp["RUL"].iloc[-1] == 0
        assert np.allclose(np.diff(grp["RUL"]), -1.0)


def test_compute_rul_cap_is_applied(synthetic_train):
    labeled = data.compute_rul(synthetic_train, cap=RUL_CAP)
    assert labeled["RUL"].max() <= RUL_CAP
    # A small cap must actually clip the early-life values.
    capped = data.compute_rul(synthetic_train, cap=5)
    assert capped["RUL"].max() == 5


def test_compute_rul_does_not_mutate_input(synthetic_train):
    before = synthetic_train.copy()
    _ = data.compute_rul(synthetic_train)
    assert "RUL" not in synthetic_train.columns
    assert before.equals(synthetic_train)


# --------------------------------------------------------------------------- #
# last_cycle_rows
# --------------------------------------------------------------------------- #
def test_last_cycle_rows_one_per_engine(synthetic_train):
    last = data.last_cycle_rows(synthetic_train)
    assert set(last["unit"]) == set(synthetic_train["unit"])
    assert last["unit"].is_unique
    for unit, grp in synthetic_train.groupby("unit"):
        assert last.loc[last["unit"] == unit, "cycle"].iloc[0] == grp["cycle"].max()


# --------------------------------------------------------------------------- #
# Integration: the committed FD001 files
# --------------------------------------------------------------------------- #
def test_load_fd001_train_shape(has_fd001):
    if not has_fd001:
        pytest.skip("FD001 data not present")
    df = data.load_subset(DEFAULT_SUBSET, "train")
    assert list(df.columns) == ALL_COLS
    assert df["unit"].nunique() == 100  # 100 training engines
    assert df["cycle"].min() == 1


def test_load_fd001_true_rul(has_fd001):
    if not has_fd001:
        pytest.skip("FD001 data not present")
    true_rul = data.load_true_rul(DEFAULT_SUBSET)
    assert len(true_rul) == 100
    assert (true_rul > 0).all()
    assert true_rul.index.name == "unit"


def test_last_cycle_rows_aligns_with_true_rul(has_fd001):
    if not has_fd001:
        pytest.skip("FD001 data not present")
    test_df = data.load_subset(DEFAULT_SUBSET, "test")
    last = data.last_cycle_rows(test_df)
    true_rul = data.load_true_rul(DEFAULT_SUBSET)
    # One prediction point per test engine, aligned by unit order.
    assert len(last) == len(true_rul)
    assert list(last["unit"]) == list(true_rul.index)
