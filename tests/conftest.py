"""Shared pytest fixtures.

Unit tests run on small synthetic frames (fast, deterministic, no data files).
An integration marker guards the few tests that read the committed FD001 data.
"""
from __future__ import annotations

import pandas as pd
import pytest

from rul.config import DEFAULT_SUBSET
from rul.data import raw_path
from synth import make_cmapss_frame


@pytest.fixture
def synthetic_train() -> pd.DataFrame:
    """Small frame (6 engines) for data/feature unit tests."""
    return make_cmapss_frame(n_units=6, seed=0)


@pytest.fixture
def synthetic_train_large() -> pd.DataFrame:
    """Larger frame (24 engines) for model/conformal tests that need splits."""
    return make_cmapss_frame(n_units=24, min_cycles=40, max_cycles=90, seed=1)


@pytest.fixture(scope="session")
def has_fd001() -> bool:
    """Whether the real FD001 files are present (they are committed by default)."""
    return raw_path(DEFAULT_SUBSET, "train").exists()
