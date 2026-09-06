"""Loading and RUL-labeling of the NASA C-MAPSS dataset.

The raw files are whitespace-separated with 26 unnamed columns. This module
turns them into tidy, typed :class:`pandas.DataFrame` objects and derives the
Remaining Useful Life (RUL) target used for supervised training.

Nothing here does feature engineering or modeling — those live in
:mod:`rul.features` and :mod:`rul.model`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (
    ALL_COLS,
    DATA_DIR,
    DEFAULT_SUBSET,
    RUL_CAP,
    RUL_COL,
    SUBSETS,
)

Split = str  # "train" | "test"


def raw_path(subset: str = DEFAULT_SUBSET, split: Split = "train") -> Path:
    """Return the path to a raw C-MAPSS file, validating the arguments."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}.")
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}.")
    return DATA_DIR / f"{split}_{subset}.txt"


def load_subset(subset: str = DEFAULT_SUBSET, split: Split = "train") -> pd.DataFrame:
    """Load a raw C-MAPSS subset into a typed DataFrame with named columns.

    Parameters
    ----------
    subset:
        One of ``FD001``..``FD004``.
    split:
        ``"train"`` (run-to-failure) or ``"test"`` (truncated).

    Returns
    -------
    DataFrame with columns :data:`rul.config.ALL_COLS`
    (``unit, cycle, op_setting_1..3, sensor_1..21``).
    """
    path = raw_path(subset, split)
    if not path.exists():
        raise FileNotFoundError(
            f"C-MAPSS file not found: {path}. Set CMAPSS_DATA_DIR or place the "
            f"raw .txt files under {DATA_DIR}."
        )
    # `\s+` collapses the runs of spaces used as separators; a trailing space on
    # each source line can yield a spurious all-NaN column, so slice to 26.
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(ALL_COLS)]
    if df.shape[1] != len(ALL_COLS):
        raise ValueError(
            f"Expected {len(ALL_COLS)} columns in {path.name}, found {df.shape[1]}."
        )
    df.columns = ALL_COLS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def compute_rul(df: pd.DataFrame, cap: int | None = RUL_CAP) -> pd.DataFrame:
    """Add a piecewise-linear ``RUL`` column to *training* data.

    For run-to-failure trajectories the RUL at a given cycle is
    ``max_cycle_of_engine - current_cycle``. It is clipped at ``cap`` because
    engines show no measurable degradation early in life, so an uncapped linear
    target is not learnable (see README → Design decisions → Labeling).

    Pass ``cap=None`` to keep the raw linear RUL (used in tests and EDA).
    """
    out = df.copy()
    max_cycle = out.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - out["cycle"]).astype(float)
    if cap is not None:
        rul = rul.clip(upper=float(cap))
    out[RUL_COL] = rul
    return out


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per engine: the last observed cycle.

    This is the point at which the test-set RUL is defined, so evaluation
    predicts on exactly these rows.
    """
    idx = df.groupby("unit")["cycle"].idxmax()
    return df.loc[idx].sort_values("unit").reset_index(drop=True)


def load_true_rul(subset: str = DEFAULT_SUBSET) -> pd.Series:
    """Load the ground-truth test RUL vector (``RUL_FDxxx.txt``).

    Returns a Series indexed by ``unit`` (1..N) so it aligns with
    :func:`last_cycle_rows` output.
    """
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}.")
    path = DATA_DIR / f"RUL_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"True-RUL file not found: {path}.")
    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    index = pd.RangeIndex(1, len(values) + 1, name="unit")
    return pd.Series(values, index=index, name=RUL_COL).astype(float)
