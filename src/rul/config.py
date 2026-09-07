"""Central configuration for the RUL-CMAPSS project.

Every "magic number" and schema constant lives here so that the data,
feature, model, evaluation and serving layers all agree on the same
definitions. Design decisions encoded in this module are documented in
the project README.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PACKAGE_DIR: Path = Path(__file__).resolve().parent          # .../src/rul
PROJECT_ROOT: Path = PACKAGE_DIR.parents[1]                  # repo root

# Raw C-MAPSS text files. Overridable so the same code runs in Docker/CI
# where the data may live elsewhere (env var CMAPSS_DATA_DIR).
DATA_DIR: Path = Path(os.environ.get("CMAPSS_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR: Path = Path(os.environ.get("RUL_MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR: Path = Path(os.environ.get("RUL_REPORTS_DIR", PROJECT_ROOT / "reports"))

# --------------------------------------------------------------------------- #
# Raw data schema
# --------------------------------------------------------------------------- #
# The C-MAPSS files are space-separated with 26 columns and no header:
#   unit, cycle, 3 operational settings, 21 sensor measurements.
INDEX_COLS: list[str] = ["unit", "cycle"]
SETTING_COLS: list[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS: list[str] = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS: list[str] = INDEX_COLS + SETTING_COLS + SENSOR_COLS

RUL_COL: str = "RUL"
SUBSETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_SUBSET: str = "FD001"

# --------------------------------------------------------------------------- #
# Labeling
# --------------------------------------------------------------------------- #
# Piecewise-linear RUL: engines show no measurable degradation early in life,
# so a linear RUL that climbs into the hundreds is not learnable. We clip the
# training target at RUL_CAP (a.k.a. R_early). 125 is the value used by Heimes
# (2008) and most subsequent C-MAPSS literature for FD001.
RUL_CAP: int = 125

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
# Rolling-window statistics are computed per engine over the most recent
# ROLLING_WINDOW cycles. This is only the library default; the window that each
# subset actually uses is set in PROFILES below and was chosen by grouped
# cross-validation on the training data.
ROLLING_WINDOW: int = 40
# Statistics computed inside each rolling window.
ROLLING_STATS: tuple[str, ...] = ("mean", "std", "min", "max")
# Sensors are dropped when their standard deviation across the whole training
# set is below this threshold (they carry no information for a single
# operating condition, e.g. FD001 sensors 1/5/6/10/16/18/19).
CONSTANT_STD_THRESHOLD: float = 1e-6

# --------------------------------------------------------------------------- #
# Modeling / reproducibility
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 42
# Fraction of *engines* (not rows) held out for early-stopping validation.
VALIDATION_FRACTION: float = 0.2
# Fraction of *engines* held out to calibrate conformal intervals.
CALIBRATION_FRACTION: float = 0.2
# Target coverage for conformal prediction intervals.
CONFIDENCE_LEVEL: float = 0.9

# --------------------------------------------------------------------------- #
# Artifact filenames (inside MODELS_DIR)
# --------------------------------------------------------------------------- #
MODEL_FILENAME: str = "rul_model.joblib"
METRICS_FILENAME: str = "metrics.json"


# --------------------------------------------------------------------------- #
# Per-subset profiles
# --------------------------------------------------------------------------- #
# Single-condition subsets (FD001/FD003) use the plain rolling pipeline.
# Multi-condition subsets (FD002/FD004) turn on regime normalization plus EWMA
# denoising and trend features, with a larger window. They also have about 2.5x
# more engines, so a smaller calibration holdout still leaves plenty of data to
# calibrate the intervals while giving the point model more to learn from.
# Every window and cap below was picked by grouped cross-validation on the
# training set only; the test set was never used to choose them.
@dataclass(frozen=True)
class SubsetProfile:
    window: int
    rul_cap: int
    regime_normalize: bool = False
    n_regimes: int = 1
    ewma_span: int | None = None
    trend: bool = False
    calibration_fraction: float = CALIBRATION_FRACTION


PROFILES: dict[str, SubsetProfile] = {
    "FD001": SubsetProfile(window=40, rul_cap=125),
    "FD003": SubsetProfile(window=40, rul_cap=125),
    "FD002": SubsetProfile(
        window=80, rul_cap=140, regime_normalize=True, n_regimes=6,
        ewma_span=26, trend=True, calibration_fraction=0.1,
    ),
    "FD004": SubsetProfile(
        window=80, rul_cap=140, regime_normalize=True, n_regimes=6,
        ewma_span=26, trend=True, calibration_fraction=0.1,
    ),
}


def profile_for(subset: str) -> SubsetProfile:
    """Return the modeling profile for a C-MAPSS subset."""
    if subset not in PROFILES:
        raise ValueError(f"No profile for subset {subset!r}; known: {sorted(PROFILES)}")
    return PROFILES[subset]
