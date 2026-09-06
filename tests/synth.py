"""Synthetic C-MAPSS-shaped data for fast, deterministic tests.

Generates frames with the exact 26-column schema, a single operating
condition, a handful of deliberately constant sensors (mimicking FD001's
1/5/6/10/16/18/19) and the rest drifting monotonically with age, so a model
can actually learn RUL from them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rul.config import ALL_COLS, SENSOR_COLS, SETTING_COLS

# Sensor ids that stay constant under a single operating condition (as in FD001).
CONSTANT_SENSOR_IDS: tuple[int, ...] = (1, 5, 6, 10, 16, 18, 19)


def make_engine(unit: int, n_cycles: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build one run-to-failure trajectory for a single engine."""
    cycles = np.arange(1, n_cycles + 1, dtype=int)
    frac = (cycles - 1) / max(n_cycles - 1, 1)  # 0 -> 1 across the engine's life
    data: dict[str, np.ndarray] = {
        "unit": np.full(n_cycles, unit, dtype=int),
        "cycle": cycles,
    }
    for col in SETTING_COLS:
        data[col] = np.full(n_cycles, 1.0)  # single operating condition
    for i, col in enumerate(SENSOR_COLS, start=1):
        if i in CONSTANT_SENSOR_IDS:
            data[col] = np.full(n_cycles, 100.0)
        else:
            base = 500.0 + i
            drift = 20.0 * frac * (1.0 if i % 2 == 0 else -1.0)  # monotone with age
            noise = rng.normal(0.0, 0.2, n_cycles)
            data[col] = base + drift + noise
    return pd.DataFrame(data)


def make_cmapss_frame(
    n_units: int = 6,
    min_cycles: int = 20,
    max_cycles: int = 45,
    seed: int = 0,
) -> pd.DataFrame:
    """Concatenate several synthetic engines into one C-MAPSS-shaped frame."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit in range(1, n_units + 1):
        n = int(rng.integers(min_cycles, max_cycles + 1))
        frames.append(make_engine(unit, n, rng))
    return pd.concat(frames, ignore_index=True)[ALL_COLS]
