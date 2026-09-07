"""Operating-condition (regime) normalization for multi-regime subsets.

FD002 and FD004 run engines under **six** operating conditions. A raw sensor
value then mostly encodes *which condition* the engine is in, not how degraded
it is, which swamps the degradation signal. The fix used across the C-MAPSS
literature is:

1. Cluster the 3 operational settings into the operating conditions (KMeans).
2. Z-score every sensor **within its condition** (stats learned on train only).

:class:`RegimeNormalizer` does that. :class:`RegimeFeatureBuilder` builds on it
with EWMA denoising, per-engine rolling statistics and short-horizon trend
features — the recipe that takes FD004 under 20 RMSE.

Both are **causal**: every feature at cycle *t* depends only on cycles ≤ *t* of
the same engine (verified by a truncation-invariance test), so the single
truncated-snapshot test protocol is valid and there is no future leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import RANDOM_SEED, SENSOR_COLS, SETTING_COLS

_STD_FLOOR = 1e-9


@dataclass
class RegimeNormalizer:
    """KMeans over operational settings + per-regime sensor standardization."""

    n_regimes: int = 6
    kmeans_: KMeans | None = None
    means_: dict[int, np.ndarray] = field(default_factory=dict)
    stds_: dict[int, np.ndarray] = field(default_factory=dict)
    _fitted: bool = False

    def fit(self, df: pd.DataFrame) -> RegimeNormalizer:
        self.kmeans_ = KMeans(
            n_clusters=self.n_regimes, n_init=10, random_state=RANDOM_SEED
        ).fit(df[SETTING_COLS].to_numpy())
        regimes = self.kmeans_.predict(df[SETTING_COLS].to_numpy())
        sensors = df[SENSOR_COLS].to_numpy(dtype=float)
        self.means_, self.stds_ = {}, {}
        for r in range(self.n_regimes):
            mask = regimes == r
            if not mask.any():
                # Empty cluster: identity transform for safety.
                self.means_[r] = np.zeros(len(SENSOR_COLS))
                self.stds_[r] = np.ones(len(SENSOR_COLS))
                continue
            self.means_[r] = sensors[mask].mean(axis=0)
            sd = sensors[mask].std(axis=0, ddof=0)
            self.stds_[r] = np.where(sd < _STD_FLOOR, 1.0, sd)  # avoid /0 for flat sensors
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* with a ``regime`` column and per-regime z-scored sensors.

        Row order is preserved; each row is transformed independently (causal).
        """
        if not self._fitted:
            raise RuntimeError("RegimeNormalizer.transform called before fit().")
        out = df.copy()
        out[SENSOR_COLS] = out[SENSOR_COLS].astype(float)  # int sensors -> float
        regimes = self.kmeans_.predict(out[SETTING_COLS].to_numpy())
        out["regime"] = regimes
        sensors = out[SENSOR_COLS].to_numpy(dtype=float)
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.any():
                sensors[mask] = (sensors[mask] - self.means_[r]) / self.stds_[r]
        out[SENSOR_COLS] = sensors
        return out


@dataclass
class RegimeFeatureBuilder:
    """Regime-normalized + EWMA-smoothed rolling/trend features.

    Interface-compatible with :class:`rul.features.FeatureBuilder` (``fit`` /
    ``transform`` / ``kept_columns_`` / ``feature_names_`` / ``window``) so it is
    a drop-in for :class:`rul.model.RULModel`.
    """

    window: int = 80
    ewma_span: int = 25
    trend: bool = True
    n_regimes: int = 6
    stats: tuple[str, ...] = ("mean", "std", "min", "max")
    std_threshold: float = 1e-6
    normalizer_: RegimeNormalizer | None = None
    kept_columns_: list[str] = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)
    _fitted: bool = False

    def _lags(self) -> tuple[int, ...]:
        return (self.window, max(self.window // 2, 1)) if self.trend else ()

    def _layout(self) -> list[str]:
        names = ["cycle", "regime", *self.kept_columns_]
        for stat in self.stats:
            names += [f"{c}_r{stat}" for c in self.kept_columns_]
        for lag in self._lags():
            names += [f"{c}_trend{lag}" for c in self.kept_columns_]
        return names

    def fit(self, df: pd.DataFrame) -> RegimeFeatureBuilder:
        self.normalizer_ = RegimeNormalizer(self.n_regimes).fit(df)
        norm = self.normalizer_.transform(df)
        # Keep sensors that still vary after regime normalization.
        stds = norm[SENSOR_COLS].std(numeric_only=True)
        self.kept_columns_ = [c for c in SENSOR_COLS if float(stds[c]) > self.std_threshold]
        if not self.kept_columns_:
            raise ValueError("No informative sensors after regime normalization.")
        self.feature_names_ = self._layout()
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("RegimeFeatureBuilder.transform called before fit().")
        missing = {"unit", "cycle", *SETTING_COLS} - set(df.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")

        norm = self.normalizer_.transform(df).reset_index(drop=True)
        # Compute on a time-ordered view, then restore the input row order so the
        # output aligns 1:1 with the caller's rows/labels.
        order = norm.sort_values(["unit", "cycle"]).index
        ordered = norm.loc[order]
        units = ordered["unit"]
        # EWMA denoising per engine (causal), indexed by original positions.
        sm = ordered.groupby("unit")[self.kept_columns_].transform(
            lambda s: s.ewm(span=self.ewma_span, min_periods=1).mean()
        )

        feats: dict[str, np.ndarray] = {
            "cycle": norm["cycle"].astype(float).to_numpy(),
            "regime": norm["regime"].astype(float).to_numpy(),
        }
        sm_in = sm.reindex(norm.index)
        for c in self.kept_columns_:
            feats[c] = sm_in[c].to_numpy()

        grouped = sm.groupby(units)
        for stat in self.stats:
            rolled = grouped.rolling(self.window, min_periods=1).agg(stat)
            rolled.index = rolled.index.droplevel(0)
            rolled = rolled.reindex(norm.index)
            for c in self.kept_columns_:
                s = rolled[c]
                if stat == "std":
                    s = s.fillna(0.0)
                feats[f"{c}_r{stat}"] = s.to_numpy()

        for lag in self._lags():
            lagged = grouped.shift(lag).reindex(norm.index)
            for c in self.kept_columns_:
                feats[f"{c}_trend{lag}"] = (sm_in[c] - lagged[c]).fillna(0.0).to_numpy()

        matrix = pd.DataFrame(feats, index=norm.index)
        return matrix[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
