"""Per-engine rolling-window feature engineering.

A single cycle's sensor snapshot tells a tree model little about *trend*, which
is what degradation is. :class:`FeatureBuilder` therefore augments each row with
rolling statistics computed **within each engine** (so a window never spans two
engines) and drops sensors that are constant under the operating condition.

The builder is stateful: :meth:`fit` learns *which* columns to keep from the
training set, and :meth:`transform` applies the exact same layout to train,
test and live-serving data — guaranteeing the model always sees identical
feature columns in identical order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import (
    CONSTANT_STD_THRESHOLD,
    ROLLING_STATS,
    ROLLING_WINDOW,
    SENSOR_COLS,
    SETTING_COLS,
)


@dataclass
class FeatureBuilder:
    """Learns informative columns, then emits raw + rolling-window features."""

    window: int = ROLLING_WINDOW
    stats: tuple[str, ...] = ROLLING_STATS
    std_threshold: float = CONSTANT_STD_THRESHOLD
    kept_columns_: list[str] = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)
    _fitted: bool = False

    # ------------------------------------------------------------------ #
    # fit / transform
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        """Select operational-setting/sensor columns that actually vary.

        Columns whose standard deviation across the training set is below
        ``std_threshold`` carry no information (e.g. FD001 sensors that are
        constant under its single operating condition) and are dropped.
        """
        candidates = [c for c in SETTING_COLS + SENSOR_COLS if c in df.columns]
        stds = df[candidates].std(numeric_only=True)
        self.kept_columns_ = [c for c in candidates if float(stds[c]) > self.std_threshold]
        if not self.kept_columns_:
            raise ValueError("No non-constant sensor/setting columns found to build features.")
        self.feature_names_ = self._layout()
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the feature matrix for *df* (rows aligned to *df*'s order)."""
        if not self._fitted:
            raise RuntimeError("FeatureBuilder.transform called before fit().")
        missing = {"unit", "cycle", *self.kept_columns_} - set(df.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")

        df = df.reset_index(drop=True)
        features: dict[str, pd.Series] = {"cycle": df["cycle"].astype(float)}
        for col in self.kept_columns_:
            features[col] = df[col].astype(float)
        features.update(self._rolling(df))
        matrix = pd.DataFrame(features, index=df.index)
        return matrix[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _layout(self) -> list[str]:
        """The ordered list of feature-column names this builder produces."""
        names = ["cycle", *self.kept_columns_]
        for stat in self.stats:
            names += [f"{c}_roll{self.window}_{stat}" for c in self.kept_columns_]
        return names

    def _rolling(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """Compute rolling stats per engine, aligned back to *df*'s row order."""
        ordered = df.sort_values(["unit", "cycle"])
        grouped = ordered.groupby("unit")[self.kept_columns_]
        out: dict[str, pd.Series] = {}
        for stat in self.stats:
            rolled = grouped.rolling(self.window, min_periods=1).agg(stat)
            # groupby.rolling yields a (unit, original_index) MultiIndex; drop the
            # unit level so the result realigns with the original rows.
            rolled.index = rolled.index.droplevel(0)
            rolled = rolled.reindex(df.index)
            for col in self.kept_columns_:
                series = rolled[col]
                if stat == "std":
                    # std of a single observation is NaN -> no variation yet.
                    series = series.fillna(0.0)
                out[f"{col}_roll{self.window}_{stat}"] = series
        return out


def make_xy(
    labeled_df: pd.DataFrame,
    builder: FeatureBuilder,
    target_col: str = "RUL",
    fit: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) from a RUL-labeled frame.

    Set ``fit=True`` on the training frame to learn the kept columns first.
    """
    if fit:
        builder.fit(labeled_df)
    X = builder.transform(labeled_df)
    y = labeled_df[target_col].astype(float).reset_index(drop=True)
    return X, y
