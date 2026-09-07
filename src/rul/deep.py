"""A 1D convolutional sequence model for RUL.

The LightGBM baseline works on one cycle at a time. This model instead reads a
sliding window of recent cycles, which is what lets it reach the low-teens RMSE
that a tree model cannot. The network follows the compact design of Li et al.
(2018), a strong and well-tested choice on C-MAPSS: a stack of temporal
convolutions, a one-filter convolution to mix them, then two dense layers.

The same ideas as the rest of the project carry over. Features are causal (a
window only looks backwards), splits are made by engine, the RUL target is the
piecewise-linear label, and the six-condition subsets (FD002, FD004) are regime
normalized and then exponentially smoothed to expose the degradation trend.
Features are standardized (not min-max scaled), which matters for the regime
subsets because their normalized values have occasional large values that would
otherwise crush the useful range. Predictions come with a conformal interval
calibrated on held-out engines, and a small ensemble averages out the run-to-run
noise of neural nets.

Training needs PyTorch, which is a development dependency only. The served API
keeps using the light LightGBM artifact, while these models give the best
offline accuracy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

from .config import (
    CONFIDENCE_LEVEL,
    RANDOM_SEED,
    SENSOR_COLS,
    SETTING_COLS,
    VALIDATION_FRACTION,
)
from .data import compute_rul
from .regime import RegimeNormalizer

_STD_FLOOR = 1e-9
_SNAPSHOTS_PER_ENGINE = 12


class _DCNN(nn.Module):
    """Temporal CNN: four conv layers, a mixing conv, then two dense layers."""

    def __init__(self, n_features: int, length: int, filters: int = 10, kernel: int = 10):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(n_features if i == 0 else filters, filters, kernel, padding="same")
                for i in range(4)
            ]
        )
        self.mix = nn.Conv1d(filters, 1, 3, padding="same")
        self.fc1 = nn.Linear(length, 100)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, features, length)
        for conv in self.convs:
            x = torch.tanh(conv(x))
        x = torch.tanh(self.mix(x)).flatten(1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(self.drop(x)).squeeze(1)


@dataclass
class DeepRULModel:
    """Windowed 1D-CNN ensemble with conformal intervals."""

    window: int = 30
    rul_cap: int = 125
    regime_normalize: bool = False
    n_regimes: int = 6
    ewma_span: int = 0  # 0 disables smoothing; used for the regime subsets
    n_models: int = 3
    epochs: int = 150
    batch_size: int = 512
    lr: float = 1e-3
    patience: int = 20
    confidence_level: float = CONFIDENCE_LEVEL
    calibration_fraction: float = 0.1
    seed: int = RANDOM_SEED
    # Learned state.
    feature_cols_: list[str] = field(default_factory=list)
    scale_mean_: np.ndarray | None = None
    scale_std_: np.ndarray | None = None
    regime_normalizer_: RegimeNormalizer | None = None
    nets_: list = field(default_factory=list)
    interval_half_width_: float | None = None
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Feature preparation
    # ------------------------------------------------------------------ #
    def _normalized_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.regime_normalize:
            return self.regime_normalizer_.transform(df)
        out = df.copy()
        out[SENSOR_COLS] = out[SENSOR_COLS].astype(float)
        return out

    def _raw_features(self, frame_sorted: pd.DataFrame) -> np.ndarray:
        """Feature matrix after optional EWMA smoothing, before standardization."""
        feats = frame_sorted[self.feature_cols_].to_numpy(dtype=float)
        if self.ewma_span and self.ewma_span > 1:
            tmp = pd.DataFrame(feats, columns=self.feature_cols_)
            tmp["unit"] = frame_sorted["unit"].to_numpy()
            feats = (
                tmp.groupby("unit")[self.feature_cols_]
                .transform(lambda s: s.ewm(span=self.ewma_span, min_periods=1).mean())
                .to_numpy()
            )
        return feats

    def _sequences(self, df: pd.DataFrame) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Per-engine standardized (and smoothed) feature matrices in cycle order.

        Returns ``(matrices, ruls)`` keyed by unit; ``ruls`` is empty when the
        frame has no RUL column.
        """
        frame = self._normalized_frame(df).sort_values(["unit", "cycle"]).reset_index(drop=True)
        feats = (self._raw_features(frame) - self.scale_mean_) / self.scale_std_
        has_rul = "RUL" in frame.columns
        rul = frame["RUL"].to_numpy(dtype=float) if has_rul else None
        matrices, ruls = {}, {}
        for unit, idx in frame.groupby("unit").groups.items():
            rows = np.asarray(idx)
            matrices[int(unit)] = feats[rows]
            if has_rul:
                ruls[int(unit)] = rul[rows]
        return matrices, ruls

    def _window_at(self, matrix: np.ndarray, end: int) -> np.ndarray:
        """Length-``window`` slice ending at row ``end`` (inclusive), padded if short."""
        start = end - self.window + 1
        if start >= 0:
            return matrix[start : end + 1]
        pad = np.repeat(matrix[:1], -start, axis=0)
        return np.vstack([pad, matrix[: end + 1]])

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self, train_df: pd.DataFrame, subset: str = "") -> DeepRULModel:
        labeled = compute_rul(train_df, cap=self.rul_cap)

        # Hold out calibration engines for the conformal interval.
        calib_split = GroupShuffleSplit(
            1, test_size=self.calibration_fraction, random_state=self.seed
        )
        pool_idx, calib_idx = next(calib_split.split(labeled, groups=labeled["unit"]))
        pool = labeled.iloc[pool_idx].reset_index(drop=True)
        calib = labeled.iloc[calib_idx].reset_index(drop=True)

        # Fit normalization and choose feature columns on the pool only.
        if self.regime_normalize:
            self.regime_normalizer_ = RegimeNormalizer(self.n_regimes).fit(pool)
            norm_pool = self.regime_normalizer_.transform(pool)
            self.feature_cols_ = [c for c in SENSOR_COLS if norm_pool[c].std() > _STD_FLOOR]
        else:
            norm_pool = pool.copy()
            norm_pool[SENSOR_COLS] = norm_pool[SENSOR_COLS].astype(float)
            candidates = SETTING_COLS + SENSOR_COLS
            self.feature_cols_ = [c for c in candidates if norm_pool[c].std() > _STD_FLOOR]

        # Standardization statistics, computed after any EWMA smoothing.
        pool_sorted = norm_pool.sort_values(["unit", "cycle"]).reset_index(drop=True)
        smoothed = self._raw_features(pool_sorted)
        self.scale_mean_ = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        self.scale_std_ = np.where(std < _STD_FLOOR, 1.0, std)

        # Grouped train/validation split inside the pool for early stopping.
        val_split = GroupShuffleSplit(1, test_size=VALIDATION_FRACTION, random_state=self.seed)
        tr_idx, va_idx = next(val_split.split(pool, groups=pool["unit"]))
        X_tr, y_tr = self._build_windows(pool.iloc[tr_idx])
        X_va, y_va = self._build_windows(pool.iloc[va_idx])

        self.nets_ = [
            self._train_one(X_tr, y_tr, X_va, y_va, seed=self.seed + i)
            for i in range(self.n_models)
        ]
        self._calibrate(calib)

        self.metadata = {
            "subset": subset,
            "model": "dcnn",
            "window": self.window,
            "rul_cap": self.rul_cap,
            "regime_normalize": self.regime_normalize,
            "ewma_span": self.ewma_span,
            "n_features": len(self.feature_cols_),
            "n_models": self.n_models,
            "confidence_level": self.confidence_level,
            "interval_half_width": self.interval_half_width_,
            "n_train_engines": int(labeled["unit"].nunique()),
        }
        return self

    def _build_windows(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrices, ruls = self._sequences(df)
        xs, ys = [], []
        for unit, mat in matrices.items():
            target = ruls[unit]
            for end in range(self.window - 1, len(mat)):
                xs.append(self._window_at(mat, end))
                ys.append(target[end])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def _train_one(self, X_tr, y_tr, X_va, y_va, seed: int) -> _DCNN:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = _DCNN(len(self.feature_cols_), self.window)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        cap = float(self.rul_cap)
        xt = torch.tensor(X_tr).transpose(1, 2)
        yt = torch.tensor(y_tr / cap)
        xv = torch.tensor(X_va).transpose(1, 2)
        best_rmse, best_state, bad = float("inf"), None, 0
        n = len(xt)
        for _epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss_fn(net(xt[idx]), yt[idx]).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                pred = np.clip(net(xv).numpy() * cap, 0.0, cap)
            rmse_va = float(np.sqrt(np.mean((pred - y_va) ** 2)))
            if rmse_va < best_rmse - 1e-3:
                best_rmse, bad = rmse_va, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        return net

    def _calibrate(self, calib: pd.DataFrame) -> None:
        """Absolute-residual split conformal on mid-life snapshots of held-out engines."""
        matrices, ruls = self._sequences(calib)
        rng = np.random.default_rng(self.seed)
        windows, targets = [], []
        for unit, mat in matrices.items():
            rul = ruls[unit]
            hi = min(float(self.rul_cap), float(rul.max()))
            if hi <= 5.0:
                ends = range(len(mat))
            else:
                ends = [int(np.argmin(np.abs(rul - t)))
                        for t in rng.uniform(5.0, hi, size=_SNAPSHOTS_PER_ENGINE)]
            for end in ends:
                windows.append(self._window_at(mat, end))
                targets.append(rul[end])
        preds = self._predict_windows(np.asarray(windows, dtype=np.float32))
        residuals = np.abs(np.asarray(targets) - preds)
        q_level = min(1.0, np.ceil((len(residuals) + 1) * self.confidence_level) / len(residuals))
        self.interval_half_width_ = float(np.quantile(residuals, q_level))

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _predict_windows(self, X: np.ndarray) -> np.ndarray:
        xt = torch.tensor(X).transpose(1, 2)
        cap = float(self.rul_cap)
        with torch.no_grad():
            preds = [net(xt).numpy() for net in self.nets_]
        return np.clip(np.mean(preds, axis=0) * cap, 0.0, cap)

    def predict_last_cycle(self, df: pd.DataFrame) -> pd.Series:
        """One prediction per engine, from the window ending at its last cycle."""
        if not self.nets_:
            raise RuntimeError("DeepRULModel is not fitted.")
        matrices, _ = self._sequences(df)
        units = sorted(matrices)
        windows = [self._window_at(matrices[u], len(matrices[u]) - 1) for u in units]
        preds = self._predict_windows(np.asarray(windows, dtype=np.float32))
        return pd.Series(preds, index=pd.Index(units, name="unit"), name="pred")

    def predict_interval_last_cycle(self, df: pd.DataFrame) -> pd.DataFrame:
        point = self.predict_last_cycle(df)
        if self.interval_half_width_ is None:
            raise RuntimeError("Model has no calibrated interval.")
        q = self.interval_half_width_
        values = point.to_numpy()
        return pd.DataFrame(
            {"point": values, "lower": np.clip(values - q, 0.0, None), "upper": values + q},
            index=point.index,
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> DeepRULModel:
        model = joblib.load(Path(path))
        for net in model.nets_:
            net.eval()
        return model

    def save_metadata(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")
