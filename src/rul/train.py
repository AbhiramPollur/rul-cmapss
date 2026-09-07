"""End-to-end training entrypoint.

Usage
-----
    python -m rul.train --subset FD001

Loads a C-MAPSS subset, trains :class:`~rul.model.RULModel`, evaluates on the
official test protocol (one prediction per engine at its last cycle vs. the
provided true RUL), and writes the model artifact plus a metrics report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import (
    DEFAULT_SUBSET,
    METRICS_FILENAME,
    MODEL_FILENAME,
    MODELS_DIR,
    REPORTS_DIR,
)
from .data import load_subset, load_true_rul
from .evaluation import interval_report, regression_report
from .model import RULModel


def train_and_evaluate(subset: str = DEFAULT_SUBSET) -> dict:
    """Train on ``train_<subset>`` and evaluate on ``test_<subset>``."""
    print(f"[train] loading subset {subset} ...")
    train_df = load_subset(subset, "train")
    test_df = load_subset(subset, "test")
    true_rul = load_true_rul(subset)

    print(f"[train] fitting RULModel on {train_df['unit'].nunique()} engines ...")
    model = RULModel().fit(train_df, subset=subset)

    print("[train] evaluating on the test set (last cycle per engine) ...")
    preds = model.predict_last_cycle(test_df)
    # Align predictions to the ground-truth unit order.
    preds = preds.reindex(true_rul.index)
    test_metrics = regression_report(true_rul.to_numpy(), preds.to_numpy())

    # Conformal interval coverage on the same test protocol.
    intervals = model.predict_interval_last_cycle(test_df).reindex(true_rul.index)
    coverage = interval_report(
        true_rul.to_numpy(),
        intervals["lower"].to_numpy(),
        intervals["upper"].to_numpy(),
        model.confidence_level,
    )

    report = {
        "subset": subset,
        "test": test_metrics,
        "conformal": coverage,
        "model": model.metadata,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model.save(MODELS_DIR / MODEL_FILENAME)
    model.save_metadata(MODELS_DIR / "model_metadata.json")
    metrics_path = REPORTS_DIR / METRICS_FILENAME
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[train] saved model  -> {model_path}")
    print(f"[train] saved metrics-> {metrics_path}")
    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    t = report["test"]
    c = report.get("conformal", {})
    print("\n" + "=" * 52)
    print(f"  Subset            : {report['subset']}")
    print(f"  Test engines      : {t['n']}")
    print(f"  RMSE              : {t['rmse']:.3f} cycles")
    print(f"  NASA score        : {t['nasa_score']:.1f}")
    print(f"  Mean abs. error   : {t['mean_abs_error']:.3f} cycles")
    print(f"  Mean error (bias) : {t['mean_error']:+.3f}  (+ = predicts late)")
    if c:
        target = c["confidence_level"] * 100
        print(f"  Conformal target  : {target:.0f}% coverage")
        print(f"  Empirical coverage: {c['coverage'] * 100:.1f}%")
        print(f"  Avg interval width: {c['avg_interval_width']:.1f} cycles")
    print("=" * 52 + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the RUL-CMAPSS model.")
    parser.add_argument(
        "--subset", default=DEFAULT_SUBSET, help="C-MAPSS subset (FD001..FD004)."
    )
    args = parser.parse_args(argv)
    train_and_evaluate(args.subset)


if __name__ == "__main__":
    main()
