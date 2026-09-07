"""Train the 1D-CNN sequence model for a subset.

Usage:  python -m rul.train_deep --subset FD001

Window 30 and a RUL cap of 125 are the standard C-MAPSS CNN settings and are
used for every subset, so nothing is tuned per subset. Regime normalization is
turned on for the six-condition subsets (FD002, FD004). Writes a model artifact
and a metrics report next to the LightGBM ones.
"""
from __future__ import annotations

import argparse
import json

from .config import DEFAULT_SUBSET, MODELS_DIR, REPORTS_DIR
from .data import load_subset, load_true_rul
from .deep import DeepRULModel
from .evaluation import interval_report, regression_report

REGIME_SUBSETS = {"FD002", "FD004"}
WINDOW = 30
RUL_CAP = 125


def train_and_evaluate(subset: str = DEFAULT_SUBSET) -> dict:
    print(f"[deep] loading subset {subset} ...")
    train_df = load_subset(subset, "train")
    test_df = load_subset(subset, "test")
    true_rul = load_true_rul(subset)

    regime = subset in REGIME_SUBSETS
    print(f"[deep] training CNN ensemble on {train_df['unit'].nunique()} engines "
          f"(regime_normalize={regime}) ...")
    model = DeepRULModel(
        window=WINDOW, rul_cap=RUL_CAP, regime_normalize=regime, n_regimes=6,
        ewma_span=10 if regime else 0,
    ).fit(train_df, subset=subset)

    preds = model.predict_last_cycle(test_df).reindex(true_rul.index)
    test_metrics = regression_report(true_rul.to_numpy(), preds.to_numpy())
    intervals = model.predict_interval_last_cycle(test_df).reindex(true_rul.index)
    coverage = interval_report(
        true_rul.to_numpy(),
        intervals["lower"].to_numpy(),
        intervals["upper"].to_numpy(),
        model.confidence_level,
    )
    report = {"subset": subset, "model": "dcnn", "test": test_metrics,
              "conformal": coverage, "model_meta": model.metadata}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model.save(MODELS_DIR / f"deep_model_{subset}.joblib")
    model.save_metadata(MODELS_DIR / f"deep_metadata_{subset}.json")
    metrics_path = REPORTS_DIR / f"deep_metrics_{subset}.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[deep] saved model  -> {model_path}")
    print(f"[deep] saved metrics-> {metrics_path}")
    t, c = report["test"], report["conformal"]
    print("\n" + "=" * 52)
    print(f"  Subset            : {subset}  (CNN sequence model)")
    print(f"  Test engines      : {t['n']}")
    print(f"  RMSE              : {t['rmse']:.3f} cycles   (vs true uncapped RUL)")
    print(f"  NASA score        : {t['nasa_score']:.1f}")
    print(f"  Mean abs. error   : {t['mean_abs_error']:.3f} cycles")
    cov_pct, tgt_pct = c["coverage"] * 100, c["confidence_level"] * 100
    print(f"  Empirical coverage: {cov_pct:.1f}%  (target {tgt_pct:.0f}%)")
    print("=" * 52 + "\n")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the CNN RUL model.")
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    args = parser.parse_args(argv)
    train_and_evaluate(args.subset)


if __name__ == "__main__":
    main()
