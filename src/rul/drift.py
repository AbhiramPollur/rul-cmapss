"""Data-drift reporting with Evidently.

Monitors whether the distribution of incoming (serving) sensor data has drifted
away from the training distribution the model learned. Drift is a leading
indicator that predictions may no longer be trustworthy and the model should be
retrained.

We compare on the **raw columns the model actually consumes** (the fitted
FeatureBuilder's kept sensors/settings) so the report reflects real model
inputs. The reference is the training data; the current data defaults to the
test set.

Note: comparing FD001 *train* (full run-to-failure) with FD001 *test*
(trajectories truncated before failure) legitimately shows drift, the test
data simply contains fewer near-failure cycles. Comparing against a different
subset (e.g. ``--current-subset FD002``, six operating conditions) shows much
larger, operating-condition drift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from .config import DEFAULT_SUBSET, REPORTS_DIR
from .data import load_subset
from .features import FeatureBuilder
from .model import RULModel


def monitored_columns(reference_df: pd.DataFrame) -> list[str]:
    """Columns to watch: the deployed model's inputs, else a variance filter."""
    try:
        model = RULModel.load()
        if model.feature_builder.kept_columns_:
            return list(model.feature_builder.kept_columns_)
    except (FileNotFoundError, TypeError):
        pass
    return list(FeatureBuilder().fit(reference_df).kept_columns_)


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: list[str] | None = None,
):
    """Run Evidently's DataDriftPreset on ``columns`` and return the snapshot."""
    columns = columns or monitored_columns(reference_df)
    data_def = DataDefinition(numerical_columns=columns)
    reference = Dataset.from_pandas(
        reference_df[columns].reset_index(drop=True), data_definition=data_def
    )
    current = Dataset.from_pandas(
        current_df[columns].reset_index(drop=True), data_definition=data_def
    )
    report = Report(metrics=[DataDriftPreset()])
    return report.run(current_data=current, reference_data=reference)


def drift_summary(snapshot) -> dict:
    """Extract a compact, JSON-serializable summary from a report snapshot."""
    metrics = snapshot.dict().get("metrics", [])
    per_column: dict[str, float] = {}
    drifted: list[str] = []
    n_drifted: int | None = None
    share: float | None = None

    for metric in metrics:
        name = metric.get("metric_name", "")
        value = metric.get("value")
        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            n_drifted = int(value.get("count", 0))
            share = float(value.get("share", 0.0))
        elif name.startswith("ValueDrift"):
            cfg = metric.get("config", {})
            col = cfg.get("column")
            method = str(cfg.get("method", ""))
            threshold = float(cfg.get("threshold", 0.1))
            score = float(value)
            per_column[col] = score
            # Evidently auto-selects the drift method by data size: statistical
            # tests (K-S, chi-square) report a p-value (drift when BELOW the
            # threshold); distance/divergence methods (Wasserstein, JS, PSI)
            # report a distance (drift when ABOVE it).
            is_pvalue = "p_value" in method.lower()
            if (score < threshold) if is_pvalue else (score > threshold):
                drifted.append(col)

    return {
        "n_columns": len(per_column),
        "n_drifted": n_drifted if n_drifted is not None else len(drifted),
        "drift_share": share,
        "drifted_columns": drifted,
        "per_column_score": per_column,
    }


def generate_report(
    reference_subset: str = DEFAULT_SUBSET,
    current_subset: str = DEFAULT_SUBSET,
    current_split: str = "test",
    out_html: Path | None = None,
    out_json: Path | None = None,
) -> dict:
    """Build train-vs-current drift report, save HTML + JSON summary."""
    reference_df = load_subset(reference_subset, "train")
    current_df = load_subset(current_subset, current_split)

    snapshot = build_drift_report(reference_df, current_df)
    summary = drift_summary(snapshot)
    summary["reference"] = f"{reference_subset}/train"
    summary["current"] = f"{current_subset}/{current_split}"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_html = out_html or REPORTS_DIR / "drift_report.html"
    out_json = out_json or REPORTS_DIR / "drift_summary.json"
    snapshot.save_html(str(out_html))
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[drift] reference={summary['reference']}  current={summary['current']}")
    print(
        f"[drift] {summary['n_drifted']}/{summary['n_columns']} columns drifted "
        f"(share={summary['drift_share']:.2f})"
    )
    print(f"[drift] HTML   -> {out_html}")
    print(f"[drift] summary-> {out_json}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate an Evidently data-drift report.")
    parser.add_argument("--reference-subset", default=DEFAULT_SUBSET)
    parser.add_argument("--current-subset", default=DEFAULT_SUBSET)
    parser.add_argument("--current-split", default="test", choices=["train", "test"])
    args = parser.parse_args(argv)
    generate_report(
        reference_subset=args.reference_subset,
        current_subset=args.current_subset,
        current_split=args.current_split,
    )


if __name__ == "__main__":
    main()
