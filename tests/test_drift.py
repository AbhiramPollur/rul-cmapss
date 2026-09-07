"""Tests for rul.drift (Evidently data-drift reporting)."""
from __future__ import annotations

from rul.config import SENSOR_COLS, SETTING_COLS
from rul.drift import build_drift_report, drift_summary, monitored_columns
from synth import make_cmapss_frame


def test_summary_has_expected_keys():
    ref = make_cmapss_frame(n_units=8, seed=0)
    cur = make_cmapss_frame(n_units=8, seed=0)
    summary = drift_summary(build_drift_report(ref, cur, columns=["sensor_2", "sensor_3"]))
    assert set(summary) >= {
        "n_columns",
        "n_drifted",
        "drift_share",
        "drifted_columns",
        "per_column_score",
    }
    assert summary["n_columns"] == 2


def test_identical_data_has_no_drift():
    ref = make_cmapss_frame(n_units=8, seed=0)
    cur = make_cmapss_frame(n_units=8, seed=0)  # byte-identical -> zero drift
    summary = drift_summary(build_drift_report(ref, cur, columns=["sensor_2", "sensor_3"]))
    assert summary["n_drifted"] == 0
    assert summary["drifted_columns"] == []


def test_shift_is_detected():
    ref = make_cmapss_frame(n_units=8, seed=0)
    cur = make_cmapss_frame(n_units=8, seed=0)
    cur["sensor_2"] = cur["sensor_2"] + 50.0  # large deterministic shift
    summary = drift_summary(build_drift_report(ref, cur, columns=["sensor_2", "sensor_3"]))
    assert "sensor_2" in summary["drifted_columns"]      # shifted column flagged
    assert "sensor_3" not in summary["drifted_columns"]  # untouched column not flagged


def test_monitored_columns_are_valid():
    ref = make_cmapss_frame(n_units=6, seed=1)
    cols = monitored_columns(ref)
    assert len(cols) > 0
    assert all(c in SETTING_COLS + SENSOR_COLS for c in cols)
