"""Tests for the FastAPI service (rul.api).

These exercise the real committed artifact when present; if no model is loaded
(e.g. a fresh checkout without training) the prediction tests skip rather than
fail, while the contract tests (validation, health) still run.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rul.api import app
from rul.config import RUL_CAP, SENSOR_COLS, SETTING_COLS
from synth import make_cmapss_frame


@pytest.fixture(scope="module")
def client():
    # `with` runs the lifespan handler, loading the model into app.state.
    with TestClient(app) as c:
        yield c


def _sample_readings(n_cycles: int = 40) -> list[dict]:
    df = make_cmapss_frame(n_units=1, min_cycles=n_cycles, max_cycles=n_cycles, seed=3)
    cols = ["cycle", *SETTING_COLS, *SENSOR_COLS]
    return df[cols].to_dict("records")


def _model_loaded(client) -> bool:
    return client.get("/health").json()["model_loaded"]


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "rul-cmapss"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "model_loaded", "model_version"}


def test_predict_happy_path(client):
    if not _model_loaded(client):
        pytest.skip("no model artifact loaded")
    payload = {"unit_id": 7, "readings": _sample_readings(40)}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["unit_id"] == 7
    assert body["n_cycles_used"] == 40
    assert body["cycle"] == 40
    assert 0.0 <= body["rul"] <= RUL_CAP
    assert body["rul_lower"] <= body["rul"] <= body["rul_upper"]
    assert body["confidence_level"] == pytest.approx(0.9)
    assert body["model_version"]


def test_predict_accepts_short_window(client):
    if not _model_loaded(client):
        pytest.skip("no model artifact loaded")
    # Fewer cycles than the rolling window must still work (min_periods=1).
    r = client.post("/predict", json={"unit_id": 1, "readings": _sample_readings(3)})
    assert r.status_code == 200, r.text
    assert r.json()["n_cycles_used"] == 3


def test_predict_missing_sensor_is_422(client):
    readings = _sample_readings(5)
    del readings[0]["sensor_2"]  # drop a required field
    r = client.post("/predict", json={"unit_id": 1, "readings": readings})
    assert r.status_code == 422


def test_predict_empty_readings_is_422(client):
    r = client.post("/predict", json={"unit_id": 1, "readings": []})
    assert r.status_code == 422
