"""FastAPI service exposing the RUL model.

Design
------
* The client sends a **window of raw cycles** for one engine (the same columns
  as the C-MAPSS files). Feature engineering happens *server-side* using the
  builder baked into the artifact, so clients never reimplement it and can never
  drift from training-time features.
* Prediction is made at the **last** supplied cycle and returns a point RUL plus
  a conformal interval at the model's confidence level.
* The model is loaded once at startup (FastAPI lifespan) and shared via
  ``app.state``. ``/health`` reports readiness; ``/predict`` returns 503 if the
  artifact is missing.

Run locally:  ``uvicorn rul.api:app --reload``  ->  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model

from . import __version__
from .config import SENSOR_COLS, SETTING_COLS
from .model import RULModel

# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
# One reading = the raw C-MAPSS columns for a single cycle. Built from config so
# the API schema can never disagree with the data schema.
_reading_fields: dict = {
    "cycle": (int, Field(..., ge=1, description="Operational cycle number.")),
}
for _c in SETTING_COLS:
    _reading_fields[_c] = (float, Field(..., description=f"Operational setting {_c[-1]}."))
for _c in SENSOR_COLS:
    _reading_fields[_c] = (float, Field(..., description=f"Sensor measurement {_c.split('_')[-1]}."))

CycleReading = create_model("CycleReading", **_reading_fields)


class PredictRequest(BaseModel):
    unit_id: int = Field(1, description="Engine identifier (for your own tracking).")
    readings: list[CycleReading] = Field(  # type: ignore[valid-type]
        ...,
        min_length=1,
        description=(
            "Consecutive cycles for ONE engine, oldest first. Provide at least "
            "the rolling window (30) cycles for the best features; fewer is "
            "accepted but yields weaker trend signal."
        ),
    )


class PredictResponse(BaseModel):
    unit_id: int
    cycle: int = Field(..., description="Last cycle, i.e. the cycle RUL is predicted at.")
    rul: float = Field(..., description="Point estimate of remaining useful life (cycles).")
    rul_lower: float
    rul_upper: float
    confidence_level: float
    n_cycles_used: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


# --------------------------------------------------------------------------- #
# App + model lifecycle
# --------------------------------------------------------------------------- #
def _model_version(model: RULModel | None) -> str | None:
    if model is None:
        return None
    subset = model.metadata.get("subset") or "unknown"
    return f"{__version__}+{subset}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once at startup (None if unavailable)."""
    try:
        app.state.model = RULModel.load()
    except FileNotFoundError:
        app.state.model = None
    yield
    app.state.model = None


app = FastAPI(
    title="RUL-CMAPSS API",
    version=__version__,
    summary="Remaining Useful Life prediction for turbofan engines (NASA C-MAPSS).",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "rul-cmapss", "version": __version__, "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model: RULModel | None = getattr(app.state, "model", None)
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_version=_model_version(model),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    model: RULModel | None = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not loaded. Train one with `python -m rul.train`.",
        )

    # Build a raw-schema DataFrame for this single engine.
    rows = [r.model_dump() for r in request.readings]
    df = pd.DataFrame(rows)
    df.insert(0, "unit", request.unit_id)
    df = df.sort_values("cycle").reset_index(drop=True)

    try:
        result = model.predict_interval_last_cycle(df)
    except Exception as exc:  # feature building / model errors -> 422
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    row = result.iloc[0]
    return PredictResponse(
        unit_id=request.unit_id,
        cycle=int(df["cycle"].max()),
        rul=round(float(row["point"]), 3),
        rul_lower=round(float(row["lower"]), 3),
        rul_upper=round(float(row["upper"]), 3),
        confidence_level=model.confidence_level,
        n_cycles_used=len(df),
        model_version=_model_version(model) or "unknown",
    )
