"""Interactive demo UI for the RUL-CMAPSS model.

This is a thin presentation layer over the trained artifacts. It loads the same
`RULModel` the API serves, so predictions here match predictions there. Nothing
is retrained or recomputed at request time beyond the model's own feature build.

Run locally:  streamlit run demo/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Make the sibling `src/` importable so the demo runs on Streamlit Community
# Cloud straight from the repo, without the package needing to be pip installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rul.config import MODELS_DIR, PROFILES  # noqa: E402
from rul.data import load_subset, load_true_rul  # noqa: E402
from rul.model import RULModel  # noqa: E402

SUBSET_BLURB = {
    "FD001": "one operating condition, one fault mode (the simplest set)",
    "FD002": "six operating conditions, one fault mode",
    "FD003": "one operating condition, two fault modes",
    "FD004": "six operating conditions, two fault modes (the hardest set)",
}

# A few sensors that carry a clear degradation trend, for the trend panel.
TREND_SENSORS = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_15"]


@st.cache_resource(show_spinner=False)
def get_model(subset: str) -> RULModel:
    name = "rul_model.joblib" if subset == "FD001" else f"rul_model_{subset}.joblib"
    return RULModel.load(MODELS_DIR / name)


@st.cache_data(show_spinner=False)
def get_test_data(subset: str):
    test = load_subset(subset, "test")
    true = load_true_rul(subset)
    return test, true


def predict(model: RULModel, engine_df: pd.DataFrame) -> dict:
    row = model.predict_interval_last_cycle(engine_df).iloc[0]
    return {
        "point": float(row["point"]),
        "lower": float(row["lower"]),
        "upper": float(row["upper"]),
        "cycle": int(engine_df["cycle"].max()),
        "n": len(engine_df),
    }


def gauge(point: float, lower: float, upper: float, true: float | None, axis_max: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(point, 1),
            number={"suffix": " cycles", "font": {"size": 40}},
            title={"text": "Predicted remaining useful life"},
            gauge={
                "axis": {"range": [0, axis_max]},
                "bar": {"color": "#2563eb", "thickness": 0.28},
                "steps": [
                    {"range": [0, 30], "color": "#fee2e2"},
                    {"range": [30, 75], "color": "#fef9c3"},
                    {"range": [75, axis_max], "color": "#dcfce7"},
                    # The 90% interval drawn as a darker band on top.
                    {"range": [lower, upper], "color": "rgba(37,99,235,0.20)"},
                ],
                "threshold": (
                    {"line": {"color": "#111827", "width": 4}, "thickness": 0.75, "value": true}
                    if true is not None
                    else {}
                ),
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
    return fig


st.set_page_config(page_title="Turbofan RUL Predictor", page_icon="🛩️", layout="wide")

st.title("🛩️ Turbofan Engine Remaining Useful Life")
st.caption(
    "Predicts how many operating cycles a jet engine has left before failure, from "
    "its sensor history, on the NASA C-MAPSS dataset. Point estimate plus a calibrated "
    "90% confidence interval."
)

with st.sidebar:
    st.header("Choose an engine")
    subset = st.selectbox("Dataset", list(PROFILES), index=0)
    st.caption(SUBSET_BLURB[subset])

    test_df, true_rul = get_test_data(subset)
    units = sorted(test_df["unit"].unique().tolist())
    unit = st.selectbox("Test engine", units, index=0)

    engine_all = test_df[test_df["unit"] == unit].sort_values("cycle").reset_index(drop=True)
    max_cycle = int(engine_all["cycle"].max())
    cut = st.slider(
        "Observe up to cycle",
        min_value=int(min(20, max_cycle)),
        max_value=max_cycle,
        value=max_cycle,
        help="Slide back to ask the model earlier in the engine's life, when less is known.",
    )
    st.caption(
        "The engine is one the model never saw in training. The known answer comes "
        "from the dataset's ground-truth file."
    )

engine_df = engine_all[engine_all["cycle"] <= cut].reset_index(drop=True)
model = get_model(subset)
result = predict(model, engine_df)

# True RUL is defined at the engine's actual last observed cycle. When the slider
# cuts earlier, the engine really has that many extra cycles still ahead.
true_at_full = float(true_rul.loc[unit])
true_now = true_at_full + (max_cycle - cut)

axis_max = max(150.0, result["upper"] + 10, true_now + 10)

left, right = st.columns([5, 4])
with left:
    st.plotly_chart(
        gauge(result["point"], result["lower"], result["upper"], true_now, axis_max),
        use_container_width=True,
    )
with right:
    st.subheader("At this point in the engine's life")
    c1, c2 = st.columns(2)
    c1.metric("Predicted RUL", f"{result['point']:.0f} cyc")
    c2.metric("True RUL", f"{true_now:.0f} cyc", delta=f"{result['point'] - true_now:+.0f} vs true")
    st.metric("90% interval", f"{result['lower']:.0f} to {result['upper']:.0f} cyc")
    st.caption(
        f"Using {result['n']} cycles of history, predicting at cycle {result['cycle']}. "
        f"The dark line on the gauge marks the true answer; the shaded band is the "
        f"90% interval."
    )

st.divider()
st.subheader("Sensor history")
st.caption(
    "Each sensor standardized to its own scale so the trends are comparable. This "
    "drift is what the model reads to judge wear."
)
present = [s for s in TREND_SENSORS if s in engine_df.columns]
hist = engine_df[["cycle"] + present]
trend = go.Figure()
for s in present:
    mu, sd = hist[s].mean(), hist[s].std()
    y = (hist[s] - mu) / sd if sd > 1e-9 else hist[s] * 0.0
    trend.add_trace(go.Scatter(x=hist["cycle"], y=y, mode="lines", name=s))
trend.update_layout(
    height=320,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="cycle",
    yaxis_title="standardized reading",
    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
)
st.plotly_chart(trend, use_container_width=True)

with st.expander("How this works"):
    st.markdown(
        "- The model is a gradient boosted ensemble (LightGBM) trained per dataset, "
        "with a 1D CNN used where it wins.\n"
        "- Features are backward looking rolling statistics per engine, plus operating "
        "condition normalization on the six condition sets, so nothing leaks from the future.\n"
        "- The interval comes from split conformal calibration, so the 90% is an honest, "
        "distribution free coverage target.\n"
        "- RUL is scored against the true uncapped answer, the same protocol used by the "
        "published literature."
    )
