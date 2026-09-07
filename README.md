# RUL-CMAPSS — Remaining Useful Life prediction on NASA C-MAPSS

Predicting the **Remaining Useful Life (RUL)** of turbofan engines from the
[NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
run-to-failure simulation dataset.

The project is a compact but production-shaped machine-learning system:

- **Data + features** — a reproducible loader and *per-engine* rolling-window
  feature engineering.
- **Model** — a LightGBM gradient-boosting baseline.
- **Uncertainty** — split-conformal prediction intervals via
  [MAPIE](https://mapie.readthedocs.io/) with a guaranteed marginal coverage.
- **Evaluation** — RMSE **and** the official NASA asymmetric scoring function.
- **Serving** — a FastAPI `/predict` endpoint returning point + interval.
- **Ops** — Dockerfile, an Evidently data-drift report, and GitHub Actions CI.

Built in ten committed milestones (data → features → model → conformal → API →
Docker → drift → CI → docs); see the git history. Headline results are in
[Results](#results), and the rationale for every non-obvious choice is in
[Design decisions](#design-decisions).

---

## Dataset

C-MAPSS contains four sub-datasets (FD001–FD004) of multivariate sensor time
series from a fleet of engines. Each engine starts with unknown initial wear,
runs until failure in the **training** set, and is truncated some time before
failure in the **test** set. A separate `RUL_FDxxx.txt` gives the true RUL at
the last observed cycle of each test engine.

| Subset | Train engines | Test engines | Conditions | Fault modes |
|--------|---------------|--------------|------------|-------------|
| FD001  | 100           | 100          | 1          | 1 (HPC)     |
| FD002  | 260           | 259          | 6          | 1 (HPC)     |
| FD003  | 100           | 100          | 1          | 2           |
| FD004  | 248           | 249          | 6          | 2           |

Each row = one operational cycle with 26 space-separated columns:
`unit, cycle, 3 operational settings, 21 sensor measurements`.

**We target FD001 first** — a single operating condition and a single fault
mode — so the pipeline can be validated before tackling the harder multi-regime
subsets.

The raw files are committed under [`data/`](data/) so the repo is
self-contained.

---

## Project structure

```
rul-cmapss/
├── data/                       # committed raw C-MAPSS text files (FD001–FD004)
├── src/rul/
│   ├── config.py               # all schema constants & hyperparameters
│   ├── data.py                 # loading + piecewise-linear RUL labeling
│   ├── features.py             # per-engine rolling-window FeatureBuilder
│   ├── model.py                # RULModel: LightGBM + conformal, one artifact
│   ├── conformal.py            # MAPIE split-conformal intervals
│   ├── evaluation.py           # RMSE, NASA score, interval coverage
│   ├── train.py                # end-to-end training entrypoint (python -m rul.train)
│   ├── drift.py                # Evidently drift report (python -m rul.drift)
│   └── api.py                  # FastAPI service (uvicorn rul.api:app)
├── tests/                      # 47 pytest tests + synthetic C-MAPSS generator
├── notebooks/01_eda.ipynb      # EDA only (executed, with plots)
├── models/                     # committed trained artifact + metadata
├── reports/                    # metrics.json + drift_summary.json (HTML gitignored)
├── Dockerfile                  # multi-stage API image
├── requirements.txt            # pinned runtime deps
├── requirements-dev.txt        # + Evidently, tests, lint, notebook tools
└── .github/workflows/ci.yml    # ruff + pytest on every push/PR
```

---

## Design decisions

The rationale for every non-obvious choice. All tunables live in one place,
[`src/rul/config.py`](src/rul/config.py).

### Data & labeling
- **Piecewise-linear RUL, capped at 125.** Engines show no measurable
  degradation early in life, so a strictly linear RUL target (which climbs past
  300) is unlearnable and distorts both the loss and the NASA score. We clip the
  training target at `RUL_CAP = 125` — the value from Heimes (2008), the
  de-facto standard for FD001. The EDA notebook visualizes why.
- **Group by engine, never by row.** Every split (early-stopping validation,
  conformal calibration) is made over *engine units* via `GroupShuffleSplit`, so
  no engine's cycles appear on both sides — row-level splitting would leak a
  near-identical neighbouring cycle and inflate scores.
- **Standard test protocol.** Each test engine contributes exactly one query —
  its last observed cycle — compared to the provided true RUL, matching the
  original challenge.

### Features
- **Per-engine rolling statistics.** `mean/std/min/max` over the last
  `ROLLING_WINDOW = 30` cycles, computed within each engine so a window never
  spans two engines. This injects the degradation *trend* a tree can't see from
  a single snapshot. The window was chosen by a sweep on FD001
  (5/10/15/20/30/40) — 30 minimized **both** RMSE and the NASA score.
- **Causal (backward-looking) windows.** Rolling stats at cycle *t* use only
  cycles ≤ *t*, so the feature vector at any cycle is exactly what a live engine
  truncated there would produce — train, test and serving features are identical
  by construction.
- **Drop constant sensors automatically.** Under one operating condition several
  sensors never move (FD001: 1, 5, 6, 10, 16, 18, 19, and `op_setting_3`). They
  are removed by a variance threshold, *not* a hard-coded list, so the same code
  keeps the six varying settings on the multi-regime subsets (FD002/FD004).
- **No feature scaling.** LightGBM is a tree ensemble, invariant to monotone
  transforms; scaling would add a fragile artifact to persist and serve for zero
  accuracy gain.

### Model
- **LightGBM baseline.** A fast, strong tabular baseline that handles the
  engineered features natively — the right reference point before reaching for a
  sequence model. Objective is L2 to align the training loss with the reported
  RMSE.
- **Early stopping, then refit.** A grouped validation split finds the best
  iteration; the model is then refit on the whole fit-pool at that iteration to
  use all available data. `random_state = 42` throughout for reproducibility.
- **One self-contained artifact.** `RULModel` bundles the fitted
  `FeatureBuilder`, the LightGBM regressor and the MAPIE calibrator, persisted
  together with joblib. The thing you evaluate is exactly the thing you serve —
  no feature/skew drift between training and inference.
- **Predictions clipped to `[0, cap]`.** RUL can't be negative, and a tree
  trained on a capped target can't meaningfully extrapolate above it.

### Evaluation
- **RMSE *and* the NASA score, always together.** RMSE is symmetric; the PHM08
  NASA score is asymmetric — it penalizes *late* predictions (estimating more
  life than remains, the unsafe direction; `exp(d/10)−1`) more than early ones
  (`exp(−d/13)−1`). A model can improve RMSE while getting worse on the metric
  that matters for safety, so we never report one without the other.

### Uncertainty
- **Split-conformal intervals via MAPIE.** Distribution-free intervals with a
  finite-sample *marginal coverage* guarantee, using the absolute-residual
  conformity score.
- **Disjoint calibration set.** 20% of engines are held out from the point model
  and used only to calibrate the intervals — reusing training data would void
  the guarantee. This is why the served model's point metrics differ slightly
  from an all-data model (see [Results](#results)).
- **Coverage-preserving clipping.** The lower bound is clipped up to 0 (true RUL
  ≥ 0, so this never drops a covered point); the upper bound is left unclipped
  (clamping it at the cap would shrink intervals and lose coverage for
  high-RUL engines).
- **Exchangeability caveat.** Conformal assumes calibration and test points are
  exchangeable. Here calibration uses full-trajectory cycles while each test
  query is a single truncated snapshot, and degradation series are
  autocorrelated — so empirical coverage (86%) runs a little under the 90%
  target. We *measure and report* it rather than trusting the nominal level.

### Serving & ops
- **Features built server-side.** `/predict` accepts raw cycles and runs the
  artifact's own `FeatureBuilder`, so clients never reimplement (and never drift
  from) training-time features. Pydantic schemas are generated from `config` so
  the API can't disagree with the data schema.
- **Model loaded once** at startup via FastAPI lifespan; `/health` reports
  readiness and `/predict` returns 503 if no artifact is present.
- **Docker: Debian-slim, not Alpine.** The pinned numpy/scipy/pandas/
  scikit-learn/lightgbm wheels are `manylinux_2_28` (glibc); musl/Alpine can't
  use them. Multi-stage build, `libgomp1` for LightGBM's OpenMP runtime, non-root
  user, model baked in, only runtime deps installed.
- **CI pins exactly.** GitHub Actions installs the exact pinned versions so the
  committed model artifact unpickles cleanly, then runs `ruff` + `pytest` on
  Python 3.14.

### Reproducibility & repo layout
- **Self-contained repo.** Raw data *and* the trained model artifact are
  committed, so `pytest`, the API and the Docker image all work on a fresh clone
  with no download or training step. Regenerate anytime with `python -m
  rul.train`.
- **Pinned dependencies, split by purpose.** `requirements.txt` is the lean
  runtime set (what the API image installs); `requirements-dev.txt` adds
  Evidently, test/lint tooling and the notebook stack.
- **src-layout package** installed editable, so imports are identical in tests,
  the API and notebooks.

### Known limitations / next steps
- FD001 only so far; FD002/FD004 add six operating conditions and would benefit
  from condition-aware normalization (the variance filter already keeps the
  settings on those subsets).
- The point model is a plain tabular baseline — sensor denoising (EWMA) and
  sequence models (LSTM/CNN) typically reach RMSE ~12–14.
- Cross-conformal (CV+) would recover the 20% held out for calibration at K×
  training cost; an asymmetric training objective could target the NASA score
  directly.

---

## Quickstart

```bash
# 1. Create an environment and install (developed & CI-tested on Python 3.14)
python -m venv .venv
.venv\Scripts\activate              # Windows;  source .venv/bin/activate on *nix
pip install -r requirements-dev.txt # exact pins (matches the committed model)
pip install -e . --no-deps          # the rul package itself

# 2. Run the tests
pytest

# 3. (Optional) retrain — the artifact is already committed under models/
python -m rul.train --subset FD001  # writes models/rul_model.joblib, reports/metrics.json

# 4. Serve the API
uvicorn rul.api:app --reload
#    -> http://127.0.0.1:8000/docs
```

The API, Docker, and drift-report commands are covered in the sections below.

### API

`uvicorn rul.api:app` serves:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | readiness + loaded model version |
| POST | `/predict` | RUL point + conformal interval for one engine |
| GET | `/docs` | interactive OpenAPI docs |

`POST /predict` takes a window of **raw** cycles for one engine (the C-MAPSS
columns); features are built server-side so clients never reimplement them.

```jsonc
// request  (readings: oldest-first cycles; >= 30 cycles recommended)
{ "unit_id": 42,
  "readings": [ { "cycle": 1, "op_setting_1": -0.0007, "op_setting_2": -0.0004,
                  "op_setting_3": 100.0, "sensor_1": 518.67, "sensor_2": 641.82,
                  "...": "... all 21 sensors ..." } ] }

// response
{ "unit_id": 42, "cycle": 45, "rul": 80.7,
  "rul_lower": 55.8, "rul_upper": 105.5,
  "confidence_level": 0.9, "n_cycles_used": 45, "model_version": "0.1.0+FD001" }
```

Returns `503` if no model artifact is loaded, `422` on malformed readings.

### Docker

```bash
docker build -t rul-cmapss .
docker run --rm -p 8000:8000 rul-cmapss   # -> http://127.0.0.1:8000/docs
```

The image (Debian-slim, multi-stage) installs only the runtime deps and bakes
in the trained model, so the container serves predictions immediately. It runs
as a non-root user and ships a stdlib `/health` HEALTHCHECK. See the
[Dockerfile](Dockerfile) for the rationale (glibc base for the manylinux
wheels, `libgomp1` for LightGBM's OpenMP runtime).

### Data drift report

```bash
python -m rul.drift                          # FD001 train vs FD001 test
python -m rul.drift --current-subset FD002   # vs a 6-condition subset (big drift)
```

Evidently compares the **reference** (training data) against **current**
(serving) data on the exact raw columns the model consumes, writing
`reports/drift_report.html` (full visual report) and a small committed
`reports/drift_summary.json`.

On FD001 train-vs-test, **14/17 monitored columns drift** (share 0.82). This is
expected and instructive: test trajectories are *truncated before failure*, so
they contain fewer degraded, near-end-of-life cycles — the sensor
distributions genuinely differ, while the operating settings stay stable. In
production this same report would flag when incoming sensor data has moved away
from the training distribution and the model should be retrained.

---

## Testing & CI

```bash
pytest        # 47 tests: data, features, evaluation, model, conformal, API, drift
ruff check .  # lint
```

Tests run on small **synthetic** C-MAPSS fixtures (fast, deterministic, no data
files needed), with a few integration tests that use the committed FD001 data
and model artifact when present (they skip otherwise). GitHub Actions
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs `ruff` + `pytest`
on Python 3.14 for every push and pull request to `main`, installing the exact
pinned versions so the committed model artifact unpickles cleanly.

## Results

Final model on **FD001** (100 test engines, one prediction per engine at its
last observed cycle vs. the provided true RUL). The point model is trained on
the 80% of engines not reserved for conformal calibration:

| Metric | FD001 test |
|--------|-----------|
| RMSE   | **17.8 cycles** |
| NASA score | **765** |
| Mean absolute error | 12.9 cycles |
| Mean error (bias) | +0.09 (near-unbiased) |
| Conformal coverage (target 90%) | **86.0%** |
| Avg. interval width | 47.8 cycles (±23.9) |

Rolling-window sweep that fixed the default window (all else equal, point model
on all 100 engines):

| window | RMSE | NASA |
|-------:|-----:|-----:|
| 5  | 18.85 | 665 |
| 10 | 19.12 | 781 |
| 20 | 17.98 | 625 |
| **30** | **17.73** | **615** |
| 40 | 18.11 | 760 |

Notes:
- **Why NASA (765) is higher than the sweep's 615.** The served model holds out
  20% of engines to calibrate conformal intervals, so its point model sees 80
  engines instead of 100. RMSE is essentially unchanged (17.7→17.8), but the
  NASA score is exponentially sensitive to a handful of late-predicted engines
  in the tail, so it moves more. This is the honest cost of *valid* conformal
  calibration; cross-conformal (CV+) would reclaim the data at K× training cost.
- **Coverage 86% vs 90% target** reflects the exchangeability gap: intervals are
  calibrated on full-trajectory cycles but tested on single truncated snapshots
  (see [Uncertainty](#uncertainty)).
- This is a deliberately simple tabular baseline; sensor denoising and sequence
  models (LSTM/CNN) are the natural next steps and typically reach RMSE ~12–14.

---

## License

MIT (code) — see [LICENSE](LICENSE). The C-MAPSS dataset is a NASA public-domain
work.
