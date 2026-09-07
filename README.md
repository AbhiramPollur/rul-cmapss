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

> Status: built milestone-by-milestone. Results below are filled in after the
> model is trained (see [Results](#results)).

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
├── data/                     # committed raw C-MAPSS text files
├── src/rul/
│   ├── config.py             # all schema constants & hyperparameters
│   ├── data.py               # loading + RUL labeling            (milestone 2)
│   ├── features.py           # per-engine rolling-window features (milestone 3)
│   ├── model.py              # LightGBM train / predict / persist (milestone 4)
│   ├── evaluation.py         # RMSE + NASA score                 (milestone 4)
│   ├── conformal.py          # MAPIE split-conformal intervals   (milestone 5)
│   ├── train.py              # end-to-end training entrypoint
│   ├── drift.py              # Evidently drift report            (milestone 8)
│   └── api.py                # FastAPI service                   (milestone 6)
├── tests/                    # pytest unit + integration tests
├── notebooks/                # EDA only (no training logic)
├── models/                   # trained artifacts (committed)
├── reports/                  # generated metrics + drift HTML
├── Dockerfile                # API image                         (milestone 7)
└── .github/workflows/ci.yml  # run tests on push                 (milestone 9)
```

---

## Design decisions

Rationale for every non-obvious choice. (Expanded as milestones land.)

### Data & labeling
- **Piecewise-linear RUL with a cap of 125.** Engines show no measurable
  degradation early in life, so a strictly linear RUL target is unlearnable and
  distorts both loss and the NASA score. We clip the training target at
  `RUL_CAP = 125` (Heimes 2008), the de-facto standard for FD001.
- **Group by engine, never by row.** All splits (validation, calibration) are
  done over *engine units* so no engine's cycles leak across the split.

### Features
- **Per-engine rolling statistics.** `mean/std/min/max` over the last
  `ROLLING_WINDOW = 30` cycles, computed within each engine so windows never
  cross engine boundaries. This injects the temporal trend a tree model cannot
  see from a single cycle. The window was chosen by a sweep on FD001
  (5/10/15/20/30/40) — 30 minimized **both** RMSE and the NASA score.
- **Drop constant sensors automatically.** Under one operating condition
  several sensors are constant (FD001: 1, 5, 6, 10, 16, 18, 19). They are
  removed by a variance threshold rather than a hard-coded list, so the same
  code generalizes to other subsets.
- **No feature scaling.** LightGBM is a tree ensemble and is invariant to
  monotone feature transforms; scaling would add a fragile artifact to persist
  and serve for no accuracy gain.

### Model
- **LightGBM baseline.** Fast, strong tabular baseline with native handling of
  the engineered features; a sensible reference before any deep sequence model.

### Evaluation
- **RMSE and the NASA scoring function.** RMSE is symmetric; the NASA score
  penalizes *late* predictions (predicting more life than remains — the unsafe
  direction) more than early ones. Reporting both avoids optimizing a metric
  that hides safety-relevant errors.

### Uncertainty
- **Split-conformal intervals (MAPIE).** Distribution-free intervals with a
  finite-sample marginal coverage guarantee, calibrated on a held-out set of
  engines. (Caveat: conformal assumes exchangeability, only approximately true
  for degradation time series — documented in the conformal section.)

### Serving & ops
- FastAPI service computes features server-side from a window of raw cycles.
- Evidently produces a train-vs-serving data-drift report.
- GitHub Actions runs the test suite on every push.

---

## Quickstart

```bash
# 1. Create an environment and install (Python 3.11+; developed on 3.14)
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on *nix
pip install -e ".[dev]"

# 2. Run the tests
pytest

# 3. Train (writes models/rul_model.joblib and reports/metrics.json)
python -m rul.train --subset FD001

# 4. Serve the API
uvicorn rul.api:app --reload
#    -> http://127.0.0.1:8000/docs
```

More commands (Docker, drift report) are documented in their milestone
sections below as they are added.

---

## Results

LightGBM baseline on **FD001** (100 test engines, one prediction per engine at
its last observed cycle vs. the provided true RUL):

| Metric | FD001 test |
|--------|-----------|
| RMSE   | **17.7 cycles** |
| NASA score | **615** |
| Mean absolute error | 12.6 cycles |
| Mean error (bias) | +0.26 (near-unbiased) |
| Conformal coverage @ 90% | _TBD (milestone 5)_ |

Rolling-window sweep that fixed the default window (all else equal):

| window | RMSE | NASA |
|-------:|-----:|-----:|
| 5  | 18.85 | 665 |
| 10 | 19.12 | 781 |
| 20 | 17.98 | 625 |
| **30** | **17.73** | **615** |
| 40 | 18.11 | 760 |

This is a deliberately simple tabular baseline; sensor denoising and sequence
models (LSTM/CNN) are the natural next steps and typically reach RMSE ~12–14.

---

## License

MIT (code) — see [LICENSE](LICENSE). The C-MAPSS dataset is a NASA public-domain
work.
