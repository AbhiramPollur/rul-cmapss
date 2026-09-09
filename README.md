# RUL-CMAPSS: Remaining Useful Life prediction on NASA C-MAPSS

This project predicts how many operating cycles a turbofan engine has left before
it fails, using the [NASA C-MAPSS](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
run-to-failure simulation data. It is small enough to read in an afternoon but
shaped like something you would actually put in production: a clean data and
feature layer, two models (a LightGBM baseline and a 1D-CNN sequence model),
calibrated uncertainty on every prediction, a FastAPI service, a Docker image, a
data-drift report, and CI.

All four sub-datasets (FD001 to FD004) are trained and evaluated. Every design
choice is explained below, and the numbers are reproducible from a fresh clone
because the data and the trained models are committed.

## The dataset

C-MAPSS has four sub-datasets. In each one, a fleet of engines is run until it
fails in the training set, and cut off some time before failure in the test set.
A separate file gives the true remaining life at the last recorded cycle of each
test engine, which is what we are scored against.

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|--------|--------------:|-------------:|:--------------------:|:-----------:|
| FD001  | 100 | 100 | 1 | 1 (HPC) |
| FD002  | 260 | 259 | 6 | 1 (HPC) |
| FD003  | 100 | 100 | 1 | 2 |
| FD004  | 248 | 249 | 6 | 2 |

Each row is one cycle with 26 space-separated numbers: an engine id, the cycle
number, three operational settings, and 21 sensor readings. FD002 and FD004 are
the hard ones because their engines switch between six operating conditions, and
FD003 and FD004 add a second fault mode.

The raw files live in [`data/`](data/) so nothing has to be downloaded.

## Project layout

```
rul-cmapss/
├── data/                       raw C-MAPSS text files (FD001 to FD004)
├── src/rul/
│   ├── config.py               schema constants, hyperparameters, per-subset profiles
│   ├── data.py                 loading and the piecewise-linear RUL label
│   ├── features.py             per-engine rolling-window features (FD001/FD003)
│   ├── regime.py               regime normalization and features (FD002/FD004)
│   ├── model.py                RULModel: LightGBM plus conformal, saved as one file
│   ├── deep.py                 DeepRULModel: 1D-CNN sequence model (best accuracy)
│   ├── conformal.py            MAPIE split-conformal prediction intervals
│   ├── evaluation.py           RMSE, the NASA score, and interval coverage
│   ├── train.py                trains the LightGBM model (python -m rul.train)
│   ├── train_deep.py           trains the CNN model (python -m rul.train_deep)
│   ├── drift.py                Evidently drift report (python -m rul.drift)
│   └── api.py                  FastAPI service (uvicorn rul.api:app)
├── tests/                      pytest suite plus a synthetic data generator
├── notebooks/01_eda.ipynb      exploratory analysis only, with plots
├── models/                     the trained models, committed
├── reports/                    metrics and the drift summary
├── Dockerfile                  the API image
└── .github/workflows/ci.yml    runs lint and tests on every push
```

## How it works, and why

### Labelling the target

An engine barely changes for most of its life and only starts degrading once a
fault appears, so a target that counts down linearly from 300+ is not something a
model can learn from the early cycles. The standard fix, which this project uses,
is a piecewise-linear label: the remaining life is held flat at a cap for the
healthy period and only counts down once it drops below that cap. The cap is
chosen per subset by cross-validation (see below). The true test labels are never
capped; we always score against the real remaining life.

### Features

A single cycle tells a tree model almost nothing about a trend, and degradation
is a trend. So for each engine we add rolling statistics (mean, standard
deviation, min and max) over a window of recent cycles. The window is computed
inside each engine so it never mixes two engines together, and it only ever looks
backwards, which matters for a reason explained in the correctness section.

Sensors that never move carry no information and are dropped automatically by a
variance check rather than a hard-coded list, so the same code adapts to whichever
sensors happen to be flat in a given subset. Tree models are invariant to
monotonic rescaling, so there is no feature scaling to fit, persist and serve.

### The six-condition subsets

FD002 and FD004 need one more idea. When an engine keeps switching between six
operating conditions, a raw sensor value mostly tells you which condition it is in
rather than how worn it is. So for those subsets we cluster the three operational
settings into the six conditions with k-means and standardize each sensor inside
its own condition, using statistics learned on the training data only. On top of
that we exponentially smooth each sensor to cut noise, and add a couple of trend
features that measure how much a smoothed sensor has moved over the window. This
is what takes FD004 from unusable to competitive (see [Results](#results)).

### The model

LightGBM is a fast, strong baseline for tabular features and a sensible thing to
beat before reaching for anything heavier. Training uses early stopping on a
validation split made by engine, then refits on the whole pool at the best
iteration so no data is wasted. Everything is seeded for reproducibility. The
fitted feature builder, the regressor and the conformal calibrator are saved
together as a single file, so the thing you evaluate is exactly the thing you
serve. Predictions are clipped to the sensible range of zero up to the cap.

There is also a second model, a 1D convolutional sequence model
([`deep.py`](src/rul/deep.py), trained with `python -m rul.train_deep`). Instead
of one cycle at a time it reads a sliding window of recent cycles, which lets it
follow the degradation trajectory. It needs PyTorch, so it is a training-time
tool only; the served API stays on the light LightGBM artifact. The two models
are compared and the better one is kept per subset. It wins on the clean
single-condition FD001 (RMSE 15.1 versus LightGBM's 18.8) and loses on the
six-condition subsets, where the tabular model with regime normalization is
stronger (see [Results](#results)).

### Two metrics, always together

RMSE is symmetric and easy to read in cycles. The NASA score from the PHM08
challenge is deliberately asymmetric: predicting more life than an engine really
has is the dangerous direction, so it is punished harder than predicting too
little. A model can quietly get worse on the dangerous kind of error while its
RMSE improves, so this project always reports both.

### Uncertainty you can act on

A point estimate is not enough for a maintenance decision, so every prediction
comes with an interval. We use split conformal prediction through MAPIE, which
gives a distribution-free coverage guarantee by measuring the model's own errors
on a set of engines it never trained on. To keep that guarantee honest, those
calibration engines are held out from the point model entirely. The lower bound is
clamped at zero because remaining life cannot be negative, and the upper bound is
left alone so the interval can still cover engines with a lot of life left.

One caveat worth stating plainly: conformal coverage assumes the calibration and
test points are interchangeable, but here calibration uses full trajectories while
each test query is a single truncated snapshot, and degradation series are
correlated in time. So the real coverage lands a little under the target, and we
measure and report it rather than quoting the nominal number.

### Choosing hyperparameters honestly

The window and cap for every subset are chosen by three-fold cross-validation on
the training engines, scored the same way the test set is scored (predict at a
truncated point, compare to the real remaining life). The test set is never used
to pick them, so the reported test numbers are a fair estimate of how the model
generalizes. The EWMA span for the six-condition subsets is tied to the window
(about a third of it) rather than tuned separately.

## Results

Every model is scored the standard way: one prediction per test engine, at its
last recorded cycle, compared against the true remaining life. Two points on the
metric, because they matter and are easy to get wrong:

- **RMSE is against the true, uncapped RUL** (the actual remaining cycles). That
  is the number the published records use, so it is directly comparable. An
  earlier version of this project mistakenly scored against a capped ground
  truth, which flatters the six-condition subsets enormously; that is fixed.
- **The RUL cap is a training-time choice only.** It caps the label the model
  learns from, not the test ground truth, and it is chosen per subset by
  cross-validation on the training data: 125 for the single-condition subsets and
  160 for the six-condition ones, whose engines run much longer (up to 195 cycles
  of true remaining life).

Best model for each subset:

| Subset | Best model | RMSE | NASA score | Coverage (target 90%) | Reference SOTA |
|--------|-----------|-----:|-----------:|:---------------------:|:--------------:|
| FD001  | CNN      | **15.1** |  360 | 87% | ~11.5 |
| FD002  | LightGBM | **20.1** | 3932 | 86% | ~14.5 |
| FD003  | LightGBM | **16.0** |  602 | 86% | ~12 |
| FD004  | LightGBM | **18.9** | 2189 | 92% | ~17 |

Two models are trained and the better one is kept per subset. On the
single-condition subsets the CNN reads the degradation trajectory well: on FD001
it cuts RMSE from LightGBM's 18.8 to 15.1. On the six-condition subsets LightGBM
with regime normalization wins, because a window that keeps jumping between six
flight conditions is hard for the sequence model even after normalization.
Regime normalization is what makes those subsets tractable at all: without it a
plain model on FD004 sits near 29 RMSE.

Honest standing versus the literature: these are solid numbers, on par with
classic methods, but a few points above the current records (roughly 11 to 17
across the four subsets). Those records come from larger, GPU-trained
architectures (multi-scale CNNs, attention, transformers) with heavy tuning. What
this project offers instead is a correct, honest, end-to-end system where every
hyperparameter is chosen on training data alone, so the scores are a fair estimate
of real performance rather than a tuned-on-test best case.

Regenerate any of it with `python -m rul.train --subset FDxxx` (LightGBM) or
`python -m rul.train_deep --subset FDxxx` (CNN).

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows; use source .venv/bin/activate elsewhere
pip install -r requirements-dev.txt    # exact pinned versions, matching the committed models
pip install -e . --no-deps             # the rul package itself

pytest                                 # run the tests
python -m rul.train --subset FD001     # retrain the LightGBM model (already committed)
python -m rul.train_deep --subset FD001 # train the CNN model (best accuracy; needs torch)
uvicorn rul.api:app --reload           # serve the API at http://127.0.0.1:8000/docs
```

### The API

`uvicorn rul.api:app` gives you `GET /health` for readiness, `POST /predict` for a
prediction, and `/docs` for the interactive OpenAPI page. You send a window of raw
cycles for one engine and the service builds the features itself, so a client never
has to reproduce the feature code. It answers with a point estimate and an interval.

```jsonc
// request: oldest cycle first, the more history the better
{ "unit_id": 42,
  "readings": [ { "cycle": 1, "op_setting_1": -0.0007, "op_setting_2": -0.0004,
                  "op_setting_3": 100.0, "sensor_1": 518.67, "sensor_2": 641.82,
                  "...": "... all 21 sensors ..." } ] }

// response
{ "unit_id": 42, "cycle": 45, "rul": 80.7,
  "rul_lower": 55.8, "rul_upper": 105.5,
  "confidence_level": 0.9, "n_cycles_used": 45, "model_version": "0.1.0+FD001" }
```

It returns 503 if no model is loaded and 422 if the readings are malformed.

### Docker

```bash
docker build -t rul-cmapss .
docker run --rm -p 8000:8000 rul-cmapss    # http://127.0.0.1:8000/docs
```

The image is a multi-stage build on Debian slim. It has to be a glibc base rather
than Alpine, because the pinned numpy, scipy, pandas, scikit-learn and lightgbm
wheels are built for glibc; it installs `libgomp1` for LightGBM's OpenMP runtime,
runs as a non-root user, bakes in the trained model, and installs only the runtime
dependencies so the image stays lean.

### Deploy to Render

The container runs as a free web service on Render, which reads the `render.yaml`
blueprint at the repository root, builds the Dockerfile, and redeploys on every push
to `main`. The service listens on the `$PORT` Render provides and falls back to 8000
locally, so the image is the same on your laptop and in the cloud.

Steps, all on your own free account and with no credit card:

1. Sign up at render.com.
2. Choose New then Blueprint and connect this repository. Render picks up
   `render.yaml` and creates the service on the free plan.
3. Click Apply and wait for the first build to finish.

The service comes up at a public `onrender.com` URL, with the interactive API at the
`/docs` path. One thing to know about the free plan: it sleeps after about fifteen
minutes without traffic, so the first request after a quiet spell takes half a minute
or so to wake before it answers normally.

### The drift report

```bash
python -m rul.drift                          # FD001 train vs FD001 test
python -m rul.drift --current-subset FD002   # against a six-condition subset
```

Evidently compares the training data against incoming data on the exact columns
the model uses, and writes a full HTML report plus a small JSON summary to
`reports/`. On FD001 train versus test it reports most sensors as drifted, which is
expected and instructive: the test trajectories are cut off before failure, so they
simply contain fewer worn-out cycles. In production this same report is what would
tell you the incoming data has moved away from what the model was trained on.

## Testing and CI

```bash
pytest        # data, features, regime, evaluation, model, conformal, CNN, API, drift
ruff check .  # lint
```

Most tests run on small synthetic C-MAPSS-shaped data so they are fast and need no
files. A few integration tests use the committed FD001 data and model when they are
present and skip otherwise. GitHub Actions runs `ruff` and `pytest` on Python 3.14
for every push and pull request, installing the exact pinned versions so the
committed model loads cleanly.

## Checking for leakage and other mistakes

RUL projects on C-MAPSS have a handful of classic ways to fool yourself. These are
the ones I checked, and each is backed by a test rather than a promise.

Future information leaking into a prediction is the big one. Every feature at a
given cycle is built only from that cycle and earlier ones. There is a test that
proves it: the feature vector at cycle *t* is identical whether it is computed from
the trajectory truncated at *t* or from the full trajectory. All three feature
paths pass this (the plain builder, the regime builder, and the CNN windows),
which also means a single truncated snapshot at serving time gets exactly the
features it would have gotten live.

The normalization statistics and the variance filter are learned on training data
only. A test shifts an unseen frame and confirms it is not quietly recentered,
which would happen if test statistics had leaked in. Validation and calibration
splits are made by engine, so no engine ever appears on both sides of a split. The
label is a genuine piecewise-linear curve (flat at the cap, then a straight line to
zero), and the true test labels are compared without any capping. Finally, scoring
is one prediction per engine at its last cycle, lined up by engine id against the
ground truth, with nothing dropped.

## Limitations and what I would do next

- **A few points short of SOTA.** The current records are roughly 11 to 17 RMSE
  across the four subsets; this project is a few points above each. Closing that
  gap is the honest hard part.
- **My sequence model only helps the single-condition subsets.** The CNN wins
  FD001 but loses to LightGBM on the six-condition subsets: a window that keeps
  switching between six flight conditions is hard for it even after regime
  normalization. I also tried a bidirectional LSTM and adding a per-cycle
  condition signal, and neither beat LightGBM there. Reaching the six-condition
  records likely needs the heavier, GPU-trained architectures the top papers use
  (multi-scale CNNs, attention, transformers), which I did not attempt here.
- **Honest tuning has a cost.** Windows and caps come from a modest CV grid on
  training data. A wider search, an asymmetric loss aimed at the NASA score, or
  cross-conformal intervals (to reuse the calibration engines) would each help a
  little, at more compute.

## License

MIT for the code, see [LICENSE](LICENSE). The C-MAPSS dataset is a NASA work in the
public domain.
