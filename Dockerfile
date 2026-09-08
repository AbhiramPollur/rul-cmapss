# syntax=docker/dockerfile:1
#
# Container image for the RUL-CMAPSS FastAPI service.
#
# Design notes
# ------------
# * Base is Debian slim (glibc), NOT Alpine: the pinned numpy/scipy/pandas/
#   scikit-learn/lightgbm wheels are manylinux_2_28 (glibc), which musl/Alpine
#   cannot use. Bookworm's glibc 2.36 satisfies that.
# * Two stages: deps are installed into a venv in the builder, then copied into
#   a clean runtime image so build metadata never ships.
# * Only requirements.txt (runtime deps) is installed — Evidently, pytest, etc.
#   stay out of the image, keeping it lean.
# * The trained model artifact is baked in and located via RUL_MODELS_DIR, so
#   the container serves predictions out of the box.

########################  Stage 1: builder  ########################
FROM python:3.14-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Runtime dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Then the package itself (deps already pinned above).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . --no-deps

########################  Stage 2: runtime  ########################
FROM python:3.14-slim-bookworm AS runtime

# libgomp1 = the OpenMP runtime LightGBM links against at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

# Prebuilt venv + trained artifact.
COPY --from=builder /opt/venv /opt/venv
COPY models/ /app/models/

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    RUL_MODELS_DIR=/app/models

USER appuser
EXPOSE 8000

# The service honours $PORT when the host provides one (Koyeb, Render and Cloud
# Run all inject it) and falls back to 8000 locally, matching EXPOSE above.
# slim has no curl, so the healthcheck uses the stdlib and reads the same port.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,sys,urllib.request; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health').status==200 else 1)"

CMD exec uvicorn rul.api:app --host 0.0.0.0 --port ${PORT:-8000}
