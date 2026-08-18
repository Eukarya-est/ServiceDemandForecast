# syntax=docker/dockerfile:1.7
# Production image for the ASP Demand Prediction API.
# Multi-stage: build the locked venv with uv, then ship a slim CPU runtime
# suitable for pushing to a container registry (ECR/GCR/GHCR).
ARG PYTHON_VERSION=3.12

# ---- builder ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# 1) Install dependencies first (cached unless the lockfile changes).
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) Add the project source (Hydra configs live in asp_demand/conf) and install it.
COPY asp_demand ./asp_demand
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---- runtime ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# LightGBM needs the OpenMP runtime; curl is handy for healthchecks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder --chown=appuser:appuser /app /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default: serve the API. Override the command to run the pipeline, e.g.:
#   docker run <image> asp-demand-pipeline preprocess.start=2025-05-06 preprocess.end=2025-05-06
CMD ["uvicorn", "asp_demand.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
