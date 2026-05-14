ARG PYTHON_VERSION=3.11.9

# ---------------------------------------------------------------------------
# Builder: install deps into a venv we can copy whole into the runtime stage.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies before copying source so the resolver layer is cached
# across source-only changes.
COPY pyproject.toml ./
RUN pip install --upgrade pip wheel \
    && pip install \
        flask>=3.0 \
        boto3>=1.34 \
        joblib>=1.3 \
        scikit-learn>=1.4 \
        pandas>=2.0 \
        pydantic>=2.5 \
        typer>=0.9 \
        loguru>=0.7 \
        gunicorn>=22.0

# Now copy source + install the package (without re-resolving deps).
COPY src ./src
RUN pip install --no-deps -e .


# ---------------------------------------------------------------------------
# Runtime: slim image with the venv + source baked in. No build toolchain.
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="price-forecast" \
      org.opencontainers.image.description="Independent serving layer for the price forecasting model."

# Non-root user — best practice; gunicorn doesn't need root.
RUN groupadd --system --gid 1001 app \
    && useradd --system --uid 1001 --gid app --home-dir /app --shell /bin/false app

# Runtime libs required by sklearn/pandas (libgomp for sklearn parallelism;
# libpq for any psycopg dependency pulled transitively).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /build/src /app/src
COPY --from=builder /build/pyproject.toml /app/pyproject.toml

WORKDIR /app
USER app:app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

EXPOSE 8000

# Container-level readiness check — orchestrator (k8s/ECS) probes /ready
# directly, but this is a useful local sanity check for `docker run`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail --silent --max-time 4 http://localhost:${APP_PORT}/ready || exit 1

# Gunicorn config: 2 sync workers by default (override via GUNICORN_CMD_ARGS).
# Sync is right for sklearn/pandas — they're CPU-bound and don't release the GIL.
# Use --preload only if you've confirmed reload-after-fork is safe; we keep it
# off here because each worker creates its own ModelStore + S3 client.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60", \
     "--access-logfile", "-", "--error-logfile", "-", "price_forecast.app:create_app()"]
