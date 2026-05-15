# price-forecast

Independent serving layer for the price forecasting model. Pulls trained
model artifacts from S3 (written by the MLOps training repo) and exposes
prediction APIs.

## Architecture

This repo is **decoupled from the training stack** — it has zero MLflow
dependency. It reads only the published S3 contract, scoped by `APP_ID`:

```
s3://{bucket}/{stack_id}/output/registry/{APP_ID}/{model_name}/pointers/stable.json
                                                                       └─ points to immutable
s3://{bucket}/{stack_id}/output/artifacts/{APP_ID}/v{N}/champion/model.pkl
```

The training repo writes `stable.json` after a successful promotion; this app
re-reads `stable.json` periodically (or on demand via `/reload`) and swaps in
the new model. Two guarantees:

1. **Checksum verification.** SHA-256 of every downloaded `model.pkl` is
   compared to the manifest before `pickle.load`. A corrupted artifact is
   refused.
2. **App-scope verification.** Both `pointer.app_id` and `manifest.app_id`
   are cross-checked against this app's configured `APP_ID`. A payload
   claiming a different scope is refused — serving the wrong model is
   worse than serving nothing.

## Endpoints

| Method | Path              | Description                                                            |
|--------|-------------------|------------------------------------------------------------------------|
| GET    | `/health`         | Liveness probe (always 200 if process is up)                           |
| GET    | `/ready`          | Readiness probe (503 in standby — no model loaded yet)                 |
| GET    | `/metrics`        | Prometheus-format counters and gauges                                  |
| GET    | `/model/info`     | Currently-loaded version metadata                                      |
| POST   | `/predict`        | Single prediction (validates features against manifest.schema_contract)|
| POST   | `/predict/batch`  | Batch predictions (capped at `APP_MAX_BATCH_SIZE`)                     |
| POST   | `/reload`         | Re-read pointer and reload (admin token)                               |
| POST   | `/trigger-train`  | Push dataset + params to S3 to kick off training (admin token)         |

Every response carries `X-Request-Id` (echoes the inbound header if present, else uuid) and `X-Model-Version`. Every request emits one structured log line with method, path, status, latency, request_id, model_version, app_id.

## Configuration

Required:

```
APP_ID=app1                          # logical app identifier (multi-tenant scope)
APP_S3_BUCKET=app1_bucket            # this app's bucket
APP_MODEL_NAME=price_forecast        # registered model name written by training
APP_ADMIN_TOKEN=...                  # required for /reload, /trigger-train (and required in prod)
AWS_DEFAULT_REGION=us-east-1
```

Optional (with defaults):

```
STACK_ID=MLOPS                       # tree prefix; "MLOPS" today, "azure" for parallel stacks
APP_CHANNEL=stable                   # which pointer to follow (stable | canary | latest)
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD_INTERVAL_S=30             # background pointer-poll interval (0 disables)
APP_STARTUP_GRACE_SECONDS=120        # standby tolerance window logged at boot
APP_MAX_BATCH_SIZE=1000              # /predict/batch row cap
APP_MAX_REQUEST_BYTES=1048576        # 413 if request body exceeds
APP_CORS_ALLOWED_ORIGINS=            # comma-separated; "*" refused in prod
APP_STRICT_SCHEMA=1                  # reject requests with unknown columns
ENV=prod                             # prod enforces APP_ADMIN_TOKEN and no-wildcard CORS
LOG_FORMAT=json                      # JSON sink for log aggregators (default: human)
```

## Production behavior worth knowing

- **Graceful startup.** If no `stable.json` exists yet (first deploy, before training has ever promoted), the app starts in standby mode: `/health` is 200, `/ready` is 503, `/predict` returns 503. The background reloader keeps polling; once the pointer arrives, the app flips to serving without a restart.
- **Atomic model swap.** Reloads happen out-of-band. `/predict` requests in flight during a swap serve the previous model; new requests after the swap serve the new model. There's no in-between state.
- **Checksum verification.** Every artifact download cross-checks `sha256(model.pkl)` against the manifest. A corrupted byte stream is refused before `pickle.load`.
- **Retry/backoff.** Pointer + manifest + pkl reads retry on 5xx / throttling with exponential backoff (4 attempts, base 0.5s + jitter). A transient S3 blip during reload doesn't propagate to callers.
- **Schema validation.** `/predict` cross-checks request features against the manifest's `schema_contract.feature_columns` before invoking the model. Missing / extra / null-required columns return 400 with explicit field lists.

## Local development

```
pip install -e ".[dev]"
price-forecast-serve --port 8000
```

## Deployment

This repo has its own Dockerfile and CI/CD. It does **not** share a
container, requirements file, or release cadence with the training repo.
The only thing both repos must agree on is the S3 contract — see
`src/price_forecast/contracts.py` (copied verbatim from the training repo's
`contracts.py`). If you change one, change both and bump
`SCHEMA_VERSION`.
