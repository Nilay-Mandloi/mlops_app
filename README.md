# price-forecast

Independent serving layer for the price forecasting model. Pulls trained
model artifacts from S3 (written by the MLOps training repo) and exposes
prediction APIs.

## Architecture

This repo is **decoupled from the training stack** — it has zero MLflow
dependency. It reads only the published S3 contract:

```
s3://{app_bucket}/{stack_id}/output/registry/{model_name}/pointers/stable.json
                                                                  └─ points to immutable
s3://{app_bucket}/{stack_id}/output/artifacts/v{N}/champion/model.pkl
```

The training repo writes `stable.json` after a successful promotion; this app
re-reads `stable.json` periodically (or on demand via `/reload`) and swaps in
the new model. Checksums in the artifact manifest are verified before any
swap — corrupted artifacts are refused.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET    | `/health`         | Liveness probe |
| GET    | `/ready`          | Readiness probe (model loaded) |
| GET    | `/model/info`     | Currently-loaded version metadata |
| POST   | `/predict`        | Single prediction |
| POST   | `/predict/batch`  | Batch predictions |
| POST   | `/reload`         | Re-read `stable.json` and reload (admin token) |
| POST   | `/trigger-train`  | Push dataset + params to S3 to kick off training (admin token) |

## Configuration

Set via env vars:

```
APP_ID=app1                       # logical app identifier (multi-tenant scope)
APP_S3_BUCKET=app1_bucket         # this app's bucket
STACK_ID=MLOPS                    # which stack's tree to read (MLOPS today, azure later)
APP_MODEL_NAME=price_forecast     # registered model name written by training
APP_CHANNEL=stable                # which pointer to follow (stable | canary | latest)
APP_ADMIN_TOKEN=...               # required for /reload and /trigger-train
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

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
