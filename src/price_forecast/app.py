"""Flask application factory + routes.

Loads the latest model at startup, exposes prediction endpoints, and supports
admin-token-gated /reload and /trigger-train operations.

Production characteristics:
  - Graceful startup: if no pointer exists yet (first deploy), the app
    starts in "standby" mode and serves 503 on /predict until the
    background reloader picks up the freshly-promoted pointer. Avoids
    the chicken-and-egg deploy problem where the app can't start because
    the model isn't trained yet.
  - Schema validation: each /predict call cross-checks features against
    the manifest's schema_contract before invoking sklearn.
  - Request-size cap: enforced via Flask MAX_CONTENT_LENGTH.
  - /metrics: Prometheus-format counters and gauges.
  - Structured logs: JSON sink via LOG_FORMAT=json, default human-friendly.
  - CORS: configurable allow-list, prod refuses '*'.
"""

from __future__ import annotations

import json as _json
import sys
import threading
import time
import uuid
from functools import wraps
from typing import Any

import pandas as pd
import typer
from flask import Flask, Response, g, jsonify, request
from loguru import logger
from pydantic import ValidationError

from price_forecast.config import AppConfig, load_config
from price_forecast.loader import ModelStore
from price_forecast.metrics import Metrics
from price_forecast.publisher import publish_trigger
from price_forecast.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    TriggerTrainRequest,
    TriggerTrainResponse,
)
from price_forecast.validation import SchemaValidationError, validate_features


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def _configure_logging(cfg: AppConfig) -> None:
    """Reconfigure loguru once at startup. Idempotent.

    LOG_FORMAT=json -> single-line JSON per record (for log aggregators).
    LOG_FORMAT=""   -> default human-readable (default for dev).
    """
    logger.remove()
    if cfg.log_format == "json":
        def _json_sink(message):
            record = message.record
            payload = {
                "ts": record["time"].isoformat(),
                "level": record["level"].name,
                "msg": record["message"],
                "app_id": cfg.app_id,
                "module": record["name"],
            }
            if record["exception"]:
                payload["exception"] = str(record["exception"].value)
            sys.stdout.write(_json.dumps(payload) + "\n")
            sys.stdout.flush()
        logger.add(_json_sink, level="INFO")
    else:
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> "
                   "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def _require_admin(cfg: AppConfig):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not cfg.admin_token:
                return jsonify(error="admin endpoints disabled (APP_ADMIN_TOKEN not set)"), 503
            token = request.headers.get("X-Admin-Token", "")
            if token != cfg.admin_token:
                return jsonify(error="invalid admin token"), 401
            return view(*args, **kwargs)
        return wrapped
    return decorator


# ---------------------------------------------------------------------------
# CORS (minimal — no flask-cors dep)
# ---------------------------------------------------------------------------

def _attach_cors(flask_app: Flask, allowed: str) -> None:
    if not allowed:
        return
    origins = [o.strip() for o in allowed.split(",") if o.strip()]

    @flask_app.after_request
    def _add_cors_headers(response):
        origin = request.headers.get("Origin", "")
        if origin and (origin in origins or "*" in origins):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Admin-Token, X-Request-Id"
            response.headers["Vary"] = "Origin"
        return response

    @flask_app.before_request
    def _handle_options():
        if request.method == "OPTIONS":
            return ("", 204)


# ---------------------------------------------------------------------------
# Background reloader
# ---------------------------------------------------------------------------

def _start_background_reloader(
    store: ModelStore,
    metrics: Metrics,
    interval_s: int,
) -> threading.Thread:
    """Periodic best-effort reload. Survives transient failures (logged at WARNING).

    Exits the loop only on process termination (daemon thread). Each
    iteration increments metrics so an outside observer can detect a
    stuck reloader via reload_total / reload_errors_total.
    """
    def _loop():
        while True:
            time.sleep(interval_s)
            try:
                store.reload()
                metrics.inc_reload(ok=True)
                metrics.set_model_loaded(True)
            except LookupError as exc:
                metrics.inc_reload(ok=False)
                logger.info("Reload: pointer still absent: {}", exc)
            except Exception as exc:
                metrics.inc_reload(ok=False)
                logger.warning("Background reload failed: {}", exc)

    t = threading.Thread(target=_loop, daemon=True, name="model-reloader")
    t.start()
    return t


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(cfg: AppConfig | None = None, store: ModelStore | None = None) -> Flask:
    cfg = cfg or load_config()
    _configure_logging(cfg)
    metrics = Metrics()

    store = store or ModelStore(cfg)
    # Graceful startup: if the pointer is absent (first deploy, training
    # hasn't produced a stable.json yet), do NOT crash. Background reloader
    # will pick it up once it appears; /predict returns 503 in the meantime.
    initial = store.try_reload()
    if initial is not None:
        metrics.set_model_loaded(True)
        logger.info("Boot: loaded model {} at startup.", initial.version_id)
    else:
        logger.warning(
            "Boot: no pointer at s3://{}/{}/output/registry/{}/pointers/{}.json — "
            "starting in standby. Background reloader will retry every {}s "
            "for up to {}s before logs escalate.",
            cfg.bucket, cfg.stack_id, cfg.model_name, cfg.channel,
            cfg.reload_interval_s, cfg.startup_grace_seconds,
        )

    if cfg.reload_interval_s > 0:
        _start_background_reloader(store, metrics, cfg.reload_interval_s)

    flask_app = Flask(__name__)
    flask_app.config["APP_CFG"] = cfg
    flask_app.config["MODEL_STORE"] = store
    flask_app.config["METRICS"] = metrics
    flask_app.config["MAX_CONTENT_LENGTH"] = cfg.max_request_bytes

    _attach_cors(flask_app, cfg.cors_allowed_origins)

    # ----- request lifecycle hooks (structured logging) -----

    @flask_app.before_request
    def _before():
        g.request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        g.start_ns = time.perf_counter_ns()

    @flask_app.after_request
    def _after(response):
        try:
            elapsed_ms = (time.perf_counter_ns() - g.start_ns) / 1e6
            response.headers["X-Request-Id"] = g.request_id
            loaded = store.current_or_none()
            response.headers["X-Model-Version"] = loaded.version_id if loaded else "none"
            log_fn = logger.debug if request.path in ("/health", "/ready", "/metrics") else logger.info
            log_fn(
                "{} {} -> {} in {:.1f}ms rid={} mv={} app={}",
                request.method, request.path, response.status_code, elapsed_ms,
                g.request_id, response.headers["X-Model-Version"], cfg.app_id,
            )
        except Exception as exc:
            logger.warning("request logging hook failed: {}", exc)
        return response

    # ----- payload-too-large handler -----

    @flask_app.errorhandler(413)
    def _too_large(_exc):
        return jsonify(error=f"request body exceeds APP_MAX_REQUEST_BYTES={cfg.max_request_bytes}"), 413

    # ----- routes -----

    @flask_app.get("/health")
    def health():
        return jsonify(status="ok")

    @flask_app.get("/ready")
    def ready():
        loaded = store.current_or_none()
        if loaded is None:
            return jsonify(status="standby", reason="no model loaded yet"), 503
        return jsonify(status="ready", version=loaded.version_id)

    @flask_app.get("/metrics")
    def prometheus_metrics():
        loaded = store.current_or_none()
        body = metrics.render(
            app_id=cfg.app_id,
            model_version=loaded.version_id if loaded else "none",
        )
        return Response(body, mimetype="text/plain; version=0.0.4")

    @flask_app.get("/model/info")
    def model_info():
        loaded = store.current_or_none()
        if loaded is None:
            return jsonify(error="no model loaded yet"), 503
        body = ModelInfoResponse(
            version_id=loaded.version_id,
            run_id=loaded.manifest.run_id,
            registry_version=loaded.manifest.registry_version,
            model_name=loaded.manifest.model_name,
            model_type=loaded.manifest.model_type,
            promoted_at=loaded.pointer.promoted_at,
            loaded_at=loaded.loaded_at,
            channel=cfg.channel,
            schema_contract=loaded.manifest.schema_contract,
        )
        return jsonify(body.model_dump())

    @flask_app.post("/predict")
    def predict():
        t0 = time.perf_counter_ns()
        try:
            payload = PredictRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            metrics.inc_predict(0.0, ok=False)
            return jsonify(error=exc.errors()), 400

        loaded = store.current_or_none()
        if loaded is None:
            metrics.inc_predict(0.0, ok=False)
            return jsonify(error="model not loaded yet (standby)"), 503

        try:
            validate_features(payload.features, loaded.manifest.schema_contract, strict=cfg.strict_schema)
        except SchemaValidationError as exc:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            metrics.inc_predict(elapsed_ms, ok=False, schema_error=True)
            return jsonify(exc.to_dict()), 400

        df = pd.DataFrame([payload.features])
        try:
            pred = float(loaded.model.predict(df)[0])
        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            metrics.inc_predict(elapsed_ms, ok=False)
            logger.exception("predict failed")
            return jsonify(error=f"prediction failed: {exc}"), 500

        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        metrics.inc_predict(elapsed_ms, ok=True)
        return jsonify(PredictResponse(prediction=pred, model_version=loaded.version_id).model_dump())

    @flask_app.post("/predict/batch")
    def predict_batch():
        try:
            payload = BatchPredictRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            metrics.inc_batch(ok=False)
            return jsonify(error=exc.errors()), 400

        if len(payload.rows) > cfg.max_batch_size:
            metrics.inc_batch(ok=False)
            return jsonify(
                error=f"batch size {len(payload.rows)} exceeds APP_MAX_BATCH_SIZE={cfg.max_batch_size}"
            ), 400

        loaded = store.current_or_none()
        if loaded is None:
            metrics.inc_batch(ok=False)
            return jsonify(error="model not loaded yet (standby)"), 503

        # Schema validation: check the first row's keys against the contract
        # (assume homogeneity — all rows share columns). Cheap; catches the
        # common case of a typo in the caller's request schema.
        if payload.rows:
            try:
                validate_features(payload.rows[0], loaded.manifest.schema_contract, strict=cfg.strict_schema)
            except SchemaValidationError as exc:
                metrics.inc_batch(ok=False)
                return jsonify(exc.to_dict()), 400

        df = pd.DataFrame(payload.rows)
        try:
            preds = [float(p) for p in loaded.model.predict(df)]
        except Exception as exc:
            metrics.inc_batch(ok=False)
            logger.exception("batch predict failed")
            return jsonify(error=f"batch prediction failed: {exc}"), 500

        metrics.inc_batch(ok=True)
        return jsonify(
            BatchPredictResponse(predictions=preds, model_version=loaded.version_id).model_dump()
        )

    @flask_app.post("/reload")
    @_require_admin(cfg)
    def reload_model():
        try:
            loaded = store.reload()
        except LookupError as exc:
            metrics.inc_reload(ok=False)
            return jsonify(error=str(exc)), 404
        except Exception as exc:
            metrics.inc_reload(ok=False)
            logger.exception("manual reload failed")
            return jsonify(error=f"reload failed: {exc}"), 500
        metrics.inc_reload(ok=True)
        metrics.set_model_loaded(True)
        return jsonify(reloaded_to=loaded.version_id)

    @flask_app.post("/trigger-train")
    @_require_admin(cfg)
    def trigger_train():
        try:
            payload = TriggerTrainRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            return jsonify(error=exc.errors()), 400
        try:
            trigger_id, uri = publish_trigger(
                payload.dataset_path,
                payload.params_path,
                model_family=payload.model_family,
                description=payload.description,
                requested_by=request.headers.get("X-User", ""),
            )
        except (FileNotFoundError, OSError) as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(
            TriggerTrainResponse(trigger_id=trigger_id, trigger_uri=uri).model_dump()
        )

    return flask_app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

cli_app = typer.Typer()


@cli_app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
) -> None:
    """Start the Flask development server (use gunicorn in production)."""
    cfg = load_config()
    flask_app = create_app(cfg)
    flask_app.run(host=host or cfg.host, port=port or cfg.port)


def cli() -> None:
    cli_app()


if __name__ == "__main__":
    cli()
