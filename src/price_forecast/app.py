"""Flask application factory + routes.

Loads the latest model at startup, exposes prediction endpoints, and supports
admin-token-gated /reload and /trigger-train operations.
"""

from __future__ import annotations

import threading
import time
from functools import wraps
from typing import Any

import pandas as pd
import typer
from flask import Flask, jsonify, request
from loguru import logger
from pydantic import ValidationError

from price_forecast.config import AppConfig, load_config
from price_forecast.loader import ModelStore
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
# Background reloader
# ---------------------------------------------------------------------------

def _start_background_reloader(store: ModelStore, interval_s: int) -> threading.Thread:
    def _loop():
        while True:
            time.sleep(interval_s)
            try:
                store.reload()
            except Exception as exc:
                logger.warning("Background reload failed: {}", exc)

    t = threading.Thread(target=_loop, daemon=True, name="model-reloader")
    t.start()
    return t


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(cfg: AppConfig | None = None, store: ModelStore | None = None) -> Flask:
    cfg = cfg or load_config()
    store = store or ModelStore(cfg)
    store.reload()  # fail-fast: don't start serving without a model

    if cfg.reload_interval_s > 0:
        _start_background_reloader(store, cfg.reload_interval_s)

    flask_app = Flask(__name__)
    flask_app.config["APP_CFG"] = cfg
    flask_app.config["MODEL_STORE"] = store

    # ----- routes -----

    @flask_app.get("/health")
    def health():
        return jsonify(status="ok")

    @flask_app.get("/ready")
    def ready():
        try:
            loaded = store.current()
        except RuntimeError as exc:
            return jsonify(status="not_ready", error=str(exc)), 503
        return jsonify(status="ready", version=loaded.version_id)

    @flask_app.get("/model/info")
    def model_info():
        loaded = store.current()
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
        try:
            payload = PredictRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            return jsonify(error=exc.errors()), 400
        loaded = store.current()
        df = pd.DataFrame([payload.features])
        try:
            pred = float(loaded.model.predict(df)[0])
        except Exception as exc:
            logger.exception("predict failed")
            return jsonify(error=f"prediction failed: {exc}"), 500
        return jsonify(PredictResponse(prediction=pred, model_version=loaded.version_id).model_dump())

    @flask_app.post("/predict/batch")
    def predict_batch():
        try:
            payload = BatchPredictRequest.model_validate(request.get_json(silent=True) or {})
        except ValidationError as exc:
            return jsonify(error=exc.errors()), 400
        loaded = store.current()
        df = pd.DataFrame(payload.rows)
        try:
            preds = [float(p) for p in loaded.model.predict(df)]
        except Exception as exc:
            logger.exception("batch predict failed")
            return jsonify(error=f"batch prediction failed: {exc}"), 500
        return jsonify(
            BatchPredictResponse(predictions=preds, model_version=loaded.version_id).model_dump()
        )

    @flask_app.post("/reload")
    @_require_admin(cfg)
    def reload_model():
        loaded = store.reload()
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
    """Start the Flask development server (use gunicorn/uwsgi in production)."""
    cfg = load_config()
    flask_app = create_app(cfg)
    flask_app.run(host=host or cfg.host, port=port or cfg.port)


def cli() -> None:
    cli_app()


if __name__ == "__main__":
    cli()
