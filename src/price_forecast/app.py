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

import datetime
import hmac
import json as _json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from flask import Flask, Response, g, jsonify, make_response, render_template, request
from loguru import logger
from pydantic import ValidationError

from price_forecast.config import AppConfig, load_config
from price_forecast.factories import get_artifact_store
from price_forecast.layout import (
    pointer_key,
    trigger_failure_key,
    trigger_metadata_key,
    trigger_running_key,
)
from price_forecast.loader import ModelStore
from price_forecast.metrics import Metrics
from price_forecast.ports.storage import ReadOnlyArtifactStore
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
                "project": cfg.project,
                "model_name": cfg.model_name,
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
                return jsonify(error="admin endpoints disabled (APP_ADMIN_TOKEN not set)"), 501
            token = request.headers.get("X-Admin-Token", "")
            # hmac.compare_digest prevents timing attacks that could leak token length/prefix.
            if not hmac.compare_digest(token, cfg.admin_token):
                return jsonify(error="invalid admin token"), 401
            return view(*args, **kwargs)

        return wrapped

    return decorator


# ---------------------------------------------------------------------------
# Admin login page (returned when token is absent or wrong)
# ---------------------------------------------------------------------------

_ADMIN_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Admin Login — Price Forecast</title>
<style>
:root{--bg:#0f1117;--s:#1a1d27;--b:#2d3148;--t:#e2e8f0;--m:#94a3b8;--a:#6366f1;--r:#ef4444;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--t);font-family:system-ui,sans-serif;
     display:flex;align-items:center;justify-content:center;min-height:100vh;}
.box{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:32px;width:340px;}
h2{color:var(--a);font-size:18px;margin-bottom:8px;}
p.sub{color:var(--m);font-size:13px;margin-bottom:20px;}
.err{color:var(--r);font-size:13px;margin-bottom:14px;display:none;}
input{width:100%;padding:9px 12px;background:var(--bg);border:1px solid var(--b);
      color:var(--t);border-radius:6px;font-size:14px;margin-bottom:12px;}
button{width:100%;padding:10px;background:var(--a);color:#fff;border:none;
       border-radius:6px;font-size:14px;cursor:pointer;}
button:hover{opacity:.85;}
</style>
</head>
<body>
<div class="box">
  <h2>Price Forecast Admin</h2>
  <p class="sub">Enter your admin token to continue.</p>
  <p class="err" id="err">Invalid token — try again.</p>
  <input type="password" id="tok" placeholder="Admin token" autofocus>
  <button onclick="go()">Sign in</button>
</div>
<script>
if (sessionStorage.getItem("admin_auth_failed")) {
  document.getElementById("err").style.display = "block";
  sessionStorage.removeItem("admin_auth_failed");
}
document.getElementById("tok").addEventListener("keydown", function(e) {
  if (e.key === "Enter") go();
});
function go() {
  var v = document.getElementById("tok").value;
  if (!v) return;
  fetch("/admin/api/status", {headers: {"X-Admin-Token": v}})
    .then(function(r) {
      if (r.ok) {
        sessionStorage.setItem("admin_token", v);
        window.location = "/admin";
      } else {
        sessionStorage.setItem("admin_auth_failed", "1");
        document.getElementById("err").style.display = "block";
      }
    })
    .catch(function() {
      document.getElementById("err").style.display = "block";
    });
}
</script>
</body>
</html>"""


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
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, X-Admin-Token, X-Request-Id"
            )
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
    stop_event: threading.Event | None = None,
) -> threading.Thread:
    """Periodic best-effort reload. Survives transient failures (logged at WARNING).

    Stops cleanly when stop_event is set (e.g. during test teardown or graceful
    shutdown). Without a stop_event the thread runs for the lifetime of the process.
    """

    def _loop():
        while stop_event is None or not stop_event.is_set():
            if stop_event is not None:
                stop_event.wait(timeout=interval_s)
                if stop_event.is_set():
                    break
            else:
                time.sleep(interval_s)
            try:
                store.reload()
                metrics.inc_reload(ok=True)
                metrics.set_model_loaded(True)
            except LookupError as exc:
                metrics.inc_reload(ok=False)
                logger.info("Reload: pointer still absent: {}", exc)
            except RuntimeError as exc:
                # Integrity failures: manifest missing, checksum mismatch, scope mismatch.
                # Log at ERROR so alerting picks it up; keep serving the current (stale) model.
                metrics.inc_reload(ok=False)
                logger.error("Reload: integrity failure — continuing with stale model: {}", exc)
            except Exception as exc:
                # Transient S3 / network errors — will self-heal on the next cycle.
                metrics.inc_reload(ok=False)
                logger.warning("Reload: transient failure (will retry in {}s): {}", interval_s, exc)

    t = threading.Thread(target=_loop, daemon=True, name="model-reloader")
    t.start()
    return t


# ---------------------------------------------------------------------------
# S3 download helper (multi-user trigger flow)
# ---------------------------------------------------------------------------


def _store_download_to_dir(s3_uri: str, suffix: str, dest_dir: Path, cfg: AppConfig) -> Path:
    """Download a user-supplied s3://bucket/key URI to a temp file inside dest_dir.

    The URI's bucket may differ from cfg.bucket (user-provided), so a
    one-shot ReadOnlyArtifactStore is built for it via the storage factory.
    suffix is the file extension including the dot.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {s3_uri!r}")
    bucket, _, key = s3_uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed s3 URI: {s3_uri!r}")

    from price_forecast.factories import get_artifact_store

    one_shot_cfg = replace_app_config(cfg, bucket=bucket, prefix="")
    transient = get_artifact_store(one_shot_cfg)
    with tempfile.NamedTemporaryFile(dir=dest_dir, suffix=suffix, delete=False) as tmp:
        dest = Path(tmp.name)
    try:
        transient.download_file(key, dest)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return dest


def replace_app_config(cfg: AppConfig, **overrides) -> AppConfig:
    from dataclasses import replace

    return replace(cfg, **overrides)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    cfg: AppConfig | None = None,
    store: ModelStore | None = None,
    artifact_store: ReadOnlyArtifactStore | None = None,
) -> Flask:
    cfg = cfg or load_config()
    _configure_logging(cfg)
    metrics = Metrics()

    if artifact_store is None:
        artifact_store = get_artifact_store(cfg)
    if store is None:
        store = ModelStore(cfg, artifact_store)
    # Graceful startup: if the pointer is absent (first deploy, training
    # hasn't produced a stable.json yet), do NOT crash. Background reloader
    # will pick it up once it appears; /predict returns 503 in the meantime.
    try:
        initial = store.try_reload()
    except Exception as exc:
        initial = None
        logger.warning("Boot: initial model load failed; continuing in standby: {}", exc)
    if initial is not None:
        metrics.set_model_loaded(True)
        logger.info("Boot: loaded model {} at startup.", initial.version_id)
    else:
        logger.warning(
            "Boot: no pointer at s3://{}/{}/{}/{}.json — "
            "starting in standby. Background reloader will retry every {}s "
            "for up to {}s before logs escalate.",
            cfg.bucket,
            cfg.project,
            cfg.model_name,
            cfg.channel,
            cfg.reload_interval_s,
            cfg.startup_grace_seconds,
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
            log_fn = (
                logger.debug if request.path in ("/health", "/ready", "/metrics") else logger.info
            )
            log_fn(
                "{} {} -> {} in {:.1f}ms rid={} mv={} app={}",
                request.method,
                request.path,
                response.status_code,
                elapsed_ms,
                g.request_id,
                response.headers["X-Model-Version"],
                f"{cfg.project}/{cfg.model_name}",
            )
        except Exception as exc:
            logger.warning("request logging hook failed: {}", exc)
        return response

    # ----- payload-too-large handler -----

    @flask_app.errorhandler(413)
    def _too_large(_exc):
        return jsonify(
            error=f"request body exceeds APP_MAX_REQUEST_BYTES={cfg.max_request_bytes}"
        ), 413

    # ----- routes -----

    @flask_app.get("/health")
    def health():
        return jsonify(status="ok")

    @flask_app.get("/ready")
    def ready():
        """Ready iff the configured target is loaded and reflects the live pointer.

        - process_alive: always true once Flask is serving.
        - model_loaded:  true once try_reload() or background reloader has produced
          a LoadedModel.
        - target:        the (category, project, model_name, channel) this app
          is configured to serve, so dashboards can disambiguate which target
          a 503 refers to in multi-tenant deployments.
        """
        loaded = store.current_or_none()
        body = {
            "process_alive": True,
            "model_loaded": loaded is not None,
            "target": {
                "category": cfg.category,
                "project": cfg.project,
                "model_name": cfg.model_name,
                "channel": cfg.channel,
            },
            "version": loaded.version_id if loaded else None,
        }
        if loaded is None:
            body["status"] = "standby"
            body["reason"] = "no model loaded yet"
            return jsonify(body), 503
        body["status"] = "ready"
        return jsonify(body)

    @flask_app.get("/metrics")
    def prometheus_metrics():
        loaded = store.current_or_none()
        body = metrics.render(
            project=cfg.project,
            model_name=cfg.model_name,
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
            validate_features(
                payload.features, loaded.manifest.schema_contract, strict=cfg.strict_schema
            )
        except SchemaValidationError as exc:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            metrics.inc_predict(elapsed_ms, ok=False, schema_error=True)
            return jsonify(exc.to_dict()), 400

        df = pd.DataFrame([payload.features])
        feature_cols = loaded.manifest.schema_contract.get("feature_columns")
        if feature_cols:
            df = df[feature_cols]
        try:
            pred = float(loaded.model.predict(df)[0])
        except Exception as exc:
            elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
            metrics.inc_predict(elapsed_ms, ok=False)
            logger.exception("predict failed")
            return jsonify(error=f"prediction failed: {exc}"), 500

        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        metrics.inc_predict(elapsed_ms, ok=True)
        return jsonify(
            PredictResponse(prediction=pred, model_version=loaded.version_id).model_dump()
        )

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

        # Schema validation must cover every row. Checking only the first row
        # allows malformed later rows to slip through and fail inside predict().
        for row in payload.rows:
            try:
                validate_features(row, loaded.manifest.schema_contract, strict=cfg.strict_schema)
            except SchemaValidationError as exc:
                metrics.inc_batch(ok=False, schema_error=True)
                return jsonify(exc.to_dict()), 400

        df = pd.DataFrame(payload.rows)
        feature_cols = loaded.manifest.schema_contract.get("feature_columns")
        if feature_cols:
            df = df[feature_cols]
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

        # Resolve dataset_path and params_path — either use what caller gave us
        # (local mode) or download from S3 into TRIGGER_DATA_ROOT (S3 mode).
        trigger_data_root = os.environ.get("TRIGGER_DATA_ROOT", "").strip()
        tmp_files: list[Path] = []

        try:
            if payload.dataset_s3_uri and payload.params_s3_uri:
                # ── S3 mode: user provided URIs; download to a temp directory ──
                dest_dir = (
                    Path(trigger_data_root) if trigger_data_root else Path(tempfile.gettempdir())
                )
                dest_dir.mkdir(parents=True, exist_ok=True)

                dataset_ext = "." + payload.dataset_s3_uri.rsplit(".", 1)[-1].lower()
                dataset_path = _store_download_to_dir(
                    payload.dataset_s3_uri, dataset_ext, dest_dir, cfg
                )
                tmp_files.append(dataset_path)

                params_path = _store_download_to_dir(payload.params_s3_uri, ".yaml", dest_dir, cfg)
                tmp_files.append(params_path)
            else:
                # ── Local mode: paths already validated by TriggerTrainRequest ──
                dataset_path = Path(payload.dataset_path)  # type: ignore[arg-type]
                params_path = Path(payload.params_path)  # type: ignore[arg-type]

            trigger_id, uri = publish_trigger(
                dataset_path,
                params_path,
                model_family=payload.model_family,
                description=payload.description,
                requested_by=request.headers.get("X-User", ""),
                dataset_format=payload.dataset_format,
                cfg=cfg,
            )
        except (FileNotFoundError, ValueError) as exc:
            return jsonify(error=str(exc)), 400
        except RuntimeError as exc:
            logger.error("trigger-train failed: {}", exc)
            return jsonify(error=str(exc)), 502
        except OSError as exc:
            return jsonify(error=str(exc)), 400
        finally:
            for tmp in tmp_files:
                tmp.unlink(missing_ok=True)

        return jsonify(TriggerTrainResponse(trigger_id=trigger_id, trigger_uri=uri).model_dump())

    _TRIGGER_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z_[0-9a-f]{8}$")

    @flask_app.get("/trigger-status/<trigger_id>")
    @_require_admin(cfg)
    def trigger_status(trigger_id: str):
        """Poll whether a triggered training run has produced a new promoted version.

        Query parameters:
          baseline (optional): version_id that was current *before* the trigger was
            created (from GET /model/info). When provided, "completed" means the
            pointer's version_id has changed away from this baseline — exact and
            reliable regardless of unrelated promotions that happened between.
            Without this param, completion falls back to timestamp comparison
            (less precise: a different trigger's promotion can satisfy it).

        Returns:
          status: "completed" | "running" | "failed" | "pending"
            - "pending"   — trigger.json exists but running.json not yet written (dispatch enqueued)
            - "running"   — training job has started (running.json present, not yet done)
            - "failed"    — training job failed (failed.json written by the CI job on failure)
            - "completed" — model promoted; pointer has moved past baseline or updated_at timestamp
          current_version_id: the version_id in the current pointer
          current_version_updated_at: ISO timestamp from the pointer
        """
        if not _TRIGGER_ID_RE.match(trigger_id):
            return jsonify(error="invalid trigger_id format"), 400

        trigger_data = artifact_store.get_json(trigger_metadata_key(cfg.project, trigger_id))
        if trigger_data is None:
            return jsonify(error=f"trigger {trigger_id!r} not found"), 404

        # State machine (checked in priority order):
        #   failed.json present  → "failed"  (terminal; stop polling)
        #   running.json present → "running" (transient; keep polling)
        #   neither              → "pending" (dispatch enqueued, worker not yet started)
        # "completed" is checked last after pointer comparison.
        failure_data = artifact_store.get_json(trigger_failure_key(cfg.project, trigger_id))
        if failure_data is not None:
            return jsonify(
                trigger_id=trigger_id,
                status="failed",
                reason=failure_data.get("reason", "training job failed"),
                current_version_id=None,
                current_version_updated_at=None,
            )

        running_data = artifact_store.get_json(trigger_running_key(cfg.project, trigger_id))

        baseline_version_id = request.args.get("baseline", "").strip() or None

        pointer_data = artifact_store.get_json(
            pointer_key(cfg.project, cfg.model_name, cfg.channel)
        )
        if pointer_data is None:
            return jsonify(
                trigger_id=trigger_id,
                status="running" if running_data is not None else "pending",
                reason="model not yet promoted",
                current_version_id=None,
                current_version_updated_at=None,
            )

        current_version_id = pointer_data.get("version_id", "")
        updated_at_str = pointer_data.get("updated_at", "")

        if baseline_version_id is not None:
            completed = bool(current_version_id and current_version_id != baseline_version_id)
        else:
            # Fallback: compare pointer timestamp against trigger creation time.
            ts_str = trigger_id[:16]
            try:
                trigger_created_at = datetime.datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=datetime.timezone.utc
                )
            except ValueError:
                return jsonify(error="cannot parse trigger_id timestamp"), 400
            completed = False
            if updated_at_str:
                try:
                    updated_at = datetime.datetime.fromisoformat(updated_at_str)
                    completed = updated_at > trigger_created_at
                except ValueError:
                    pass

        if completed:
            status = "completed"
        elif running_data is not None:
            status = "running"
        else:
            status = "pending"

        return jsonify(
            trigger_id=trigger_id,
            status=status,
            current_version_id=current_version_id,
            current_version_updated_at=updated_at_str,
        )

    @flask_app.get("/admin")
    def admin_dashboard():
        if not cfg.admin_token:
            return jsonify(error="admin endpoints disabled (APP_ADMIN_TOKEN not set)"), 501
        token = request.headers.get("X-Admin-Token", "")
        if not token or not hmac.compare_digest(token, cfg.admin_token):
            return Response(_ADMIN_LOGIN_HTML, mimetype="text/html")
        resp = make_response(
            render_template("admin.html", token=token, app_id=f"{cfg.project}/{cfg.model_name}")
        )
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"
        )
        return resp

    @flask_app.get("/admin/api/status")
    @_require_admin(cfg)
    def admin_status():
        loaded = store.current_or_none()
        m = metrics
        avg_lat = (
            m.predict_latency_sum_ms / m.predict_latency_count
            if m.predict_latency_count > 0
            else 0.0
        )
        model_section: dict[str, Any] = {"status": "standby"}
        if loaded is not None:
            loaded_iso = datetime.datetime.fromtimestamp(
                loaded.loaded_at, tz=datetime.timezone.utc
            ).isoformat()
            model_section = {
                "status": "ready",
                "version_id": loaded.version_id,
                "run_id": loaded.manifest.run_id,
                "registry_version": loaded.manifest.registry_version,
                "model_name": loaded.manifest.model_name,
                "model_type": loaded.manifest.model_type,
                "promoted_at": loaded.pointer.promoted_at,
                "loaded_at": loaded_iso,
                "channel": cfg.channel,
                "schema_contract": loaded.manifest.schema_contract,
            }
        return jsonify(
            {
                "model": model_section,
                "app": {
                    "category": cfg.category,
                    "project": cfg.project,
                    "model_name": cfg.model_name,
                    "env": cfg.env,
                    "bucket": cfg.bucket,
                    "prefix": cfg.prefix,
                    "channel": cfg.channel,
                },
                "metrics": {
                    "predict_total": m.predict_total,
                    "predict_errors": m.predict_errors,
                    "predict_schema_errors": m.predict_schema_errors,
                    "predict_latency_count": m.predict_latency_count,
                    "batch_predict_total": m.batch_predict_total,
                    "batch_predict_errors": m.batch_predict_errors,
                    "reload_total": m.reload_total,
                    "reload_errors": m.reload_errors,
                    "model_loaded": m.model_loaded,
                    "last_reload_unixtime": m.last_reload_unixtime,
                    "predict_avg_latency_ms": round(avg_lat, 2),
                },
                "server_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )

    return flask_app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

cli_app = typer.Typer()


@cli_app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),  # noqa: B008, UP045
    port: int | None = typer.Option(None, "--port"),  # noqa: B008, UP045
) -> None:
    """Start the Flask development server (use gunicorn in production)."""
    cfg = load_config()
    flask_app = create_app(cfg)
    flask_app.run(host=host or cfg.host, port=port or cfg.port)


def cli() -> None:
    cli_app()


if __name__ == "__main__":
    cli()
