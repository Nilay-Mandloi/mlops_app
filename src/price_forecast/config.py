"""Runtime config for the serving app, loaded from environment.

The training repo's settings.py is intentionally NOT imported here. Both
repos own their own config surface independently.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")
_PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_MODEL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _require(name: str) -> str:
    value = _env(name)
    if not value:
        raise OSError(f"{name} environment variable is required.")
    return value


def _parse_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise OSError(f"{name} must be an integer; got {value!r}.") from None


@dataclass(frozen=True)
class AppConfig:
    category: str
    project: str
    model_name: str
    bucket: str
    prefix: str
    channel: str  # which pointer to follow: stable | canary | latest
    aws_region: str
    admin_token: str
    reload_interval_s: int
    host: str
    port: int
    max_batch_size: int
    max_request_bytes: int
    cors_allowed_origins: str
    env: str  # "dev" | "prod" | "test"
    log_format: str
    strict_schema: bool
    startup_grace_seconds: int
    training_repo: str
    training_repo_token: str
    training_auto_promote: bool


def _parse_bool(value: str, *, default: bool) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    channel = _env("APP_CHANNEL", "stable")
    if channel not in {"stable", "canary", "latest"}:
        raise OSError("APP_CHANNEL must be one of stable, canary, latest.")

    category = _require("CATEGORY")
    project = _require("PROJECT")
    model_name = _require("MODEL_NAME")
    if not _CATEGORY_RE.match(category):
        raise ValueError(f"CATEGORY must match {_CATEGORY_RE.pattern}; got {category!r}")
    if not _PROJECT_RE.match(project):
        raise ValueError(f"PROJECT must match {_PROJECT_RE.pattern}; got {project!r}")
    if not _MODEL_NAME_RE.match(model_name):
        raise ValueError(f"MODEL_NAME must match {_MODEL_NAME_RE.pattern}; got {model_name!r}")

    bucket = _env("ARTIFACT_STORE_BUCKET") or f"{category}-artifacts"
    prefix = _env("ARTIFACT_STORE_PREFIX", "")

    cfg = AppConfig(
        category=category,
        project=project,
        model_name=model_name,
        bucket=bucket,
        prefix=prefix,
        channel=channel,
        aws_region=_env("AWS_DEFAULT_REGION", "us-east-1"),
        admin_token=_env("APP_ADMIN_TOKEN"),
        reload_interval_s=_parse_int("APP_RELOAD_INTERVAL_S", _env("APP_RELOAD_INTERVAL_S", "30")),
        host=_env("APP_HOST", "0.0.0.0"),
        port=_parse_int("APP_PORT", _env("APP_PORT", "8000")),
        max_batch_size=_parse_int("APP_MAX_BATCH_SIZE", _env("APP_MAX_BATCH_SIZE", "1000")),
        max_request_bytes=_parse_int(
            "APP_MAX_REQUEST_BYTES", _env("APP_MAX_REQUEST_BYTES", "1048576")
        ),
        cors_allowed_origins=_env("APP_CORS_ALLOWED_ORIGINS", ""),
        env=_env("ENV", "dev").lower(),
        log_format=_env("LOG_FORMAT", "").lower(),
        strict_schema=_parse_bool(_env("APP_STRICT_SCHEMA"), default=True),
        startup_grace_seconds=_parse_int(
            "APP_STARTUP_GRACE_SECONDS", _env("APP_STARTUP_GRACE_SECONDS", "120")
        ),
        training_repo=_env("TRAINING_REPO"),
        training_repo_token=_env("TRAINING_REPO_TOKEN"),
        training_auto_promote=_parse_bool(_env("TRAINING_AUTO_PROMOTE"), default=False),
    )

    if cfg.env == "prod":
        if not cfg.admin_token:
            raise OSError("APP_ADMIN_TOKEN is required in prod.")
        if "*" in cfg.cors_allowed_origins:
            raise OSError("APP_CORS_ALLOWED_ORIGINS must not contain '*' in prod.")
        if cfg.channel != "stable":
            raise OSError("APP_CHANNEL must be 'stable' in prod.")
        if not cfg.training_repo or not cfg.training_repo_token:
            raise OSError("TRAINING_REPO and TRAINING_REPO_TOKEN are both required in prod.")
        if not os.environ.get("TRIGGER_DATA_ROOT", "").strip():
            raise OSError("TRIGGER_DATA_ROOT is required in prod (guards path traversal).")
    if cfg.max_batch_size <= 0:
        raise OSError(f"APP_MAX_BATCH_SIZE must be > 0; got {cfg.max_batch_size}.")
    if cfg.max_request_bytes <= 0:
        raise OSError(f"APP_MAX_REQUEST_BYTES must be > 0; got {cfg.max_request_bytes}.")
    return cfg
