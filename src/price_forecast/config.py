"""Runtime config for the serving app, loaded from environment.

The training repo's settings.py is intentionally NOT imported here. This
keeps the two repos decoupled — each owns its own config surface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _require(name: str) -> str:
    value = _env(name)
    if not value:
        raise OSError(f"{name} environment variable is required.")
    return value


@dataclass(frozen=True)
class AppConfig:
    app_id: str
    bucket: str
    stack_id: str
    model_name: str
    channel: str  # which pointer to follow: stable | canary | latest
    aws_region: str
    admin_token: str
    reload_interval_s: int
    host: str
    port: int
    # Production knobs
    max_batch_size: int            # cap on /predict/batch rows
    max_request_bytes: int         # cap on request body size
    cors_allowed_origins: str      # comma-separated list; "*" only allowed when ENV!=prod
    env: str                       # "dev" | "prod" | "test"
    log_format: str                # "" (default loguru) | "json"
    strict_schema: bool            # if True, reject requests whose features don't match manifest schema
    startup_grace_seconds: int     # serve 503 for this long if pointer absent, instead of crashing

    @property
    def prefix(self) -> str:
        """Full S3 prefix this app reads from: e.g. 'MLOPS' or 'azure'."""
        return self.stack_id


def _parse_bool(value: str, *, default: bool) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    cfg = AppConfig(
        app_id=_require("APP_ID"),
        bucket=_require("APP_S3_BUCKET"),
        stack_id=_env("STACK_ID", "MLOPS"),
        model_name=_require("APP_MODEL_NAME"),
        channel=_env("APP_CHANNEL", "stable"),
        aws_region=_env("AWS_DEFAULT_REGION", "us-east-1"),
        admin_token=_env("APP_ADMIN_TOKEN"),
        reload_interval_s=int(_env("APP_RELOAD_INTERVAL_S", "30")),
        host=_env("APP_HOST", "0.0.0.0"),
        port=int(_env("APP_PORT", "8000")),
        max_batch_size=int(_env("APP_MAX_BATCH_SIZE", "1000")),
        max_request_bytes=int(_env("APP_MAX_REQUEST_BYTES", "1048576")),
        cors_allowed_origins=_env("APP_CORS_ALLOWED_ORIGINS", ""),
        env=_env("ENV", "dev").lower(),
        log_format=_env("LOG_FORMAT", "").lower(),
        strict_schema=_parse_bool(_env("APP_STRICT_SCHEMA"), default=True),
        startup_grace_seconds=int(_env("APP_STARTUP_GRACE_SECONDS", "120")),
    )

    if cfg.env == "prod":
        if not cfg.admin_token:
            raise OSError("APP_ADMIN_TOKEN is required in prod (guards /reload and /trigger-train).")
        if "*" in cfg.cors_allowed_origins:
            raise OSError("APP_CORS_ALLOWED_ORIGINS must not contain '*' in prod.")
    if cfg.max_batch_size <= 0:
        raise OSError(f"APP_MAX_BATCH_SIZE must be > 0; got {cfg.max_batch_size}.")
    if cfg.max_request_bytes <= 0:
        raise OSError(f"APP_MAX_REQUEST_BYTES must be > 0; got {cfg.max_request_bytes}.")
    return cfg
