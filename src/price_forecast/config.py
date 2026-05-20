"""Runtime config for the serving app, loaded from environment.

The training repo's settings.py is intentionally NOT imported here. This
keeps the two repos decoupled — each owns its own config surface.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_APP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")


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
        raise OSError(
            f"{name} must be an integer; got {value!r}. "
            f"Check the {name} environment variable."
        ) from None


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
    # Training dispatch — fire a repository_dispatch to the training repo after /trigger-train.
    # Both must be set; if either is empty the dispatch is silently skipped (backward compatible).
    training_repo: str             # env: TRAINING_REPO  (e.g. "my-org/mlops")
    training_repo_token: str       # env: TRAINING_REPO_TOKEN  (PAT with Contents:write on training repo)
    training_auto_promote: bool    # env: TRAINING_AUTO_PROMOTE  (default false)

    @property
    def prefix(self) -> str:
        """Full S3 prefix this app reads from: e.g. 'MLOPS' or 'azure'."""
        return self.stack_id


def _parse_bool(value: str, *, default: bool) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> AppConfig:
    channel = _env("APP_CHANNEL", "stable")
    if channel not in {"stable", "canary", "latest"}:
        raise OSError("APP_CHANNEL must be one of stable, canary, latest.")
    raw_app_id = _require("APP_ID")
    if not _APP_ID_RE.match(raw_app_id):
        raise ValueError(
            f"APP_ID must match ^[a-z0-9][a-z0-9_-]{{0,62}}$; got {raw_app_id!r}"
        )
    cfg = AppConfig(
        app_id=raw_app_id,
        bucket=_require("APP_S3_BUCKET"),
        stack_id=_env("STACK_ID", "MLOPS"),
        model_name=_require("APP_MODEL_NAME"),
        channel=channel,
        aws_region=_env("AWS_DEFAULT_REGION", "us-east-1"),
        admin_token=_env("APP_ADMIN_TOKEN"),
        reload_interval_s=_parse_int("APP_RELOAD_INTERVAL_S", _env("APP_RELOAD_INTERVAL_S", "30")),
        host=_env("APP_HOST", "0.0.0.0"),
        port=_parse_int("APP_PORT", _env("APP_PORT", "8000")),
        max_batch_size=_parse_int("APP_MAX_BATCH_SIZE", _env("APP_MAX_BATCH_SIZE", "1000")),
        max_request_bytes=_parse_int("APP_MAX_REQUEST_BYTES", _env("APP_MAX_REQUEST_BYTES", "1048576")),
        cors_allowed_origins=_env("APP_CORS_ALLOWED_ORIGINS", ""),
        env=_env("ENV", "dev").lower(),
        log_format=_env("LOG_FORMAT", "").lower(),
        strict_schema=_parse_bool(_env("APP_STRICT_SCHEMA"), default=True),
        startup_grace_seconds=_parse_int("APP_STARTUP_GRACE_SECONDS", _env("APP_STARTUP_GRACE_SECONDS", "120")),
        training_repo=_env("TRAINING_REPO"),
        training_repo_token=_env("TRAINING_REPO_TOKEN"),
        training_auto_promote=_parse_bool(_env("TRAINING_AUTO_PROMOTE"), default=False),
    )

    if cfg.env == "prod":
        if not cfg.admin_token:
            raise OSError("APP_ADMIN_TOKEN is required in prod (guards /reload and /trigger-train).")
        if "*" in cfg.cors_allowed_origins:
            raise OSError("APP_CORS_ALLOWED_ORIGINS must not contain '*' in prod.")
        if cfg.channel != "stable":
            raise OSError("APP_CHANNEL must be 'stable' in prod.")
        if not cfg.training_repo or not cfg.training_repo_token:
            raise OSError(
                "TRAINING_REPO and TRAINING_REPO_TOKEN are both required in prod. "
                "Set both env vars so /trigger-train can dispatch to the training repo."
            )
        if not os.environ.get("TRIGGER_DATA_ROOT", "").strip():
            raise OSError(
                "TRIGGER_DATA_ROOT is required in prod (guards path traversal in /trigger-train)."
            )
    if cfg.max_batch_size <= 0:
        raise OSError(f"APP_MAX_BATCH_SIZE must be > 0; got {cfg.max_batch_size}.")
    if cfg.max_request_bytes <= 0:
        raise OSError(f"APP_MAX_REQUEST_BYTES must be > 0; got {cfg.max_request_bytes}.")
    return cfg
