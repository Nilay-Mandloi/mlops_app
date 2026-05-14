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

    @property
    def prefix(self) -> str:
        """Full S3 prefix this app reads from: e.g. 'MLOPS' or 'azure'."""
        return self.stack_id


def load_config() -> AppConfig:
    return AppConfig(
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
    )
