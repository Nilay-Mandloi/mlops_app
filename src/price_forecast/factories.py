"""Adapter factories — the only file in the serving package that names
concrete adapter classes. Business logic depends on the ports.

Selection:
    STORAGE_BACKEND        (default: "s3")
    ORCHESTRATION_BACKEND  (default: "github" if TRAINING_REPO configured, else "noop")
"""

from __future__ import annotations

import os

from price_forecast.config import AppConfig
from price_forecast.ports.orchestration import OrchestrationAdapter
from price_forecast.ports.storage import ArtifactStore


def _storage_backend() -> str:
    return os.environ.get("STORAGE_BACKEND", "s3").strip().lower() or "s3"


def _orchestration_backend(cfg: AppConfig) -> str:
    raw = os.environ.get("ORCHESTRATION_BACKEND", "").strip().lower()
    if raw:
        return raw
    return "github" if (cfg.training_repo and cfg.training_repo_token) else "noop"


def get_artifact_store(cfg: AppConfig) -> ArtifactStore:
    backend = _storage_backend()
    if backend == "s3":
        from price_forecast.adapters.s3_store import S3ReadStore

        return S3ReadStore(bucket=cfg.bucket, prefix=cfg.prefix, region=cfg.aws_region)
    raise ValueError(
        f"Unknown STORAGE_BACKEND='{backend}'. Supported: s3. "
        "Add an adapter under adapters/<backend>_store.py and a branch here."
    )


def get_orchestrator(cfg: AppConfig) -> OrchestrationAdapter:
    backend = _orchestration_backend(cfg)
    if backend == "github":
        from price_forecast.adapters.github_dispatch import GitHubDispatchAdapter

        return GitHubDispatchAdapter(
            training_repo=cfg.training_repo,
            training_repo_token=cfg.training_repo_token,
        )
    if backend == "noop":
        from price_forecast.adapters.github_dispatch import NoopDispatchAdapter

        return NoopDispatchAdapter()
    raise ValueError(
        f"Unknown ORCHESTRATION_BACKEND='{backend}'. Supported: github, noop. "
        "Add an adapter under adapters/<backend>_dispatch.py and a branch here."
    )
