"""Publisher — pushes a training trigger (dataset + params + metadata) and
asks the orchestrator to start a run. Backend-neutral.

Producer side of the training contract. The training repo's `pull_trigger`
in quantity_forecast.trigger reads the same folder shape.
"""

from __future__ import annotations

import json as _json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from loguru import logger

from price_forecast.config import AppConfig, load_config
from price_forecast.contracts import TriggerFile
from price_forecast.layout import (
    trigger_dataset_key,
    trigger_failure_key,
    trigger_metadata_key,
    trigger_params_key,
)
from price_forecast.ports.orchestration import OrchestrationAdapter
from price_forecast.ports.storage import ArtifactStore


def _full_key(prefix: str, logical_key: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/{logical_key}" if prefix else logical_key


_EXTENSION_TO_FORMAT = {
    ".csv": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
}


def _infer_dataset_format(dataset_path: Path, override: str | None) -> str:
    if override:
        if override not in {"csv", "parquet"}:
            raise ValueError(
                f"dataset_format override must be 'csv' or 'parquet'; got {override!r}"
            )
        return override
    suffix = dataset_path.suffix.lower()
    fmt = _EXTENSION_TO_FORMAT.get(suffix)
    if fmt is None:
        raise ValueError(
            f"Cannot infer dataset_format from extension {suffix!r} for path "
            f"{dataset_path}. Supported: .csv, .parquet (.pq). Pass dataset_format= "
            "explicitly to override."
        )
    return fmt


def publish_trigger(
    dataset_path: str | Path,
    params_path: str | Path,
    *,
    model_family: str,
    description: str = "",
    requested_by: str = "",
    dataset_format: str | None = None,
    cfg: AppConfig | None = None,
    store: ArtifactStore | None = None,
    orchestrator: OrchestrationAdapter | None = None,
) -> tuple[str, str]:
    """Push a trigger folder via the storage port, then dispatch via the
    orchestration port. Returns (trigger_id, trigger_uri).

    Layout: ``s3://{bucket}/[<prefix>/]_triggers/{project}/{trigger_id}/...``
    """
    if cfg is None:
        cfg = load_config()
    if store is None or orchestrator is None:
        from price_forecast.factories import get_artifact_store, get_orchestrator

        store = store or get_artifact_store(cfg)
        orchestrator = orchestrator or get_orchestrator(cfg)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trigger_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

    dataset_path = Path(dataset_path)
    params_path = Path(params_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"params_path does not exist: {params_path}")

    fmt = _infer_dataset_format(dataset_path, dataset_format)

    dataset_key = trigger_dataset_key(cfg.project, trigger_id, fmt)
    params_key = trigger_params_key(cfg.project, trigger_id)
    metadata_key = trigger_metadata_key(cfg.project, trigger_id)

    metadata = TriggerFile(
        trigger_id=trigger_id,
        category=cfg.category,
        project=cfg.project,
        model_name=cfg.model_name,
        model_family=model_family,
        dataset_uri=f"s3://{cfg.bucket}/{_full_key(cfg.prefix, dataset_key)}",
        params_uri=f"s3://{cfg.bucket}/{_full_key(cfg.prefix, params_key)}",
        dataset_format=fmt,
        requested_by=requested_by,
        description=description,
    )

    # Order is load-bearing: dataset + params first, trigger.json LAST. The
    # puller treats trigger.json as the completion marker — its presence
    # guarantees the other two keys are already in place. On any upload
    # failure delete already-uploaded keys so the trigger folder does not
    # linger as orphaned partial data.
    uploaded: list[str] = []
    content_type = "text/csv" if fmt == "csv" else "application/octet-stream"
    try:
        store.upload_file(dataset_path, dataset_key, content_type=content_type)
        uploaded.append(dataset_key)
        store.upload_file(params_path, params_key)
        uploaded.append(params_key)
        store.put_bytes(
            _json.dumps(metadata.to_dict(), indent=2).encode("utf-8"),
            metadata_key,
            content_type="application/json",
        )
        uploaded.append(metadata_key)
    except Exception:
        for key in reversed(uploaded):
            try:
                store.delete(key)
            except Exception as cleanup_exc:
                logger.warning("Failed to clean up orphaned key {}: {}", key, cleanup_exc)
        raise

    prefix_part = f"{cfg.prefix.strip('/')}/" if cfg.prefix.strip("/") else ""
    trigger_uri = f"s3://{cfg.bucket}/{prefix_part}_triggers/{cfg.project}/{trigger_id}/"
    logger.info(
        "Published trigger {} ({}/{}, format={}) -> {}",
        trigger_id,
        cfg.project,
        cfg.model_name,
        fmt,
        trigger_uri,
    )

    # Hand off to the orchestrator. If it refuses, write a failed.json marker
    # so /trigger-status/<id> reports "failed" instead of hanging in "pending".
    try:
        orchestrator.dispatch_training(
            trigger_id=trigger_id,
            category=cfg.category,
            project=cfg.project,
            model_name=cfg.model_name,
            bucket=cfg.bucket,
            prefix=cfg.prefix,
            auto_promote=cfg.training_auto_promote,
        )
    except RuntimeError as dispatch_exc:
        failure_body = _json.dumps(
            {
                "status": "failed",
                "reason": f"orchestrator dispatch failed: {dispatch_exc}",
                "trigger_id": trigger_id,
            }
        ).encode("utf-8")
        try:
            store.put_bytes(
                failure_body,
                trigger_failure_key(cfg.project, trigger_id),
                content_type="application/json",
            )
        except Exception as marker_exc:
            logger.warning(
                "Could not write dispatch-failure marker for trigger {}: {}",
                trigger_id,
                marker_exc,
            )
        raise

    return trigger_id, trigger_uri


cli_app = typer.Typer()


@cli_app.command("push")
def push_cmd(
    dataset: Path = typer.Option(..., "--dataset"),  # noqa: B008
    params: Path = typer.Option(..., "--params"),  # noqa: B008
    model_family: str = typer.Option(..., "--model-family"),
    description: str = typer.Option("", "--description"),
    requested_by: str = typer.Option("", "--requested-by"),
    dataset_format: str | None = typer.Option(  # noqa: B008, UP045
        None,
        "--dataset-format",
        help="Override auto-detection: 'csv' or 'parquet'.",
    ),
) -> None:
    """Push a trigger folder from local files."""
    trigger_id, uri = publish_trigger(
        dataset,
        params,
        model_family=model_family,
        description=description,
        requested_by=requested_by,
        dataset_format=dataset_format,
    )
    typer.echo(f"trigger_id: {trigger_id}")
    typer.echo(f"trigger_uri: {uri}")


def cli() -> None:
    cli_app()


if __name__ == "__main__":
    cli()
