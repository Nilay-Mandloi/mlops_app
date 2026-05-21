"""Publisher — pushes a training trigger (dataset + params + metadata) to S3.

This is the producer side of the training contract. The Flask app calls
``publish_trigger`` when it wants to kick off a new training run; the
training repo's ``pull_trigger`` (in quantity_forecast.trigger) reads the
same folder shape.
"""

from __future__ import annotations

import json as _json
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import boto3
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


def _dispatch_training(trigger_id: str, cfg: AppConfig) -> None:
    """Fire a GitHub repository_dispatch to the training repo.

    Short-circuits silently when TRAINING_REPO or TRAINING_REPO_TOKEN are not
    configured — backward compatible with deployments that don't use GitHub Actions.

    Raises RuntimeError on a non-2xx response so the /trigger-train caller gets
    a clear signal that the training job could not be queued.
    """
    if not cfg.training_repo or not cfg.training_repo_token:
        logger.warning(
            "TRAINING_REPO or TRAINING_REPO_TOKEN not set — skipping GitHub dispatch "
            "for trigger_id={}. Set both env vars to enable automatic training dispatch.",
            trigger_id,
        )
        return

    url = f"https://api.github.com/repos/{cfg.training_repo}/dispatches"
    payload = _json.dumps(
        {
            "event_type": "train-model",
            "client_payload": {
                "trigger_id": trigger_id,
                "auto_promote": cfg.training_auto_promote,
                # Multi-tenant: training repo reads these from the payload so
                # each app's run is scoped to its own bucket and namespace.
                "app_id": cfg.app_id,
                "artifact_store_bucket": cfg.bucket,
                "artifact_store_prefix": cfg.stack_id,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {cfg.training_repo_token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    _RETRYABLE_HTTP = {429, 500, 502, 503, 504}
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                status = resp.status
            break
        except urllib.error.HTTPError as exc:
            if exc.code in _RETRYABLE_HTTP and attempt == 0:
                last_exc = exc
                logger.warning("GitHub dispatch HTTP {} (attempt 1/2), retrying in 2s", exc.code)
                time.sleep(2)
            else:
                raise RuntimeError(
                    f"GitHub dispatch failed: HTTP {exc.code} for repo={cfg.training_repo} "
                    f"trigger_id={trigger_id}. Check TRAINING_REPO_TOKEN permissions "
                    "(needs Contents:write on the training repo)."
                ) from exc
        except OSError as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning(
                    "GitHub dispatch network error (attempt 1/2), retrying in 2s: {}", exc
                )
                time.sleep(2)
    else:
        raise RuntimeError(
            f"GitHub dispatch failed after retry for trigger_id={trigger_id}: {last_exc}"
        )

    # GitHub returns 204 No Content on success.
    if status not in (200, 201, 204):
        raise RuntimeError(
            f"GitHub dispatch returned unexpected status {status} "
            f"for repo={cfg.training_repo} trigger_id={trigger_id}."
        )
    logger.info(
        "Dispatched train-model event to {} (trigger_id={} auto_promote={})",
        cfg.training_repo,
        trigger_id,
        cfg.training_auto_promote,
    )


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
    s3_client: Any | None = None,
) -> tuple[str, str]:
    """Push a trigger folder to S3 and return (trigger_id, trigger_uri).

    Layout written (the extension matches the actual file format):
        s3://{bucket}/{stack_id}/triggers/{app_id}/{trigger_id}/dataset.{csv|parquet}
        s3://{bucket}/{stack_id}/triggers/{app_id}/{trigger_id}/params.yaml
        s3://{bucket}/{stack_id}/triggers/{app_id}/{trigger_id}/trigger.json

    Pass ``cfg`` and ``s3_client`` to inject dependencies (avoids load_config() side-effects
    in tests and in the Flask request handler which already holds a config object).
    """
    if cfg is None:
        cfg = load_config()
    client = s3_client or boto3.client("s3", region_name=cfg.aws_region)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trigger_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

    dataset_path = Path(dataset_path)
    params_path = Path(params_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"params_path does not exist: {params_path}")

    fmt = _infer_dataset_format(dataset_path, dataset_format)

    dataset_key = trigger_dataset_key(cfg.app_id, trigger_id, fmt)
    params_key_logical = trigger_params_key(cfg.app_id, trigger_id)
    metadata_key = trigger_metadata_key(cfg.app_id, trigger_id)

    metadata = TriggerFile(
        trigger_id=trigger_id,
        app_id=cfg.app_id,
        model_family=model_family,
        dataset_uri=f"s3://{cfg.bucket}/{_full_key(cfg.prefix, dataset_key)}",
        params_uri=f"s3://{cfg.bucket}/{_full_key(cfg.prefix, params_key_logical)}",
        dataset_format=fmt,
        requested_by=requested_by,
        description=description,
    )

    # Order is load-bearing: dataset + params first, trigger.json LAST.
    # The puller treats trigger.json as the completion marker — its presence
    # guarantees the other two keys are already in place.
    # On any upload failure we delete already-uploaded keys so the trigger
    # folder does not linger as orphaned partial data in S3.
    uploaded_keys: list[str] = []

    def _rollback_uploads() -> None:
        for key in reversed(uploaded_keys):
            try:
                client.delete_object(Bucket=cfg.bucket, Key=key)
            except Exception as cleanup_exc:
                logger.warning("Failed to clean up orphaned S3 key {}: {}", key, cleanup_exc)

    content_type = "text/csv" if fmt == "csv" else "application/octet-stream"
    try:
        dataset_full_key = _full_key(cfg.prefix, dataset_key)
        client.upload_file(
            Filename=str(dataset_path),
            Bucket=cfg.bucket,
            Key=dataset_full_key,
            ExtraArgs={"ContentType": content_type},
        )
        uploaded_keys.append(dataset_full_key)

        params_full_key = _full_key(cfg.prefix, params_key_logical)
        client.upload_file(
            Filename=str(params_path),
            Bucket=cfg.bucket,
            Key=params_full_key,
        )
        uploaded_keys.append(params_full_key)

        metadata_full_key = _full_key(cfg.prefix, metadata_key)
        client.put_object(
            Bucket=cfg.bucket,
            Key=metadata_full_key,
            Body=_json.dumps(metadata.to_dict(), indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        uploaded_keys.append(metadata_full_key)
    except Exception:
        _rollback_uploads()
        raise

    trigger_uri = f"s3://{cfg.bucket}/{cfg.prefix.strip('/')}/triggers/{cfg.app_id}/{trigger_id}/"
    logger.info(
        "Published trigger {} (app_id={}, format={}) -> {}",
        trigger_id,
        cfg.app_id,
        fmt,
        trigger_uri,
    )

    # Notify the training repo so it can start a training run immediately.
    # Skipped silently when TRAINING_REPO / TRAINING_REPO_TOKEN are unset.
    # If dispatch fails after the S3 uploads committed, write failed.json so
    # /trigger-status/<id> reports "failed" instead of hanging in "pending".
    try:
        _dispatch_training(trigger_id, cfg)
    except RuntimeError as dispatch_exc:
        failure_body = _json.dumps(
            {
                "status": "failed",
                "reason": f"GitHub dispatch failed: {dispatch_exc}",
                "trigger_id": trigger_id,
            }
        ).encode("utf-8")
        failure_key = _full_key(cfg.prefix, trigger_failure_key(cfg.app_id, trigger_id))
        try:
            client.put_object(
                Bucket=cfg.bucket,
                Key=failure_key,
                Body=failure_body,
                ContentType="application/json",
            )
        except Exception as marker_exc:
            logger.warning(
                "Could not write dispatch-failure marker for trigger {}: {}", trigger_id, marker_exc
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
    dataset_format: Optional[str] = typer.Option(  # noqa: B008, UP045
        None,
        "--dataset-format",
        help="Override auto-detection: 'csv' or 'parquet'. Default: inferred from file extension.",
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
