"""Publisher — pushes a training trigger (dataset + params + metadata) to S3.

This is the producer side of the training contract. The Flask app calls
``publish_trigger`` when it wants to kick off a new training run; the
training repo's ``pull_trigger`` (in quantity_forecast.trigger) reads the
same folder shape.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import typer
from loguru import logger

from price_forecast.config import load_config
from price_forecast.contracts import TriggerFile
from price_forecast.layout import (
    trigger_dataset_key,
    trigger_metadata_key,
    trigger_params_key,
)


def _full_key(prefix: str, logical_key: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/{logical_key}" if prefix else logical_key


def publish_trigger(
    dataset_path: str | Path,
    params_path: str | Path,
    *,
    model_family: str,
    description: str = "",
    requested_by: str = "",
) -> tuple[str, str]:
    """Push a trigger folder to S3 and return (trigger_id, trigger_uri).

    Layout written:
        s3://{bucket}/{stack_id}/triggers/{trigger_id}/dataset.parquet
        s3://{bucket}/{stack_id}/triggers/{trigger_id}/params.yaml
        s3://{bucket}/{stack_id}/triggers/{trigger_id}/trigger.json
    """
    cfg = load_config()
    client = boto3.client("s3", region_name=cfg.aws_region)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trigger_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"

    dataset_path = Path(dataset_path)
    params_path = Path(params_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset_path does not exist: {dataset_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"params_path does not exist: {params_path}")

    metadata = TriggerFile(
        trigger_id=trigger_id,
        app_id=cfg.app_id,
        model_family=model_family,
        dataset_uri=f"s3://{cfg.bucket}/{_full_key(cfg.prefix, trigger_dataset_key(trigger_id))}",
        params_uri=f"s3://{cfg.bucket}/{_full_key(cfg.prefix, trigger_params_key(trigger_id))}",
        requested_by=requested_by,
        description=description,
    )

    client.upload_file(
        Filename=str(dataset_path),
        Bucket=cfg.bucket,
        Key=_full_key(cfg.prefix, trigger_dataset_key(trigger_id)),
    )
    client.upload_file(
        Filename=str(params_path),
        Bucket=cfg.bucket,
        Key=_full_key(cfg.prefix, trigger_params_key(trigger_id)),
    )
    import json as _json

    client.put_object(
        Bucket=cfg.bucket,
        Key=_full_key(cfg.prefix, trigger_metadata_key(trigger_id)),
        Body=_json.dumps(metadata.to_dict(), indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    trigger_uri = f"s3://{cfg.bucket}/{cfg.prefix.strip('/')}/triggers/{trigger_id}/"
    logger.info("Published trigger {} -> {}", trigger_id, trigger_uri)
    return trigger_id, trigger_uri


cli_app = typer.Typer()


@cli_app.command("push")
def push_cmd(
    dataset: Path = typer.Option(..., "--dataset"),  # noqa: B008
    params: Path = typer.Option(..., "--params"),  # noqa: B008
    model_family: str = typer.Option(..., "--model-family"),
    description: str = typer.Option("", "--description"),
    requested_by: str = typer.Option("", "--requested-by"),
) -> None:
    """Push a trigger folder from local files."""
    trigger_id, uri = publish_trigger(
        dataset, params, model_family=model_family, description=description, requested_by=requested_by
    )
    typer.echo(f"trigger_id: {trigger_id}")
    typer.echo(f"trigger_uri: {uri}")


def cli() -> None:
    cli_app()


if __name__ == "__main__":
    cli()
