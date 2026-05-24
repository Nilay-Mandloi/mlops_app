"""S3 implementation of ReadOnlyArtifactStore. The only file in the serving
package that imports boto3."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger

from price_forecast.ports.storage import ArtifactStore

T = TypeVar("T")

_RETRYABLE_S3_CODES = frozenset(
    {
        "InternalError",
        "ServiceUnavailable",
        "SlowDown",
        "RequestTimeout",
        "RequestTimeoutException",
        "ProvisionedThroughputExceededException",
        "ThrottlingException",
        "Throttling",
        "500",
        "502",
        "503",
        "504",
    }
)


def _retry_s3(
    label: str, func: Callable[[], T], *, attempts: int = 4, base_delay: float = 0.5
) -> T:
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return func()
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in _RETRYABLE_S3_CODES:
                raise
            last_exc = exc
        except BotoCoreError as exc:
            last_exc = exc
        if attempt == attempts - 1:
            break
        delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
        logger.warning(
            "{} failed (attempt {}/{}): {} — retrying in {:.1f}s",
            label,
            attempt + 1,
            attempts,
            last_exc,
            delay,
        )
        time.sleep(delay)
    if last_exc is None:
        raise RuntimeError(f"{label}: retry loop exhausted with no recorded exception")
    raise last_exc


class S3ReadStore(ArtifactStore):
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        client=None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = client or boto3.client(
            "s3",
            region_name=region,
            config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}),
        )

    def _full_key(self, logical_key: str) -> str:
        logical_key = logical_key.lstrip("/")
        return f"{self._prefix}/{logical_key}" if self._prefix else logical_key

    def get_json(self, logical_key: str) -> dict | None:
        def _call() -> dict | None:
            try:
                resp = self._client.get_object(Bucket=self._bucket, Key=self._full_key(logical_key))
                return json.loads(resp["Body"].read().decode("utf-8"))
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                    return None
                raise

        return _retry_s3(f"get_json({logical_key})", _call)

    def download_file(self, logical_key: str, local_path: Path | str) -> None:
        _retry_s3(
            f"download({logical_key})",
            lambda: self._client.download_file(
                Bucket=self._bucket,
                Key=self._full_key(logical_key),
                Filename=str(local_path),
            ),
        )

    def exists(self, logical_key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=self._full_key(logical_key))
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
                return False
            raise

    def list_subkeys(self, prefix: str) -> Iterator[str]:
        full_prefix = self._full_key(prefix)
        if not full_prefix.endswith("/"):
            full_prefix += "/"
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []) or []:
                key_dir = cp.get("Prefix", "").rstrip("/")
                if not key_dir:
                    continue
                yield key_dir.rsplit("/", 1)[-1]

    def upload_file(
        self,
        local_path: Path | str,
        logical_key: str,
        *,
        content_type: str | None = None,
    ) -> None:
        extra: dict = {"ContentType": content_type} if content_type else {}
        _retry_s3(
            f"upload_file({logical_key})",
            lambda: self._client.upload_file(
                Filename=str(local_path),
                Bucket=self._bucket,
                Key=self._full_key(logical_key),
                ExtraArgs=extra,
            ),
        )

    def put_bytes(
        self,
        data: bytes,
        logical_key: str,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        _retry_s3(
            f"put_bytes({logical_key})",
            lambda: self._client.put_object(
                Bucket=self._bucket,
                Key=self._full_key(logical_key),
                Body=data,
                ContentType=content_type,
            ),
        )

    def delete(self, logical_key: str) -> None:
        try:
            self._client.delete_object(Bucket=self._bucket, Key=self._full_key(logical_key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return
            raise
