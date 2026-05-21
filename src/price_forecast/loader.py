"""Model loader — reads pointer.json from S3, downloads model.pkl, verifies checksum.

This is the consumer side of the training/serving contract. It has zero
MLflow dependency — only boto3 and the shared `contracts` + `layout` modules.

Thread-safe: a single ``ModelStore`` instance is shared across Flask request
threads. State swaps are atomic via a single replace of the internal
``_current`` reference (Python's GIL guarantees atomicity of attr assignment).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TypeVar

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger

from price_forecast.config import AppConfig
from price_forecast.contracts import ArtifactManifest, PointerFile
from price_forecast.layout import artifact_model_pkl_key, pointer_key

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
    """Exponential backoff with jitter for transient S3 errors.

    Distinct from boto3's built-in retries (which cover the low-level call)
    in that this wraps a higher-level operation — e.g. "read pointer, then
    read manifest, then download pkl" — and retries the whole step if any
    inner call hit a known-transient failure.
    """
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
        raise RuntimeError(
            f"{label}: retry loop exhausted with no recorded exception (bug in _retry_s3)"
        )
    raise last_exc


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class LoadedModel:
    """A loaded model plus the metadata that identifies it."""

    version_id: str
    pointer: PointerFile
    manifest: ArtifactManifest
    model: Any  # sklearn pipeline, etc.
    loaded_at: float


class ModelStore:
    """Single-flight model loader.

    Usage:
        store = ModelStore(config)
        store.reload()              # eager initial load
        loaded = store.current()    # never None after first successful reload
    """

    def __init__(self, config: AppConfig) -> None:
        self._cfg = config
        # boto3's built-in standard retry mode + 5 attempts handles the
        # request-level transient errors; _retry_s3 around it wraps the
        # whole "read pointer + manifest + pkl" sequence.
        self._client = boto3.client(
            "s3",
            region_name=config.aws_region,
            config=BotoConfig(retries={"max_attempts": 5, "mode": "standard"}),
        )
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # S3 helpers
    # ------------------------------------------------------------------

    def _full_key(self, logical_key: str) -> str:
        prefix = self._cfg.prefix.strip("/")
        return f"{prefix}/{logical_key}" if prefix else logical_key

    def _get_json(self, logical_key: str) -> dict | None:
        def _call() -> dict | None:
            try:
                resp = self._client.get_object(
                    Bucket=self._cfg.bucket, Key=self._full_key(logical_key)
                )
                return json.loads(resp["Body"].read().decode("utf-8"))
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                    return None
                raise

        return _retry_s3(f"get_json({logical_key})", _call)

    def _download_to(self, logical_key: str, dest: Path) -> None:
        _retry_s3(
            f"download({logical_key})",
            lambda: self._client.download_file(
                Bucket=self._cfg.bucket, Key=self._full_key(logical_key), Filename=str(dest)
            ),
        )

    def _logical_key_from_uri(self, uri: str) -> str:
        prefix = f"s3://{self._cfg.bucket}/"
        if not uri.startswith(prefix):
            raise RuntimeError(
                f"Pointer manifest_uri '{uri}' does not belong to bucket '{self._cfg.bucket}'."
            )
        key = uri[len(prefix) :]
        stack_prefix = self._cfg.prefix.strip("/")
        if stack_prefix:
            expected = f"{stack_prefix}/"
            if not key.startswith(expected):
                raise RuntimeError(
                    f"Pointer manifest_uri '{uri}' does not belong to stack prefix '{stack_prefix}'."
                )
            key = key[len(expected) :]
        return key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def current(self) -> LoadedModel:
        with self._lock:
            if self._current is None:
                raise RuntimeError(
                    "No model loaded. Call reload() at startup before serving requests."
                )
            return self._current

    def current_or_none(self) -> LoadedModel | None:
        """Non-raising getter for status / probe endpoints."""
        with self._lock:
            return self._current

    def try_reload(self) -> LoadedModel | None:
        """Reload without raising. Returns None if pointer is absent (first deploy).

        Re-raises on any non-LookupError failure so transient infra issues
        are still visible via the caller's exception handling.
        """
        try:
            return self.reload()
        except LookupError as exc:
            logger.info("No pointer yet — serving will start in standby mode: {}", exc)
            return None

    def reload(self) -> LoadedModel:
        """Re-read the pointer, download the artifact if version changed, swap atomically.

        Every key is app-scoped: pointer, manifest, and pkl are all read
        from ``output/.../{app_id}/...``. Both pointer.app_id and
        manifest.app_id are cross-checked against ``self._cfg.app_id``;
        a payload claiming a different scope is refused — that's either
        a misroute or a config error and serving the wrong model is worse
        than serving nothing.
        """
        app_id = self._cfg.app_id
        pointer_data = self._get_json(pointer_key(app_id, self._cfg.model_name, self._cfg.channel))
        if pointer_data is None:
            raise LookupError(
                f"Pointer '{self._cfg.channel}' for app_id='{app_id}' "
                f"model='{self._cfg.model_name}' not found. "
                "Has the training pipeline run and promoted a version yet?"
            )
        pointer = PointerFile.from_dict(pointer_data)
        if pointer.app_id != app_id:
            raise RuntimeError(
                f"Pointer scope mismatch: file claims app_id='{pointer.app_id}' but "
                f"this app is configured for app_id='{app_id}'. Refusing to load — "
                "this would serve another app's model."
            )
        with self._lock:
            if self._current is not None and self._current.version_id == pointer.version_id:
                logger.debug("Pointer unchanged at {} — skipping reload.", pointer.version_id)
                return self._current

        manifest_key = self._logical_key_from_uri(pointer.manifest_uri)
        manifest_data = self._get_json(manifest_key)
        if not manifest_data:
            raise RuntimeError(
                f"Manifest for app_id='{app_id}' version {pointer.version_id} missing. "
                "Refusing to load model without a manifest to verify checksums against."
            )
        manifest = ArtifactManifest.from_dict(manifest_data)
        if manifest.app_id != app_id:
            raise RuntimeError(
                f"Manifest scope mismatch: file claims app_id='{manifest.app_id}' but "
                f"this app is configured for app_id='{app_id}'. Refusing to load."
            )
        if manifest.model_name != self._cfg.model_name:
            raise RuntimeError(
                f"Manifest model mismatch: file claims model_name='{manifest.model_name}' but "
                f"this app is configured for model_name='{self._cfg.model_name}'. Refusing to load."
            )

        expected_checksum = manifest.artifact_checksums.get("model.pkl")
        if not expected_checksum:
            raise RuntimeError(
                f"Manifest for {pointer.version_id} has no checksum for model.pkl. "
                "Refusing to load — re-promote the model to regenerate the manifest."
            )

        with TemporaryDirectory() as tmp:
            local_pkl = Path(tmp) / "model.pkl"
            self._download_to(artifact_model_pkl_key(app_id, pointer.version_id), local_pkl)

            actual = _sha256_file(local_pkl)
            if actual != expected_checksum:
                raise RuntimeError(
                    f"Checksum mismatch for {pointer.version_id} model.pkl: "
                    f"expected {expected_checksum}, got {actual}. Refusing to serve."
                )

            with local_pkl.open("rb") as fh:
                model = pickle.load(fh)  # noqa: S301 — trusted artifact w/ verified checksum

        loaded = LoadedModel(
            version_id=pointer.version_id,
            pointer=pointer,
            manifest=manifest,
            model=model,
            loaded_at=time.time(),
        )
        with self._lock:
            previous = self._current
            self._current = loaded

        logger.info(
            "Loaded model {} (was {}); checksum verified.",
            loaded.version_id,
            previous.version_id if previous else "<none>",
        )
        return loaded
