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
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import boto3
from botocore.exceptions import ClientError
from loguru import logger

from price_forecast.config import AppConfig
from price_forecast.contracts import ArtifactManifest, PointerFile
from price_forecast.layout import (
    artifact_manifest_key,
    artifact_model_pkl_key,
    pointer_key,
)


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
        self._client = boto3.client("s3", region_name=config.aws_region)
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # S3 helpers
    # ------------------------------------------------------------------

    def _full_key(self, logical_key: str) -> str:
        prefix = self._cfg.prefix.strip("/")
        return f"{prefix}/{logical_key}" if prefix else logical_key

    def _get_json(self, logical_key: str) -> dict | None:
        try:
            resp = self._client.get_object(
                Bucket=self._cfg.bucket, Key=self._full_key(logical_key)
            )
            return json.loads(resp["Body"].read().decode("utf-8"))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise

    def _download_to(self, logical_key: str, dest: Path) -> None:
        self._client.download_file(
            Bucket=self._cfg.bucket, Key=self._full_key(logical_key), Filename=str(dest)
        )

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

    def reload(self) -> LoadedModel:
        """Re-read the pointer, download the artifact if version changed, swap atomically."""
        pointer_data = self._get_json(pointer_key(self._cfg.model_name, self._cfg.channel))
        if pointer_data is None:
            raise RuntimeError(
                f"Pointer '{self._cfg.channel}' for model '{self._cfg.model_name}' not found. "
                "Has the training pipeline run and promoted a version yet?"
            )
        pointer = PointerFile.from_dict(pointer_data)

        with self._lock:
            if self._current is not None and self._current.version_id == pointer.version_id:
                logger.debug("Pointer unchanged at {} — skipping reload.", pointer.version_id)
                return self._current

        manifest_data = self._get_json(artifact_manifest_key(pointer.version_id))
        if not manifest_data:
            raise RuntimeError(
                f"Manifest for version {pointer.version_id} missing in S3. "
                "Refusing to load model without a manifest to verify checksums against."
            )
        manifest = ArtifactManifest.from_dict(manifest_data)

        expected_checksum = manifest.artifact_checksums.get("model.pkl")
        if not expected_checksum:
            raise RuntimeError(
                f"Manifest for {pointer.version_id} has no checksum for model.pkl. "
                "Refusing to load — re-promote the model to regenerate the manifest."
            )

        with TemporaryDirectory() as tmp:
            local_pkl = Path(tmp) / "model.pkl"
            self._download_to(artifact_model_pkl_key(pointer.version_id), local_pkl)

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
