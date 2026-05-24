"""Model loader — reads pointer.json, downloads model.pkl, verifies checksum.

The serving consumer of the training/serving contract. Depends ONLY on
ReadOnlyArtifactStore + contracts + layout — no boto3, no MLflow, no cloud
SDK. Swap S3 for GCS by passing a different adapter.

Thread-safe: a single ``ModelStore`` instance is shared across Flask request
threads. State swaps are atomic via a single replace of the internal
``_current`` reference (Python's GIL guarantees atomicity of attr assignment).
"""

from __future__ import annotations

import hashlib
import pickle
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from loguru import logger

from price_forecast.config import AppConfig
from price_forecast.contracts import ArtifactManifest, PointerFile
from price_forecast.layout import manifest_key, model_pkl_key, pointer_key
from price_forecast.ports.storage import ReadOnlyArtifactStore


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
        store = ModelStore(config, artifact_store)
        store.reload()              # eager initial load
        loaded = store.current()    # never None after first successful reload
    """

    def __init__(self, config: AppConfig, artifact_store: ReadOnlyArtifactStore) -> None:
        self._cfg = config
        self._store = artifact_store
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()

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
        """Reload without raising. Returns None if pointer is absent (first deploy)."""
        try:
            return self.reload()
        except LookupError as exc:
            logger.info("No pointer yet — serving will start in standby mode: {}", exc)
            return None

    def reload(self) -> LoadedModel:
        """Re-read the pointer, download the artifact if version changed, swap atomically."""
        project = self._cfg.project
        model_name = self._cfg.model_name
        pointer_data = self._store.get_json(pointer_key(project, model_name, self._cfg.channel))
        if pointer_data is None:
            raise LookupError(
                f"Pointer '{self._cfg.channel}' for {project}/{model_name} not found. "
                "Has the training pipeline run and promoted a version yet?"
            )
        pointer = PointerFile.from_dict(pointer_data)
        if pointer.project != project or pointer.model_name != model_name:
            raise RuntimeError(
                f"Pointer scope mismatch: file claims {pointer.project}/{pointer.model_name} "
                f"but this app is configured for {project}/{model_name}. Refusing to load."
            )
        with self._lock:
            if self._current is not None and self._current.version_id == pointer.version_id:
                logger.debug("Pointer unchanged at {} — skipping reload.", pointer.version_id)
                return self._current

        manifest_data = self._store.get_json(manifest_key(project, model_name, pointer.version))
        if not manifest_data:
            raise RuntimeError(
                f"Manifest for {project}/{model_name} v{pointer.version} missing. "
                "Refusing to load without a manifest to verify checksums."
            )
        manifest = ArtifactManifest.from_dict(manifest_data)
        if manifest.project != project or manifest.model_name != model_name:
            raise RuntimeError(
                f"Manifest scope mismatch: {manifest.project}/{manifest.model_name} "
                f"vs configured {project}/{model_name}. Refusing to load."
            )

        expected_checksum = manifest.artifact_checksums.get("model.pkl")
        if not expected_checksum:
            raise RuntimeError(f"Manifest for v{pointer.version} has no checksum for model.pkl.")

        with TemporaryDirectory() as tmp:
            local_pkl = Path(tmp) / "model.pkl"
            self._store.download_file(
                model_pkl_key(project, model_name, pointer.version), local_pkl
            )
            actual = _sha256_file(local_pkl)
            if actual != expected_checksum:
                raise RuntimeError(
                    f"Checksum mismatch for v{pointer.version} model.pkl: "
                    f"expected {expected_checksum}, got {actual}. Refusing to serve."
                )
            with local_pkl.open("rb") as fh:
                model = pickle.load(fh)  # noqa: S301 — verified checksum

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
