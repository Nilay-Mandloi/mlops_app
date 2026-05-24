"""Serving-side storage port.

Read + write operations the serving app needs: resolving pointers,
downloading artifacts, and (for the publisher) uploading trigger files.
No locks, no version counters — those live on the training side.
Swap S3 for GCS/Azure Blob by writing a new adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path


class ArtifactStore(ABC):
    @abstractmethod
    def get_json(self, logical_key: str) -> dict | None:
        """Return parsed JSON at logical_key, or None if absent."""

    @abstractmethod
    def download_file(self, logical_key: str, local_path: Path | str) -> None:
        """Download logical_key to local_path."""

    @abstractmethod
    def exists(self, logical_key: str) -> bool:
        """Return True iff logical_key exists."""

    @abstractmethod
    def list_subkeys(self, prefix: str) -> Iterator[str]:
        """Yield immediate child names (one level deep) under prefix."""

    @abstractmethod
    def upload_file(
        self,
        local_path: Path | str,
        logical_key: str,
        *,
        content_type: str | None = None,
    ) -> None:
        """Upload local_path to logical_key."""

    @abstractmethod
    def put_bytes(
        self,
        data: bytes,
        logical_key: str,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        """Write raw bytes at logical_key."""

    @abstractmethod
    def delete(self, logical_key: str) -> None:
        """Delete logical_key. Idempotent — silent on missing."""


# Back-compat alias for callers that already import ReadOnlyArtifactStore.
ReadOnlyArtifactStore = ArtifactStore
