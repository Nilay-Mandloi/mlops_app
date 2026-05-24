"""Data contracts that span the training/serving boundary.

These dataclasses mirror the JSON Schemas in ``schemas/*.v1.json`` exactly.
The schemas are the source of truth; these classes are a Python convenience
for producers/consumers in this repo. The same schemas live at
``s3://{bucket}/_schemas/`` so any language can validate the same files.

Versioning rules
----------------
- Adding an *optional* field is non-breaking. Bump nothing.
- Renaming or removing a field is breaking: cut a new ``schemas/X.v2.json``,
  introduce a new dataclass, and run both in parallel during migration.
- ``schema_version`` is pinned to the schema file version, not edited freely.

Identifier model
----------------
Tenants are identified by ``(category, project, model_name)``. ``category``
also encodes the bucket name (``{category}-artifacts``). Inside a project,
each model has its own version sequence ``v1, v2, ...`` independent of
MLflow's registry version counter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1.0"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ArtifactManifest:
    """Immutable per-version metadata. Sits next to model.pkl in S3.

    Path: ``s3://{category}-artifacts/{project}/{model_name}/v{version}/manifest.json``

    ``artifact_checksums`` is authoritative — consumers MUST verify
    ``sha256(model.pkl)`` against ``artifact_checksums["model.pkl"]`` before
    serving. Mismatched (category, project, model_name) on read = reject.
    """

    category: str
    project: str
    model_name: str
    version: int
    run_id: str
    registry_version: str
    model_type: str
    schema_hash: str
    artifact_checksums: dict[str, str]
    schema_contract: dict[str, Any] = field(default_factory=dict)
    published_at: str = ""
    # Legacy field names — populated by ANY ExperimentTracker adapter, not just
    # MLflow. The names are kept stable to avoid a cross-repo schema break;
    # treat their values as opaque backend-specific URLs.
    mlflow_tracking_uri: str | None = None
    mlflow_run_url: str | None = None
    mlflow_model_url: str | None = None
    git_commit: str | None = None
    code_version: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("published_at"):
            d["published_at"] = _utcnow_iso()
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(frozen=True)
class PointerFile:
    """Mutable per-model channel pointer.

    Path: ``s3://{category}-artifacts/{project}/{model_name}/{channel}.json``
    where channel ∈ {stable, latest, canary, shadow}.

    Flipped atomically inside the promotion lock. Consumers poll this file
    (or get notified via S3 events) and reload when ``version`` changes.
    """

    category: str
    project: str
    model_name: str
    version: int
    version_id: str
    run_id: str
    registry_version: str
    manifest_uri: str
    status: str
    updated_at: str = ""
    # Legacy field names — populated by ANY ExperimentTracker adapter, not just
    # MLflow. The names are kept stable to avoid a cross-repo schema break;
    # treat their values as opaque backend-specific URLs.
    mlflow_tracking_uri: str | None = None
    mlflow_run_url: str | None = None
    mlflow_model_url: str | None = None
    promoted_at: str | None = None
    promoted_by: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("updated_at"):
            d["updated_at"] = _utcnow_iso()
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointerFile:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass(frozen=True)
class TriggerFile:
    """Input contract for a training run.

    Written by the REST app (or any orchestrator) to
    ``s3://{category}-artifacts/_triggers/{project}/{trigger_id}/trigger.json``
    alongside ``dataset.{csv|parquet}`` and ``params.yaml``.
    """

    trigger_id: str
    category: str
    project: str
    model_name: str
    model_family: str
    dataset_uri: str
    params_uri: str
    dataset_format: str = "parquet"
    requested_by: str = ""
    created_at: str = ""
    description: str = ""
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("created_at"):
            d["created_at"] = _utcnow_iso()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TriggerFile:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


CHANNEL_STABLE = "stable"
CHANNEL_LATEST = "latest"
CHANNEL_CANARY = "canary"
CHANNEL_SHADOW = "shadow"

VALID_CHANNELS: frozenset[str] = frozenset(
    {CHANNEL_STABLE, CHANNEL_LATEST, CHANNEL_CANARY, CHANNEL_SHADOW}
)

VALID_STATUSES: frozenset[str] = frozenset({"stable", "candidate", "canary", "shadow"})

VALID_MODEL_FAMILIES: frozenset[str] = frozenset(
    {
        "regression",
        "classification",
        "forecasting",
        "clustering",
        "ranking",
        "nlp",
        "vision",
        "other",
    }
)
