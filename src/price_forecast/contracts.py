"""Data contracts that span the training/serving boundary.

These dataclasses define the *exact* JSON shape of the files that move
between the training repo and any consumer app (Flask, batch job, stream
processor). They are the single source of truth for what fields exist on
manifest.json, stable.json / latest.json (pointers), and trigger.json.

When the Flask app is moved to its own repo, these dataclasses (plus the
layout module) form the shared contract. Two repos that pin the same
versioned copy of this file cannot drift on schema. The serving repo
implements its own resolver against this contract; nothing in this
repo needs to read pointer files.

Versioning rules
----------------
- Adding an *optional* field is a non-breaking change.
- Renaming or removing a field is a breaking change — bump the
  ``SCHEMA_VERSION`` constant and update consumers.
- Producers should always emit the latest schema; consumers should tolerate
  missing optional fields for forward-compat.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1.0"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Manifest — written once per artifact version, immutable.
# Path: output/artifacts/v{N}/champion/manifest.json
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArtifactManifest:
    """Immutable per-version metadata. Sits next to model.pkl in S3.

    artifact_checksums is the authoritative checksum source. Consumers MUST
    verify sha256(model.pkl) against artifact_checksums["model.pkl"] before
    serving — never trust the byte stream alone.

    ``app_id`` is mandatory and identifies which app this artifact belongs
    to. The consumer-side loader cross-checks ``manifest.app_id`` against
    its configured ``APP_ID`` env var — a mismatched manifest is refused.
    """

    app_id: str
    run_id: str
    artifact_version: str
    registry_version: str
    model_name: str
    model_type: str
    schema_hash: str
    schema_contract: dict[str, Any] = field(default_factory=dict)
    registry_uri: str = ""
    artifact_checksums: dict[str, str] = field(default_factory=dict)
    published_at: str = ""
    schema_version: str = SCHEMA_VERSION
    git_commit: str | None = None
    code_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("published_at"):
            d["published_at"] = _utcnow_iso()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactManifest:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ---------------------------------------------------------------------------
# Pointer — mutable. The ONLY thing that flips when a new model is promoted.
# Path: output/registry/{model}/pointers/{stable|latest|canary}.json
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PointerFile:
    """Mutable pointer at a specific channel (stable / latest / canary / ...).

    Consumers poll this file (or get notified via S3 Event → SNS) and reload
    the model when version_id changes. The pointer points at an immutable
    manifest_uri, which in turn points at an immutable model.pkl.

    ``app_id`` is mandatory. A pointer payload travelling between training
    and serving carries its app scope explicitly so consumers can detect
    misrouted files (e.g. a copy-paste typo putting app1's pointer in
    app2's path).
    """

    app_id: str
    version_id: str
    run_id: str
    registry_version: str
    manifest_uri: str
    status: str  # "stable" | "candidate" | "canary" | "shadow"
    updated_at: str = ""
    promoted_at: str | None = None
    promoted_by: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d.get("updated_at"):
            d["updated_at"] = _utcnow_iso()
        # Drop None promoted_at/promoted_by to keep stable.json minimal.
        if d.get("promoted_at") is None:
            d.pop("promoted_at", None)
        if d.get("promoted_by") is None:
            d.pop("promoted_by", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PointerFile:
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


# ---------------------------------------------------------------------------
# Trigger — written by the consumer app, read by the training pipeline.
# Path: triggers/{trigger_id}/trigger.json (with dataset.parquet, params.yaml
# alongside)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriggerFile:
    """Input contract for a training run.

    Written by the Flask app (or any orchestrator) to S3. Training picks it
    up, reads dataset_uri + params_uri, runs the pipeline, and publishes
    artifacts back to the same app's bucket.
    """

    trigger_id: str
    app_id: str
    model_family: str  # "regression" | "classification" | "forecasting" | ...
    dataset_uri: str  # s3://.../triggers/{id}/dataset.parquet
    params_uri: str  # s3://.../triggers/{id}/params.yaml
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


# ---------------------------------------------------------------------------
# Channel names (stable values; do not rename without bumping SCHEMA_VERSION)
# ---------------------------------------------------------------------------

CHANNEL_STABLE = "stable"
CHANNEL_LATEST = "latest"
CHANNEL_CANARY = "canary"
CHANNEL_SHADOW = "shadow"

POINTER_STATUS_STABLE = "stable"
POINTER_STATUS_CANDIDATE = "candidate"
POINTER_STATUS_CANARY = "canary"
POINTER_STATUS_SHADOW = "shadow"
