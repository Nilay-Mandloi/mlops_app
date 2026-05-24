"""S3 key layout — single source of truth for object key derivation.

Functions return logical keys relative to the bucket. The bucket itself
encodes the category: ``{category}-artifacts`` (e.g. ``mlops-artifacts``).
Code that builds full ``s3://`` URIs composes ``bucket_for(category)`` with
these keys.

Logical key shapes
------------------
    {project}/{model_name}/v{N}/{model.pkl|manifest.json|schema_contract.json|requirements.lock}
    {project}/{model_name}/{stable|latest|canary|shadow}.json
    {project}/{model_name}/_pointer_history/{ts}_{channel}_v{N}_{uid}.json
    {project}/{model_name}/_counter.json
    {project}/{model_name}/_reports/v{N}/{report_name}
    _locks/{project}/{model_name}/{lock_name}.lock
    _triggers/{project}/{trigger_id}/{dataset.parquet|params.yaml|trigger.json|running.json|failed.json}
    _schemas/{schema_name}.v{N}.json
    _feature_store/{project}/{dataset_name}/manifests/{version}.json

Prefixes starting with ``_`` (``_locks``, ``_triggers``, ``_schemas``,
``_pointer_history``, ``_counter``, ``_reports``, ``_feature_store``) are
operational and live outside the project tree visually. They cannot
collide with project names because project names match ``^[a-z0-9]...``.
"""

from __future__ import annotations

import re
import uuid

CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,30}$")
PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
MODEL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
TRIGGER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

SUPPORTED_DATASET_FORMATS: frozenset[str] = frozenset({"csv", "parquet"})


def _check(value: str, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or not pattern.match(value):
        raise ValueError(f"{name} must match {pattern.pattern}; got {value!r}")
    return value


def _check_project(project: str) -> str:
    return _check(project, PROJECT_RE, "project")


def _check_model_name(model_name: str) -> str:
    return _check(model_name, MODEL_NAME_RE, "model_name")


def _check_category(category: str) -> str:
    return _check(category, CATEGORY_RE, "category")


def _check_trigger_id(trigger_id: str) -> str:
    return _check(trigger_id, TRIGGER_ID_RE, "trigger_id")


def _check_dataset_format(fmt: str) -> str:
    if fmt not in SUPPORTED_DATASET_FORMATS:
        raise ValueError(
            f"dataset_format must be one of {sorted(SUPPORTED_DATASET_FORMATS)}; got {fmt!r}"
        )
    return fmt


def _vtag(version: int | str) -> str:
    s = str(version).lstrip("v")
    if not s.isdigit() or int(s) < 1:
        raise ValueError(f"version must be a positive integer; got {version!r}")
    return f"v{int(s)}"


# ---------------------------------------------------------------------------
# Bucket name (category encodes the bucket)
# ---------------------------------------------------------------------------


def bucket_for(category: str) -> str:
    """Default bucket name for a category. Override via settings if needed."""
    return f"{_check_category(category)}-artifacts"


# ---------------------------------------------------------------------------
# Per-model artifact tree (immutable, versioned)
# ---------------------------------------------------------------------------


def model_root(project: str, model_name: str) -> str:
    return f"{_check_project(project)}/{_check_model_name(model_name)}"


def artifact_root(project: str, model_name: str, version: int | str) -> str:
    return f"{model_root(project, model_name)}/{_vtag(version)}"


def artifact_key(project: str, model_name: str, version: int | str, filename: str) -> str:
    return f"{artifact_root(project, model_name, version)}/{filename}"


def model_pkl_key(project: str, model_name: str, version: int | str) -> str:
    return artifact_key(project, model_name, version, "model.pkl")


def manifest_key(project: str, model_name: str, version: int | str) -> str:
    return artifact_key(project, model_name, version, "manifest.json")


def schema_contract_key(project: str, model_name: str, version: int | str) -> str:
    return artifact_key(project, model_name, version, "schema_contract.json")


def requirements_key(project: str, model_name: str, version: int | str) -> str:
    return artifact_key(project, model_name, version, "requirements.lock")


def counter_key(project: str, model_name: str) -> str:
    """Mutable monotonic version counter for this (project, model_name)."""
    return f"{model_root(project, model_name)}/_counter.json"


# ---------------------------------------------------------------------------
# Per-model pointers (mutable — the only mutable surface)
# ---------------------------------------------------------------------------


def pointer_key(project: str, model_name: str, channel: str) -> str:
    """e.g. {project}/{model_name}/stable.json"""
    if not channel or "/" in channel or channel.startswith("_"):
        raise ValueError(f"channel must be a simple name; got {channel!r}")
    return f"{model_root(project, model_name)}/{channel}.json"


def pointer_history_key(
    project: str,
    model_name: str,
    channel: str,
    version: int | str,
    timestamp: str,
) -> str:
    """Immutable audit trail entry for a pointer flip.

    UUID suffix prevents collision when two flips fire within the same second.
    """
    uid = uuid.uuid4().hex[:8]
    return (
        f"{model_root(project, model_name)}/_pointer_history/"
        f"{timestamp}_{channel}_{_vtag(version)}_{uid}.json"
    )


# ---------------------------------------------------------------------------
# Reports (per-model, per-version)
# ---------------------------------------------------------------------------


def report_key(project: str, model_name: str, version: int | str, report_name: str) -> str:
    return f"{model_root(project, model_name)}/_reports/{_vtag(version)}/{report_name}"


# ---------------------------------------------------------------------------
# Promotion locks (per-model so different models in the same project don't block)
# ---------------------------------------------------------------------------


def lock_key(project: str, model_name: str, lock_name: str) -> str:
    if not lock_name or "/" in lock_name:
        raise ValueError(f"lock_name must be a simple name; got {lock_name!r}")
    return f"_locks/{_check_project(project)}/{_check_model_name(model_name)}/{lock_name}.lock"


# ---------------------------------------------------------------------------
# Schemas (canonical contract files published by scripts/publish_schemas.py)
# ---------------------------------------------------------------------------


def schema_key(schema_name: str) -> str:
    """e.g. _schemas/pointer.v1.json"""
    if not re.match(r"^[a-z][a-z0-9_]*\.v[1-9][0-9]*\.json$", schema_name):
        raise ValueError(f"schema_name must look like 'pointer.v1.json'; got {schema_name!r}")
    return f"_schemas/{schema_name}"


# ---------------------------------------------------------------------------
# Feature store (per-project dataset snapshots)
# ---------------------------------------------------------------------------


def feature_store_manifest_key(project: str, dataset_name: str, version: str) -> str:
    return f"_feature_store/{_check_project(project)}/{dataset_name}/manifests/{version}.json"


# ---------------------------------------------------------------------------
# Triggers (per-project queue; trigger.json carries category/project/model_name)
# ---------------------------------------------------------------------------


def trigger_root(project: str, trigger_id: str) -> str:
    return f"_triggers/{_check_project(project)}/{_check_trigger_id(trigger_id)}"


def trigger_dataset_key(project: str, trigger_id: str, dataset_format: str = "parquet") -> str:
    return f"{trigger_root(project, trigger_id)}/dataset.{_check_dataset_format(dataset_format)}"


def trigger_params_key(project: str, trigger_id: str) -> str:
    return f"{trigger_root(project, trigger_id)}/params.yaml"


def trigger_metadata_key(project: str, trigger_id: str) -> str:
    return f"{trigger_root(project, trigger_id)}/trigger.json"


def trigger_running_key(project: str, trigger_id: str) -> str:
    return f"{trigger_root(project, trigger_id)}/running.json"


def trigger_failure_key(project: str, trigger_id: str) -> str:
    return f"{trigger_root(project, trigger_id)}/failed.json"
