"""S3 layout — single source of truth for object key paths.

These functions return *logical keys*: paths relative to the bucket+prefix
configured for the stack. The S3Store adapter prepends the bucket-level
prefix (typically the stack id, e.g. "MLOPS").

App scoping
-----------
Every path is scoped by ``app_id`` so multiple apps share one bucket
without collisions:

    s3://{bucket}/{stack_prefix}/output/artifacts/{app_id}/v{N}/...

This is required (not optional). Single-tenant deployments still set
``app_id`` to a constant value. Code that omits app_id is a bug —
the function signatures enforce it.

Stack scoping
-------------
The ``{stack_prefix}`` lives in the bucket-level prefix, not here. Switch
a deployment to a parallel Azure ML stack by setting ``STACK_ID=azure``;
the layout module produces identical keys, the adapter prepends the new
prefix:

    s3://{bucket}/MLOPS/output/artifacts/{app_id}/v42/champion/model.pkl
    s3://{bucket}/azure/output/artifacts/{app_id}/v42/champion/model.pkl

Logical key shapes
------------------
    output/artifacts/{app_id}/v{N}/champion/{model.pkl|manifest.json|...}
    output/artifacts/{app_id}/{dataset_name}/counter.json
    output/reports/{app_id}/v{N}/{report_name}
    output/registry/{app_id}/{model_name}/pointers/{pointer_name}.json
    output/registry/{app_id}/{model_name}/pointers/history/{ts}_{pointer}_v{N}.json
    output/locks/{app_id}/{lock_name}.lock
    feature-store/{app_id}/input/{dataset_name}/manifests/{version}.json
    triggers/{app_id}/{trigger_id}/{dataset.parquet|params.yaml|trigger.json}
"""

from __future__ import annotations

STACK_PREFIX_MLOPS = "MLOPS"


def _vtag(version: str | int) -> str:
    """Coerce '42' / 42 / 'v42' to 'v42'."""
    s = str(version)
    return s if s.startswith("v") else f"v{s}"


def _check_app_id(app_id: str) -> str:
    """Validate app_id is non-empty and safe to put in an S3 key.

    Empty app_id would silently produce "//" in the path and risk
    cross-tenant collisions. Forward-slash characters would let a
    caller escape their scope. Reject both up front.
    """
    if not app_id or not isinstance(app_id, str):
        raise ValueError(f"app_id is required and must be a non-empty string; got {app_id!r}")
    if "/" in app_id or app_id.startswith(".") or app_id.strip() != app_id:
        raise ValueError(
            f"app_id must not contain '/' or surrounding whitespace and must not start "
            f"with '.'; got {app_id!r}"
        )
    return app_id


# ---------------------------------------------------------------------------
# Artifact snapshots (immutable, versioned, app-scoped)
# ---------------------------------------------------------------------------

def artifact_key(app_id: str, artifact_version: str | int, filename: str) -> str:
    return f"output/artifacts/{_check_app_id(app_id)}/{_vtag(artifact_version)}/champion/{filename}"


def artifact_model_pkl_key(app_id: str, artifact_version: str | int) -> str:
    return artifact_key(app_id, artifact_version, "model.pkl")


def artifact_manifest_key(app_id: str, artifact_version: str | int) -> str:
    return artifact_key(app_id, artifact_version, "manifest.json")


def artifact_requirements_key(app_id: str, artifact_version: str | int) -> str:
    return artifact_key(app_id, artifact_version, "requirements.lock")


def artifact_schema_key(app_id: str, artifact_version: str | int) -> str:
    return artifact_key(app_id, artifact_version, "schema_contract.json")


def artifact_counter_key(app_id: str, dataset_name: str) -> str:
    """Mutable counter used by get_next_serial_version.

    App-scoped: each app has its own monotonic v{N} sequence regardless
    of what other apps are doing. dataset_name sub-scopes within an app
    in case one app trains multiple distinct models.
    """
    return f"output/artifacts/{_check_app_id(app_id)}/{dataset_name}/counter.json"


# ---------------------------------------------------------------------------
# Reports (immutable, versioned, app-scoped)
# ---------------------------------------------------------------------------

def report_key(app_id: str, artifact_version: str | int, report_name: str) -> str:
    return f"output/reports/{_check_app_id(app_id)}/{_vtag(artifact_version)}/{report_name}"


# ---------------------------------------------------------------------------
# Registry pointers (mutable — the only mutable surface in the store)
# ---------------------------------------------------------------------------

def registry_key(app_id: str, model_name: str, filename: str) -> str:
    return f"output/registry/{_check_app_id(app_id)}/{model_name}/{filename}"


def pointer_key(app_id: str, model_name: str, pointer_name: str) -> str:
    """Mutable pointer key for stable.json / latest.json / canary.json / etc."""
    return (
        f"output/registry/{_check_app_id(app_id)}/{model_name}/pointers/{pointer_name}.json"
    )


def pointer_history_key(
    app_id: str,
    model_name: str,
    pointer_name: str,
    version: str | int,
    timestamp: str,
) -> str:
    """Immutable audit trail for pointer flips: rollback by copying back."""
    return (
        f"output/registry/{_check_app_id(app_id)}/{model_name}/pointers/history/"
        f"{timestamp}_{pointer_name}_{_vtag(version)}.json"
    )


# ---------------------------------------------------------------------------
# Lock keys (promotion concurrency)
# ---------------------------------------------------------------------------

def lock_key(app_id: str, lock_name: str) -> str:
    return f"output/locks/{_check_app_id(app_id)}/{lock_name}.lock"


# ---------------------------------------------------------------------------
# Feature store (dataset snapshots) — app-scoped
# ---------------------------------------------------------------------------

def feature_store_input_manifest_key(app_id: str, dataset_name: str, version: str) -> str:
    return (
        f"feature-store/{_check_app_id(app_id)}/input/{dataset_name}/manifests/{version}.json"
    )


# ---------------------------------------------------------------------------
# Triggers (input contract from app → training) — app-scoped
# ---------------------------------------------------------------------------

def trigger_root(app_id: str, trigger_id: str) -> str:
    """Folder prefix for a single trigger."""
    return f"triggers/{_check_app_id(app_id)}/{trigger_id}"


def trigger_dataset_key(app_id: str, trigger_id: str) -> str:
    return f"{trigger_root(app_id, trigger_id)}/dataset.parquet"


def trigger_params_key(app_id: str, trigger_id: str) -> str:
    return f"{trigger_root(app_id, trigger_id)}/params.yaml"


def trigger_metadata_key(app_id: str, trigger_id: str) -> str:
    return f"{trigger_root(app_id, trigger_id)}/trigger.json"
