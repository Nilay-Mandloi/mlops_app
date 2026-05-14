"""S3 layout — single source of truth for object key paths.

These functions return *logical keys*: paths relative to the bucket+prefix
configured for an app. The `S3Store` adapter prepends the bucket-level prefix.

Multi-tenant scoping
--------------------
Each consuming app gets its own bucket + stack-level prefix:

    s3://{app_bucket}/{stack_prefix}/output/...

For the MLOps training pipeline:
    bucket        = MLFLOW_S3_BUCKET   (e.g. "app1_bucket", "app2_bucket")
    stack_prefix  = S3_ARTIFACTS_PREFIX (e.g. "MLOPS")

Switching an app to a different training stack (e.g. Azure ML) only changes
the stack_prefix — the key shapes here stay the same:

    s3://app1_bucket/MLOPS/output/artifacts/v42/champion/model.pkl
    s3://app1_bucket/azure/output/artifacts/v42/champion/model.pkl

Logical key shapes
------------------
    output/artifacts/v{N}/champion/{model.pkl|manifest.json|...}
    output/reports/v{N}/{report_name}
    output/registry/{model_name}/pointers/{pointer_name}.json
    output/registry/{model_name}/pointers/history/{ts}_{pointer}_v{N}.json
    output/locks/{lock_name}.lock
    feature-store/input/{dataset_name}/manifests/{version}.json
    triggers/{trigger_id}/{dataset.parquet|params.yaml|trigger.json}
"""

from __future__ import annotations

STACK_PREFIX_MLOPS = "MLOPS"


def _vtag(version: str | int) -> str:
    """Coerce '42' / 42 / 'v42' to 'v42'."""
    s = str(version)
    return s if s.startswith("v") else f"v{s}"


# ---------------------------------------------------------------------------
# Artifact snapshots (immutable, versioned)
# ---------------------------------------------------------------------------

def artifact_key(artifact_version: str | int, filename: str) -> str:
    return f"output/artifacts/{_vtag(artifact_version)}/champion/{filename}"


def artifact_model_pkl_key(artifact_version: str | int) -> str:
    return artifact_key(artifact_version, "model.pkl")


def artifact_manifest_key(artifact_version: str | int) -> str:
    return artifact_key(artifact_version, "manifest.json")


def artifact_requirements_key(artifact_version: str | int) -> str:
    return artifact_key(artifact_version, "requirements.lock")


def artifact_schema_key(artifact_version: str | int) -> str:
    return artifact_key(artifact_version, "schema_contract.json")


# ---------------------------------------------------------------------------
# Reports (immutable, versioned)
# ---------------------------------------------------------------------------

def report_key(artifact_version: str | int, report_name: str) -> str:
    return f"output/reports/{_vtag(artifact_version)}/{report_name}"


# ---------------------------------------------------------------------------
# Registry pointers (mutable — the only mutable surface in the store)
# ---------------------------------------------------------------------------

def registry_key(model_name: str, filename: str) -> str:
    return f"output/registry/{model_name}/{filename}"


def pointer_key(model_name: str, pointer_name: str) -> str:
    """Mutable pointer key for stable.json / latest.json / etc."""
    return f"output/registry/{model_name}/pointers/{pointer_name}.json"


def pointer_history_key(
    model_name: str,
    pointer_name: str,
    version: str | int,
    timestamp: str,
) -> str:
    """Immutable audit trail for pointer flips: rollback by copying back."""
    return (
        f"output/registry/{model_name}/pointers/history/"
        f"{timestamp}_{pointer_name}_{_vtag(version)}.json"
    )


# ---------------------------------------------------------------------------
# Lock keys (promotion concurrency)
# ---------------------------------------------------------------------------

def lock_key(lock_name: str) -> str:
    return f"output/locks/{lock_name}.lock"


# ---------------------------------------------------------------------------
# Feature store (dataset snapshots)
# ---------------------------------------------------------------------------

def feature_store_input_manifest_key(dataset_name: str, version: str) -> str:
    return f"feature-store/input/{dataset_name}/manifests/{version}.json"


def artifact_counter_key(dataset_name: str) -> str:
    """Mutable counter used by get_next_serial_version."""
    return f"output/artifacts/{dataset_name}/counter.json"


# ---------------------------------------------------------------------------
# Triggers (input contract from app → training)
# ---------------------------------------------------------------------------

def trigger_dataset_key(trigger_id: str) -> str:
    return f"triggers/{trigger_id}/dataset.parquet"


def trigger_params_key(trigger_id: str) -> str:
    return f"triggers/{trigger_id}/params.yaml"


def trigger_metadata_key(trigger_id: str) -> str:
    return f"triggers/{trigger_id}/trigger.json"
