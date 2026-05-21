"""Request/response schemas for the price-forecast API.

These are local to THIS app — they describe the HTTP contract, not the
S3 contract. The S3 contract lives in contracts.py and is shared with
the training repo.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ..., description="Single feature row as a dict of column -> value."
    )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="One or more feature rows; upper bound enforced by APP_MAX_BATCH_SIZE.",
    )


class PredictResponse(BaseModel):
    prediction: float
    model_version: str


class BatchPredictResponse(BaseModel):
    predictions: list[float]
    model_version: str


class ModelInfoResponse(BaseModel):
    version_id: str
    run_id: str
    registry_version: str
    model_name: str
    model_type: str
    promoted_at: str | None
    loaded_at: float
    channel: str
    schema_contract: dict[str, Any] = Field(default_factory=dict)


class TriggerTrainRequest(BaseModel):
    """Request body for POST /trigger-train.

    Two mutually exclusive source modes:
    - S3 mode  (multi-user): supply dataset_s3_uri + params_s3_uri.
      The server downloads both files from S3 into TRIGGER_DATA_ROOT, then
      publishes them under a new trigger_id.  Use this when users upload their
      own dataset and params to S3 and call the endpoint.
    - Local mode (admin/CI):  supply dataset_path + params_path as paths that
      already exist on the server filesystem within TRIGGER_DATA_ROOT.
      Retained for backward-compatibility and internal tooling.
    Exactly one mode must be supplied.
    """

    model_config = ConfigDict(extra="forbid")

    # ── S3 mode ──────────────────────────────────────────────────────────────
    dataset_s3_uri: str | None = Field(
        None,
        description="s3://bucket/key to a CSV or Parquet dataset. Server downloads it.",
    )
    params_s3_uri: str | None = Field(
        None,
        description="s3://bucket/key to a params.yaml. Server downloads it.",
    )

    # ── Local mode (admin / backward-compat) ─────────────────────────────────
    dataset_path: str | None = Field(
        None,
        description="Server-local path to a dataset file (CSV or Parquet).",
    )
    params_path: str | None = Field(
        None,
        description="Server-local path to a params YAML file.",
    )

    # ── Common ────────────────────────────────────────────────────────────────
    model_family: str = Field(..., description="regression | classification | forecasting | ...")
    description: str = ""
    dataset_format: str | None = Field(
        None,
        description="Override format auto-detection: 'csv' or 'parquet'. Default: inferred from file extension.",
    )

    @model_validator(mode="after")
    def _check_source_mode(self) -> TriggerTrainRequest:
        has_s3 = bool(self.dataset_s3_uri) and bool(self.params_s3_uri)
        has_local = bool(self.dataset_path) and bool(self.params_path)
        if has_s3 and has_local:
            raise ValueError(
                "Provide either S3 URIs (dataset_s3_uri + params_s3_uri) "
                "or local paths (dataset_path + params_path), not both."
            )
        if not has_s3 and not has_local:
            raise ValueError(
                "Provide either (dataset_s3_uri + params_s3_uri) for the S3 source mode, "
                "or (dataset_path + params_path) for the local source mode."
            )
        # Partial S3 pairs are always wrong.
        if bool(self.dataset_s3_uri) != bool(self.params_s3_uri):
            raise ValueError("dataset_s3_uri and params_s3_uri must both be supplied together.")
        return self

    @field_validator("dataset_s3_uri", "params_s3_uri")
    @classmethod
    def _validate_s3_uri(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.startswith("s3://"):
            raise ValueError(f"Expected an s3:// URI, got: {v!r}")
        parts = v[5:].split("/", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(f"S3 URI must include both bucket and key: {v!r}")
        return v

    @field_validator("dataset_path", "params_path")
    @classmethod
    def _no_path_traversal(cls, v: str | None) -> str | None:
        if v is None:
            return v
        root = os.environ.get("TRIGGER_DATA_ROOT", "").strip().rstrip(os.sep)
        env = os.environ.get("ENV", "dev").strip().lower()
        if not root:
            if env == "prod":
                raise ValueError(
                    "TRIGGER_DATA_ROOT must be set before accepting trigger-train requests."
                )
            return v
        abs_path = os.path.realpath(os.path.abspath(v))
        abs_root = os.path.realpath(os.path.abspath(root))
        if not (abs_path == abs_root or abs_path.startswith(abs_root + os.sep)):
            raise ValueError(
                f"path {v!r} is outside TRIGGER_DATA_ROOT ({root!r}). Path traversal not permitted."
            )
        return v


class TriggerTrainResponse(BaseModel):
    trigger_id: str
    trigger_uri: str
