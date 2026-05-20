"""Request/response schemas for the price-forecast API.

These are local to THIS app — they describe the HTTP contract, not the
S3 contract. The S3 contract lives in contracts.py and is shared with
the training repo.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ..., description="Single feature row as a dict of column -> value."
    )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(
        ..., min_length=1, description="One or more feature rows; upper bound enforced by APP_MAX_BATCH_SIZE."
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
    model_config = ConfigDict(extra="forbid")

    dataset_path: str = Field(..., description="Server-local path to a dataset file (CSV or Parquet).")
    params_path: str = Field(..., description="Server-local path to a params YAML file.")
    model_family: str = Field(..., description="regression | classification | forecasting | ...")
    description: str = ""
    dataset_format: str | None = Field(
        None,
        description="Override format auto-detection: 'csv' or 'parquet'. Default: inferred from file extension.",
    )

    @field_validator("dataset_path", "params_path")
    @classmethod
    def _no_path_traversal(cls, v: str) -> str:
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
                f"path {v!r} is outside TRIGGER_DATA_ROOT ({root!r}). "
                "Path traversal not permitted."
            )
        return v


class TriggerTrainResponse(BaseModel):
    trigger_id: str
    trigger_uri: str
