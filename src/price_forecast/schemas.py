"""Request/response schemas for the price-forecast API.

These are local to THIS app — they describe the HTTP contract, not the
S3 contract. The S3 contract lives in contracts.py and is shared with
the training repo.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, Any] = Field(
        ..., description="Single feature row as a dict of column -> value."
    )


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=10_000, description="Up to 10k feature rows."
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

    dataset_path: str = Field(..., description="Server-local path to a parquet file.")
    params_path: str = Field(..., description="Server-local path to a params YAML file.")
    model_family: str = Field(..., description="regression | classification | forecasting | ...")
    description: str = ""


class TriggerTrainResponse(BaseModel):
    trigger_id: str
    trigger_uri: str
