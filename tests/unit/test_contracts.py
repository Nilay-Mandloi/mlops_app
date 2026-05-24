"""Round-trip tests for the shared dataclasses."""

from __future__ import annotations

import pytest

from price_forecast.contracts import (
    SCHEMA_VERSION,
    ArtifactManifest,
    PointerFile,
    TriggerFile,
)


def test_manifest_round_trip():
    m = ArtifactManifest(
        category="mlops",
        project="product_dq",
        model_name="price_forecast",
        version=42,
        run_id="run-abc",
        registry_version="5",
        model_type="random_forest",
        schema_hash="abc123",
        schema_contract={"feature_columns": ["a", "b"]},
        artifact_checksums={"model.pkl": "deadbeef"},
    )
    payload = m.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["published_at"]
    assert payload["project"] == "product_dq"
    assert payload["version"] == 42

    back = ArtifactManifest.from_dict(payload)
    assert back.run_id == "run-abc"
    assert back.project == "product_dq"
    assert back.version == 42
    assert back.artifact_checksums == {"model.pkl": "deadbeef"}


def test_pointer_drops_none_optional_fields():
    p = PointerFile(
        category="mlops",
        project="product_dq",
        model_name="price_forecast",
        version=42,
        version_id="v42",
        run_id="run-abc",
        registry_version="5",
        manifest_uri="s3://bucket/x.json",
        status="stable",
    )
    payload = p.to_dict()
    assert "promoted_at" not in payload
    assert "promoted_by" not in payload
    assert "mlflow_tracking_uri" not in payload
    assert payload["updated_at"]
    assert payload["project"] == "product_dq"
    assert payload["version"] == 42


def test_trigger_round_trip():
    t = TriggerFile(
        trigger_id="t1",
        category="mlops",
        project="product_dq",
        model_name="price_forecast",
        model_family="regression",
        dataset_uri="s3://b/x.parquet",
        params_uri="s3://b/x.yaml",
    )
    payload = t.to_dict()
    back = TriggerFile.from_dict(payload)
    assert back.trigger_id == "t1"
    assert back.project == "product_dq"
    assert payload["created_at"]


def test_from_dict_tolerates_unknown_fields():
    """Forward-compat: older app loading a payload with new fields must not crash."""
    p = PointerFile.from_dict(
        {
            "category": "mlops",
            "project": "product_dq",
            "model_name": "price_forecast",
            "version": 42,
            "version_id": "v42",
            "run_id": "abc",
            "registry_version": "5",
            "manifest_uri": "s3://x",
            "status": "stable",
            "future_field": "should_be_ignored",
        }
    )
    assert p.version_id == "v42"
    assert p.project == "product_dq"


def test_manifest_missing_project_raises():
    with pytest.raises(TypeError):
        ArtifactManifest.from_dict(
            {
                "run_id": "r",
                "version": 1,
                "registry_version": "1",
                "model_type": "t",
                "schema_hash": "h",
                "artifact_checksums": {"model.pkl": "x"},
            }
        )
