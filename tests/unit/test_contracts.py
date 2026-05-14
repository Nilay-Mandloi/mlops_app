"""Round-trip tests for the shared dataclasses.

Verifies the JSON shape that crosses the training/serving boundary stays
backwards-compatible: a payload written by the training repo must be
loadable by this app.
"""

from __future__ import annotations

from price_forecast.contracts import (
    SCHEMA_VERSION,
    ArtifactManifest,
    PointerFile,
    TriggerFile,
)


def test_manifest_round_trip():
    m = ArtifactManifest(
        run_id="run-abc",
        artifact_version="v42",
        registry_version="5",
        model_name="price_forecast",
        model_type="random_forest",
        schema_hash="abc123",
        schema_contract={"feature_columns": ["a", "b"]},
        artifact_checksums={"model.pkl": "deadbeef"},
    )
    payload = m.to_dict()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["published_at"]  # filled automatically

    back = ArtifactManifest.from_dict(payload)
    assert back.run_id == "run-abc"
    assert back.artifact_checksums == {"model.pkl": "deadbeef"}


def test_pointer_drops_none_optional_fields():
    p = PointerFile(
        version_id="v42",
        run_id="run-abc",
        registry_version="5",
        manifest_uri="s3://bucket/x.json",
        status="stable",
    )
    payload = p.to_dict()
    assert "promoted_at" not in payload  # None should be stripped
    assert "promoted_by" not in payload
    assert payload["updated_at"]  # auto-filled


def test_trigger_round_trip():
    t = TriggerFile(
        trigger_id="t1",
        app_id="price-forecast",
        model_family="regression",
        dataset_uri="s3://b/x.parquet",
        params_uri="s3://b/x.yaml",
    )
    payload = t.to_dict()
    back = TriggerFile.from_dict(payload)
    assert back.trigger_id == "t1"
    assert back.app_id == "price-forecast"
    assert payload["created_at"]


def test_from_dict_tolerates_unknown_fields():
    """Forward-compat: an older app loading a payload with new fields must not crash."""
    p = PointerFile.from_dict({
        "version_id": "v42",
        "run_id": "abc",
        "registry_version": "5",
        "manifest_uri": "s3://x",
        "status": "stable",
        "future_field_we_dont_know_yet": "should_be_ignored",
    })
    assert p.version_id == "v42"
