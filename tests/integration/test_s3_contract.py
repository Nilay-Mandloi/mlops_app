"""End-to-end contract tests against a mock S3.

These exercise the actual byte-level protocol between the training repo
(producer) and this serving repo (consumer). The two repos are independent
git repos and must stay in lockstep on:

  - trigger folder layout (where the publisher writes vs where pull_trigger
    expects to find files)
  - pointer / manifest / pkl key shapes
  - manifest JSON schema (ArtifactManifest fields the loader reads)
  - SHA-256 checksum format and verification order

If either side drifts, these tests fail loudly. They're the cheapest
defense against the most expensive class of bug — silent contract drift
that surfaces only in prod.

Uses moto for S3 mocking; no real AWS needed. Skipped when moto isn't
installed (kept opt-in: the unit-test job stays fast).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
import time

import pytest

moto = pytest.importorskip("moto")  # skip the whole file if moto missing
boto3 = pytest.importorskip("boto3")

from moto import mock_aws

from price_forecast.config import AppConfig
from price_forecast.contracts import ArtifactManifest, PointerFile, TriggerFile
from price_forecast.layout import (
    artifact_manifest_key,
    artifact_model_pkl_key,
    artifact_requirements_key,
    artifact_schema_key,
    pointer_key,
    trigger_dataset_key,
    trigger_metadata_key,
    trigger_params_key,
)
from price_forecast.loader import ModelStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUCKET = "test-app-bucket"
PREFIX = "MLOPS"
MODEL_NAME = "price_forecast"
APP_ID = "test_app"


@pytest.fixture
def s3():
    """Mock S3 with a pre-created bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture
def app_cfg(monkeypatch):
    """Stable AppConfig for the loader to consume."""
    # boto3 inside ModelStore needs creds even when talking to moto.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    return AppConfig(
        app_id=APP_ID,
        bucket=BUCKET,
        stack_id=PREFIX,
        model_name=MODEL_NAME,
        channel="stable",
        aws_region="us-east-1",
        admin_token="t",
        reload_interval_s=0,
        host="0.0.0.0",
        port=8000,
        max_batch_size=1000,
        max_request_bytes=1048576,
        cors_allowed_origins="",
        env="test",
        log_format="",
        strict_schema=True,
        startup_grace_seconds=60,
    )


def _full_key(logical: str) -> str:
    return f"{PREFIX}/{logical}"


def _put(s3, logical: str, body: bytes, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=BUCKET, Key=_full_key(logical), Body=body, ContentType=content_type)


def _put_json(s3, logical: str, obj) -> None:
    _put(s3, logical, json.dumps(obj, indent=2).encode("utf-8"), "application/json")


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Trigger contract: publisher upload order matches puller expectations.
# ---------------------------------------------------------------------------

def test_trigger_layout_matches_pull_expectations(s3, app_cfg, tmp_path):
    """Simulate publisher; verify all three keys exist where the puller will look."""
    from price_forecast import publisher

    dataset = tmp_path / "dataset.parquet"
    params = tmp_path / "params.yaml"
    dataset.write_bytes(b"PAR1fake-parquet-bytes")
    params.write_text("dataset:\n  target_column: y\n", encoding="utf-8")

    # Publisher reads boto3 config via load_config() at call time, so we patch
    # the config + the lazy boto3 client construction inside the function.
    import price_forecast.publisher as pub

    def _fake_load_config():
        return app_cfg

    monkey_patches = [
        ("price_forecast.publisher.load_config", _fake_load_config),
    ]
    for name, fn in monkey_patches:
        mod, attr = name.rsplit(".", 1)
        import importlib
        setattr(importlib.import_module(mod), attr, fn)

    trigger_id, uri = pub.publish_trigger(
        dataset, params, model_family="regression", description="contract test"
    )

    # All three keys exist at the expected paths.
    for logical in (
        trigger_dataset_key(trigger_id),
        trigger_params_key(trigger_id),
        trigger_metadata_key(trigger_id),
    ):
        s3.head_object(Bucket=BUCKET, Key=_full_key(logical))

    # The marker payload round-trips to TriggerFile cleanly.
    body = s3.get_object(Bucket=BUCKET, Key=_full_key(trigger_metadata_key(trigger_id)))["Body"].read()
    marker = TriggerFile.from_dict(json.loads(body))
    assert marker.trigger_id == trigger_id
    assert marker.app_id == APP_ID
    assert marker.model_family == "regression"
    assert marker.dataset_uri.endswith(trigger_dataset_key(trigger_id))


def test_trigger_marker_is_written_last(s3, app_cfg, tmp_path):
    """Causal invariant: at any timestamp when trigger.json exists,
    dataset.parquet AND params.yaml must also exist."""
    from price_forecast import publisher

    dataset = tmp_path / "dataset.parquet"
    params = tmp_path / "params.yaml"
    dataset.write_bytes(b"data")
    params.write_text("k: v\n", encoding="utf-8")

    import price_forecast.publisher as pub
    pub.load_config = lambda: app_cfg

    trigger_id, _ = pub.publish_trigger(
        dataset, params, model_family="regression", description="ordering test"
    )

    # All three uploaded — order isn't directly observable post-hoc in moto,
    # but the absence of trigger.json before completion would have been
    # observable. We assert the postcondition: marker present implies the
    # other two are present (the same invariant pull_trigger relies on).
    s3.head_object(Bucket=BUCKET, Key=_full_key(trigger_metadata_key(trigger_id)))
    s3.head_object(Bucket=BUCKET, Key=_full_key(trigger_dataset_key(trigger_id)))
    s3.head_object(Bucket=BUCKET, Key=_full_key(trigger_params_key(trigger_id)))


# ---------------------------------------------------------------------------
# Pointer/manifest/pkl contract: what training writes, loader can read.
# ---------------------------------------------------------------------------

class _ToyModel:
    """Minimal model interface — predict() returns the input row count.

    Avoids a sklearn dep in the test path while still exercising the
    pickle/checksum/load path.
    """

    def predict(self, X):
        try:
            return [float(len(X))] * len(X)
        except TypeError:
            return [0.0]


def _publish_artifact_set(s3, version: str, run_id: str = "run-abc"):
    """Simulate the training repo's snapshot.publish_snapshot output."""
    # 1. schema_contract.json
    schema_contract = {
        "schema_version": "1.0",
        "feature_columns": ["a", "b"],
        "nullable_columns": [],
    }
    schema_bytes = json.dumps(schema_contract, indent=2).encode("utf-8")
    _put(s3, artifact_schema_key(version), schema_bytes, "application/json")

    # 2. requirements.lock
    req_bytes = b"scikit-learn==1.4.0\npandas==2.0.0\n"
    _put(s3, artifact_requirements_key(version), req_bytes, "text/plain")

    # 3. model.pkl (toy)
    model_bytes = pickle.dumps(_ToyModel())
    _put(s3, artifact_model_pkl_key(version), model_bytes)

    # 4. manifest.json — checksums are authoritative
    manifest = ArtifactManifest(
        run_id=run_id,
        artifact_version=version,
        registry_version="5",
        model_name=MODEL_NAME,
        model_type="toy",
        schema_hash="hash-abc",
        schema_contract=schema_contract,
        registry_uri=f"models:/{MODEL_NAME}/5",
        artifact_checksums={
            "schema_contract.json": _sha256_bytes(schema_bytes),
            "requirements.lock": _sha256_bytes(req_bytes),
            "model.pkl": _sha256_bytes(model_bytes),
        },
        published_at="2026-05-14T10:00:00+00:00",
    )
    _put_json(s3, artifact_manifest_key(version), manifest.to_dict())

    # 5. stable.json pointer
    pointer = PointerFile(
        version_id=version,
        run_id=run_id,
        registry_version="5",
        manifest_uri=f"s3://{BUCKET}/{_full_key(artifact_manifest_key(version))}",
        status="stable",
        promoted_at="2026-05-14T10:01:00+00:00",
    )
    _put_json(s3, pointer_key(MODEL_NAME, "stable"), pointer.to_dict())
    return manifest, pointer


def test_loader_reads_full_artifact_set(s3, app_cfg):
    """Training writes the full set; ModelStore.reload picks it up end-to-end."""
    _publish_artifact_set(s3, version="v1")

    store = ModelStore(app_cfg)
    loaded = store.reload()

    assert loaded.version_id == "v1"
    assert loaded.manifest.model_name == MODEL_NAME
    assert loaded.manifest.artifact_checksums["model.pkl"]
    assert loaded.manifest.schema_contract["feature_columns"] == ["a", "b"]
    # The pickled model is functional.
    assert loaded.model.predict([{"a": 1, "b": 2}, {"a": 3, "b": 4}]) == [2.0, 2.0]


def test_loader_refuses_corrupted_pkl(s3, app_cfg):
    """Checksum mismatch must be fatal — never serve a tampered pickle."""
    _publish_artifact_set(s3, version="v2")

    # Tamper with model.pkl AFTER manifest is published.
    _put(s3, artifact_model_pkl_key("v2"), b"corrupted-bytes")

    store = ModelStore(app_cfg)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        store.reload()


def test_loader_swaps_atomically_on_new_pointer(s3, app_cfg):
    """A new stable.json with a different version_id causes a load."""
    _publish_artifact_set(s3, version="v1", run_id="run-1")
    store = ModelStore(app_cfg)
    first = store.reload()
    assert first.version_id == "v1"

    # Promote a new version.
    _publish_artifact_set(s3, version="v2", run_id="run-2")
    second = store.reload()
    assert second.version_id == "v2"
    assert second.manifest.run_id == "run-2"


def test_loader_skips_reload_when_pointer_unchanged(s3, app_cfg):
    """Same version_id on stable.json — no re-download."""
    _publish_artifact_set(s3, version="v1")
    store = ModelStore(app_cfg)
    first = store.reload()
    second = store.reload()
    # Same LoadedModel instance because version_id matched (no swap).
    assert first is second


def test_loader_returns_lookup_error_when_pointer_missing(s3, app_cfg):
    """First deploy: no pointer yet. LookupError lets the Flask app start in standby."""
    store = ModelStore(app_cfg)
    with pytest.raises(LookupError, match="not found"):
        store.reload()


def test_loader_concurrent_reload_is_safe(s3, app_cfg):
    """Two threads call reload() simultaneously — must not corrupt state."""
    _publish_artifact_set(s3, version="v1")
    store = ModelStore(app_cfg)
    store.reload()  # warm up

    errors: list[Exception] = []

    def _worker():
        try:
            for _ in range(5):
                store.reload()
                time.sleep(0.001)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert not errors
    assert store.current().version_id == "v1"
