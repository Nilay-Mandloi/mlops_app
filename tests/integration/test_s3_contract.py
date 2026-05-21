"""End-to-end contract tests against a mock S3.

Exercises the byte-level protocol between the training repo (producer)
and this serving repo (consumer). The two repos are independent git
repos and must stay in lockstep on:

  - app_id is a mandatory path component AND a mandatory contract field
  - trigger folder layout: triggers/{app_id}/{trigger_id}/...
  - pointer / manifest / pkl key shapes under output/.../{app_id}/...
  - manifest + pointer JSON schemas (ArtifactManifest, PointerFile)
  - SHA-256 checksum verification order

If either side drifts, these tests fail loudly.

Uses moto for S3 mocking; no real AWS needed. Skipped when moto isn't
installed.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import threading
import time

import pytest

moto = pytest.importorskip("moto")
boto3 = pytest.importorskip("boto3")

from moto import mock_aws  # noqa: E402

from price_forecast.config import AppConfig  # noqa: E402
from price_forecast.contracts import ArtifactManifest, PointerFile, TriggerFile  # noqa: E402
from price_forecast.layout import (  # noqa: E402
    artifact_manifest_key,
    artifact_model_pkl_key,
    artifact_requirements_key,
    artifact_schema_key,
    pointer_key,
    trigger_dataset_key,
    trigger_metadata_key,
    trigger_params_key,
)
from price_forecast.loader import ModelStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BUCKET = "shared-mlops-bucket"
PREFIX = "MLOPS"
MODEL_NAME = "price_forecast"
APP_ID = "app1"
APP_ID_OTHER = "app2"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _make_cfg(app_id: str = APP_ID) -> AppConfig:
    return AppConfig(
        app_id=app_id,
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
        training_repo="",
        training_repo_token="",
        training_auto_promote=False,
    )


@pytest.fixture
def app_cfg(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    return _make_cfg()


def _full(logical: str) -> str:
    return f"{PREFIX}/{logical}"


def _put(s3, logical: str, body: bytes, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=BUCKET, Key=_full(logical), Body=body, ContentType=content_type)


def _put_json(s3, logical: str, obj) -> None:
    _put(s3, logical, json.dumps(obj, indent=2).encode("utf-8"), "application/json")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# Trigger contract
# ---------------------------------------------------------------------------


def test_trigger_layout_is_app_scoped(s3, app_cfg, tmp_path):
    """Publisher writes all three keys under triggers/{app_id}/{trigger_id}/."""
    from price_forecast import publisher

    dataset = tmp_path / "dataset.parquet"
    params = tmp_path / "params.yaml"
    dataset.write_bytes(b"PAR1fake-parquet")
    params.write_text("dataset:\n  target_column: y\n", encoding="utf-8")

    publisher.load_config = lambda: app_cfg

    trigger_id, uri = publisher.publish_trigger(dataset, params, model_family="regression")

    # The URI itself encodes the app scope.
    assert f"/triggers/{APP_ID}/{trigger_id}/" in uri

    # All three keys land at app-scoped paths.
    for logical in (
        trigger_dataset_key(APP_ID, trigger_id),
        trigger_params_key(APP_ID, trigger_id),
        trigger_metadata_key(APP_ID, trigger_id),
    ):
        assert f"/{APP_ID}/" in logical
        s3.head_object(Bucket=BUCKET, Key=_full(logical))


def test_trigger_marker_carries_app_id(s3, app_cfg, tmp_path):
    from price_forecast import publisher

    dataset = tmp_path / "dataset.parquet"
    params = tmp_path / "params.yaml"
    dataset.write_bytes(b"data")
    params.write_text("k: v\n", encoding="utf-8")

    publisher.load_config = lambda: app_cfg
    trigger_id, _ = publisher.publish_trigger(dataset, params, model_family="regression")

    body = s3.get_object(Bucket=BUCKET, Key=_full(trigger_metadata_key(APP_ID, trigger_id)))[
        "Body"
    ].read()
    marker = TriggerFile.from_dict(json.loads(body))
    assert marker.app_id == APP_ID
    assert marker.dataset_format == "parquet"
    assert marker.dataset_uri.endswith(trigger_dataset_key(APP_ID, trigger_id, "parquet"))


def test_csv_trigger_uses_csv_key_and_marker_format(s3, app_cfg, tmp_path):
    """A cleaned .csv input lands at .../dataset.csv and the marker says so."""
    from price_forecast import publisher

    dataset = tmp_path / "cleaned.csv"
    params = tmp_path / "params.yaml"
    dataset.write_text("id,y\n1,2.5\n3,4.5\n", encoding="utf-8")
    params.write_text("dataset:\n  target_column: y\n", encoding="utf-8")

    publisher.load_config = lambda: app_cfg
    trigger_id, uri = publisher.publish_trigger(dataset, params, model_family="regression")

    # The .csv key (not .parquet) exists.
    s3.head_object(Bucket=BUCKET, Key=_full(trigger_dataset_key(APP_ID, trigger_id, "csv")))

    # The marker declares csv, and dataset_uri points at the .csv key.
    body = s3.get_object(Bucket=BUCKET, Key=_full(trigger_metadata_key(APP_ID, trigger_id)))[
        "Body"
    ].read()
    marker = TriggerFile.from_dict(json.loads(body))
    assert marker.dataset_format == "csv"
    assert marker.dataset_uri.endswith(".csv")

    # Round-trip content: what we PUT is what's stored. Use splitlines so we
    # don't care about LF vs CRLF (Windows tempfile writes can introduce CR).
    fetched = (
        s3.get_object(Bucket=BUCKET, Key=_full(trigger_dataset_key(APP_ID, trigger_id, "csv")))[
            "Body"
        ]
        .read()
        .decode("utf-8")
    )
    assert fetched.splitlines()[0] == "id,y"


def test_explicit_dataset_format_override_wins_over_extension(s3, app_cfg, tmp_path):
    from price_forecast import publisher

    # Misnamed file — extension says .data, override says csv.
    dataset = tmp_path / "rows.data"
    params = tmp_path / "params.yaml"
    dataset.write_text("a,b\n1,2\n", encoding="utf-8")
    params.write_text("k: v\n", encoding="utf-8")

    publisher.load_config = lambda: app_cfg
    trigger_id, _ = publisher.publish_trigger(
        dataset, params, model_family="r", dataset_format="csv"
    )
    s3.head_object(Bucket=BUCKET, Key=_full(trigger_dataset_key(APP_ID, trigger_id, "csv")))


def test_unknown_dataset_extension_without_override_raises(s3, app_cfg, tmp_path):
    from price_forecast import publisher

    dataset = tmp_path / "rows.xyz"
    params = tmp_path / "params.yaml"
    dataset.write_text("noop\n", encoding="utf-8")
    params.write_text("k: v\n", encoding="utf-8")

    publisher.load_config = lambda: app_cfg
    with pytest.raises(ValueError, match="Cannot infer dataset_format"):
        publisher.publish_trigger(dataset, params, model_family="r")


def test_two_apps_publish_to_independent_trigger_dirs(s3, monkeypatch, tmp_path):
    """APP1 and APP2 triggers don't collide on the same trigger_id collision either."""
    from price_forecast import publisher

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    dataset = tmp_path / "d.parquet"
    params = tmp_path / "p.yaml"
    dataset.write_bytes(b"d")
    params.write_text("k: v\n", encoding="utf-8")

    publisher.load_config = lambda: _make_cfg(APP_ID)
    t1, uri1 = publisher.publish_trigger(dataset, params, model_family="r")

    publisher.load_config = lambda: _make_cfg(APP_ID_OTHER)
    t2, uri2 = publisher.publish_trigger(dataset, params, model_family="r")

    assert f"/triggers/{APP_ID}/" in uri1
    assert f"/triggers/{APP_ID_OTHER}/" in uri2
    # Both markers exist independently.
    s3.head_object(Bucket=BUCKET, Key=_full(trigger_metadata_key(APP_ID, t1)))
    s3.head_object(Bucket=BUCKET, Key=_full(trigger_metadata_key(APP_ID_OTHER, t2)))


# ---------------------------------------------------------------------------
# Pointer/manifest/pkl contract
# ---------------------------------------------------------------------------


class _ToyModel:
    def predict(self, X):
        try:
            return [float(len(X))] * len(X)
        except TypeError:
            return [0.0]


def _publish_artifact_set(s3, version: str, app_id: str = APP_ID, run_id: str = "run-abc"):
    schema_contract = {
        "schema_version": "1.0",
        "feature_columns": ["a", "b"],
        "nullable_columns": [],
    }
    schema_bytes = json.dumps(schema_contract, indent=2).encode("utf-8")
    _put(s3, artifact_schema_key(app_id, version), schema_bytes, "application/json")

    req_bytes = b"scikit-learn==1.4.0\n"
    _put(s3, artifact_requirements_key(app_id, version), req_bytes, "text/plain")

    model_bytes = pickle.dumps(_ToyModel())
    _put(s3, artifact_model_pkl_key(app_id, version), model_bytes)

    manifest = ArtifactManifest(
        app_id=app_id,
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
    _put_json(s3, artifact_manifest_key(app_id, version), manifest.to_dict())

    pointer = PointerFile(
        app_id=app_id,
        version_id=version,
        run_id=run_id,
        registry_version="5",
        manifest_uri=f"s3://{BUCKET}/{_full(artifact_manifest_key(app_id, version))}",
        status="stable",
        promoted_at="2026-05-14T10:01:00+00:00",
    )
    _put_json(s3, pointer_key(app_id, MODEL_NAME, "stable"), pointer.to_dict())
    return manifest, pointer


def test_loader_reads_full_artifact_set(s3, app_cfg):
    _publish_artifact_set(s3, version="v1")
    store = ModelStore(app_cfg)
    loaded = store.reload()

    assert loaded.version_id == "v1"
    assert loaded.manifest.app_id == APP_ID
    assert loaded.manifest.model_name == MODEL_NAME
    assert loaded.manifest.schema_contract["feature_columns"] == ["a", "b"]
    assert loaded.model.predict([{"a": 1, "b": 2}, {"a": 3, "b": 4}]) == [2.0, 2.0]


def test_loader_refuses_cross_app_pointer(s3, app_cfg):
    """Pointer at APP1's path but claiming app_id=APP2 must be refused."""
    # Stage an artifact set for APP2 first.
    _publish_artifact_set(s3, version="v1", app_id=APP_ID_OTHER, run_id="r2")
    # Now hand-place a pointer at APP1's pointer key but with APP2's payload.
    body = s3.get_object(Bucket=BUCKET, Key=_full(pointer_key(APP_ID_OTHER, MODEL_NAME, "stable")))[
        "Body"
    ].read()
    _put_json(s3, pointer_key(APP_ID, MODEL_NAME, "stable"), json.loads(body))

    store = ModelStore(app_cfg)
    with pytest.raises(RuntimeError, match="scope mismatch"):
        store.reload()


def test_loader_refuses_cross_app_manifest(s3, app_cfg):
    """Pointer is correct, but the manifest it points to claims a different app_id."""
    _publish_artifact_set(s3, version="v1")
    # Overwrite the manifest with one claiming APP2.
    bad_manifest = ArtifactManifest(
        app_id=APP_ID_OTHER,
        run_id="r",
        artifact_version="v1",
        registry_version="5",
        model_name=MODEL_NAME,
        model_type="toy",
        schema_hash="h",
        schema_contract={"feature_columns": ["a"]},
        artifact_checksums={"model.pkl": "x"},
    )
    _put_json(s3, artifact_manifest_key(APP_ID, "v1"), bad_manifest.to_dict())

    store = ModelStore(app_cfg)
    with pytest.raises(RuntimeError, match="Manifest scope mismatch"):
        store.reload()


def test_loader_refuses_cross_model_manifest(s3, app_cfg):
    """Pointer is correct, but the manifest belongs to a different model_name."""
    _publish_artifact_set(s3, version="v1")
    bad_manifest = ArtifactManifest(
        app_id=APP_ID,
        run_id="r",
        artifact_version="v1",
        registry_version="5",
        model_name="different_model",
        model_type="toy",
        schema_hash="h",
        schema_contract={"feature_columns": ["a"]},
        artifact_checksums={"model.pkl": "x"},
    )
    _put_json(s3, artifact_manifest_key(APP_ID, "v1"), bad_manifest.to_dict())

    store = ModelStore(app_cfg)
    with pytest.raises(RuntimeError, match="Manifest model mismatch"):
        store.reload()


def test_loader_refuses_corrupted_pkl(s3, app_cfg):
    _publish_artifact_set(s3, version="v2")
    _put(s3, artifact_model_pkl_key(APP_ID, "v2"), b"corrupted-bytes")
    store = ModelStore(app_cfg)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        store.reload()


def test_loader_swaps_atomically_on_new_pointer(s3, app_cfg):
    _publish_artifact_set(s3, version="v1", run_id="run-1")
    store = ModelStore(app_cfg)
    first = store.reload()
    assert first.version_id == "v1"

    _publish_artifact_set(s3, version="v2", run_id="run-2")
    second = store.reload()
    assert second.version_id == "v2"
    assert second.manifest.run_id == "run-2"


def test_loader_skips_reload_when_pointer_unchanged(s3, app_cfg):
    _publish_artifact_set(s3, version="v1")
    store = ModelStore(app_cfg)
    first = store.reload()
    second = store.reload()
    assert first is second


def test_loader_returns_lookup_error_when_pointer_missing(s3, app_cfg):
    store = ModelStore(app_cfg)
    with pytest.raises(LookupError, match="not found"):
        store.reload()


def test_loader_concurrent_reload_is_safe(s3, app_cfg):
    _publish_artifact_set(s3, version="v1")
    store = ModelStore(app_cfg)
    store.reload()

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


def test_loader_in_app2_ignores_app1_artifacts(s3, monkeypatch):
    """APP2's loader sees only APP2's pointer — APP1's artifacts are invisible."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    # APP1 has artifacts; APP2 has nothing.
    _publish_artifact_set(s3, version="v1", app_id=APP_ID)

    store2 = ModelStore(_make_cfg(APP_ID_OTHER))
    with pytest.raises(LookupError):
        store2.reload()


# ---------------------------------------------------------------------------
# publish_trigger — upload failure + rollback
# ---------------------------------------------------------------------------


class _FailingUploader:
    """Wraps a boto3 S3 client; makes upload_file raise on the Nth call."""

    def __init__(self, real_client, *, fail_on_call: int):
        self._real = real_client
        self._call_count = 0
        self._fail_on_call = fail_on_call

    def upload_file(self, **kwargs):
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError(f"simulated S3 failure on upload call {self._call_count}")
        return self._real.upload_file(**kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _trigger_keys_in_s3(s3, trigger_id: str) -> list[str]:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/triggers/{APP_ID}/{trigger_id}/")
    return [obj["Key"] for obj in resp.get("Contents", [])]


def test_publish_trigger_rolls_back_dataset_when_params_upload_fails(s3, app_cfg, tmp_path):
    """If the params upload fails, the already-uploaded dataset key must be deleted."""
    from price_forecast import publisher

    dataset = tmp_path / "dataset.parquet"
    params = tmp_path / "params.yaml"
    dataset.write_bytes(b"PAR1fake-parquet")
    params.write_text("k: v\n", encoding="utf-8")

    # Dataset is the 1st upload_file call; params is the 2nd.
    failing_client = _FailingUploader(s3, fail_on_call=2)

    with pytest.raises(RuntimeError, match="simulated S3 failure"):
        publisher.publish_trigger(
            dataset, params, model_family="regression", cfg=app_cfg, s3_client=failing_client
        )

    # S3 must be empty — the dataset key uploaded on call 1 must have been rolled back.
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/triggers/{APP_ID}/")
    orphaned = [obj["Key"] for obj in resp.get("Contents", [])]
    assert not orphaned, f"Expected empty S3 after rollback, found: {orphaned}"


def test_publish_trigger_rolls_back_all_keys_when_metadata_put_fails(s3, app_cfg, tmp_path):
    """If the metadata put_object fails, all three already-uploaded keys are rolled back."""
    from price_forecast import publisher

    dataset = tmp_path / "dataset.parquet"
    params = tmp_path / "params.yaml"
    dataset.write_bytes(b"PAR1fake-parquet")
    params.write_text("k: v\n", encoding="utf-8")

    real_put = s3.put_object
    put_count = [0]

    def failing_put(**kwargs):
        put_count[0] += 1
        if "trigger" in kwargs.get("Key", "") and put_count[0] == 1:
            raise RuntimeError("simulated S3 failure on trigger.json put")
        return real_put(**kwargs)

    s3.put_object = failing_put

    with pytest.raises(RuntimeError, match="simulated S3 failure"):
        publisher.publish_trigger(
            dataset, params, model_family="regression", cfg=app_cfg, s3_client=s3
        )

    # Both dataset and params keys uploaded before the put_object failure must be gone.
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/triggers/{APP_ID}/")
    orphaned = [obj["Key"] for obj in resp.get("Contents", [])]
    assert not orphaned, f"Expected empty S3 after rollback, found: {orphaned}"


def test_publish_trigger_dispatch_skipped_when_no_training_repo(s3, app_cfg, tmp_path):
    """publish_trigger completes successfully when training_repo is unset (dispatch skipped)."""
    from dataclasses import replace

    from price_forecast import publisher

    cfg_no_dispatch = replace(app_cfg, training_repo="", training_repo_token="")
    dataset = tmp_path / "d.parquet"
    params = tmp_path / "p.yaml"
    dataset.write_bytes(b"data")
    params.write_text("k: v\n", encoding="utf-8")

    trigger_id, uri = publisher.publish_trigger(
        dataset, params, model_family="regression", cfg=cfg_no_dispatch, s3_client=s3
    )

    # All three keys must exist — upload succeeded.
    keys = _trigger_keys_in_s3(s3, trigger_id)
    assert len(keys) == 3, f"Expected 3 trigger keys, found: {keys}"


def test_dispatch_failure_writes_failed_marker_to_s3(s3, app_cfg, tmp_path):
    """If GitHub dispatch fails after S3 uploads, a failed.json marker is written
    so /trigger-status/<id> can return 'failed' instead of hanging in 'pending'.
    """
    from dataclasses import replace

    from price_forecast import publisher
    from price_forecast.layout import trigger_failure_key

    # Point at a non-existent repo to make dispatch fail with a RuntimeError.
    cfg = replace(
        app_cfg,
        training_repo="org/nonexistent-repo",
        training_repo_token="fake-token",
    )
    dataset = tmp_path / "d.parquet"
    params = tmp_path / "p.yaml"
    dataset.write_bytes(b"data")
    params.write_text("k: v\n", encoding="utf-8")

    # Patch _dispatch_training so it raises without hitting the real GitHub API.
    original_dispatch = publisher._dispatch_training

    def _failing_dispatch(trigger_id, cfg):
        raise RuntimeError("simulated dispatch failure")

    publisher._dispatch_training = _failing_dispatch
    try:
        with pytest.raises(RuntimeError, match="simulated dispatch failure"):
            publisher.publish_trigger(
                dataset, params, model_family="regression", cfg=cfg, s3_client=s3
            )
    finally:
        publisher._dispatch_training = original_dispatch

    # S3 uploads committed before dispatch — all three payload keys are present.
    # ADDITIONALLY: a failed.json marker must exist so /trigger-status returns "failed".
    # Find the trigger_id by listing everything under triggers/{app_id}/
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{PREFIX}/triggers/{APP_ID}/")
    all_keys = [obj["Key"] for obj in resp.get("Contents", [])]

    trigger_ids = set()
    for key in all_keys:
        parts = key.split(f"triggers/{APP_ID}/")
        if len(parts) == 2:
            trigger_ids.add(parts[1].split("/")[0])

    assert len(trigger_ids) == 1, f"Expected exactly one trigger folder, got: {trigger_ids}"
    tid = trigger_ids.pop()

    failure_full_key = f"{PREFIX}/{trigger_failure_key(APP_ID, tid)}"
    body = s3.get_object(Bucket=BUCKET, Key=failure_full_key)["Body"].read()
    payload = json.loads(body)
    assert payload["status"] == "failed"
    assert "trigger_id" in payload
    assert "dispatch" in payload["reason"].lower() or "failed" in payload["reason"].lower()
