"""End-to-end contract tests against a mock S3.

Exercises the byte-level protocol between the training repo (producer) and this
serving repo (consumer). Tenant identity = (category, project, model_name, version).
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

from price_forecast.adapters.s3_store import S3ReadStore  # noqa: E402
from price_forecast.config import AppConfig  # noqa: E402
from price_forecast.contracts import ArtifactManifest, PointerFile, TriggerFile  # noqa: E402
from price_forecast.layout import (  # noqa: E402
    manifest_key,
    model_pkl_key,
    pointer_key,
    requirements_key,
    schema_contract_key,
    trigger_dataset_key,
    trigger_metadata_key,
    trigger_params_key,
)
from price_forecast.loader import ModelStore  # noqa: E402


def _make_store(cfg: AppConfig, client) -> ModelStore:
    return ModelStore(cfg, S3ReadStore(bucket=cfg.bucket, prefix=cfg.prefix, client=client))


BUCKET = "mlops-artifacts"
PREFIX = ""
CATEGORY = "mlops"
PROJECT = "product_dq"
PROJECT_OTHER = "product_bi"
MODEL = "price_forecast"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _make_cfg(project: str = PROJECT) -> AppConfig:
    return AppConfig(
        category=CATEGORY,
        project=project,
        model_name=MODEL,
        bucket=BUCKET,
        prefix=PREFIX,
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
    return logical if not PREFIX else f"{PREFIX}/{logical}"


def _put(s3, logical: str, body: bytes, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=BUCKET, Key=_full(logical), Body=body, ContentType=content_type)


def _put_json(s3, logical: str, obj) -> None:
    _put(s3, logical, json.dumps(obj, indent=2).encode("utf-8"), "application/json")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_trigger_layout_is_project_scoped(s3, app_cfg, tmp_path):
    from price_forecast import publisher

    dataset = tmp_path / "dataset.parquet"
    params = tmp_path / "params.yaml"
    dataset.write_bytes(b"PAR1fake-parquet")
    params.write_text("dataset:\n  target_column: y\n", encoding="utf-8")

    publisher.load_config = lambda: app_cfg
    trigger_id, uri = publisher.publish_trigger(dataset, params, model_family="regression")

    assert f"/_triggers/{PROJECT}/{trigger_id}/" in uri
    for logical in (
        trigger_dataset_key(PROJECT, trigger_id),
        trigger_params_key(PROJECT, trigger_id),
        trigger_metadata_key(PROJECT, trigger_id),
    ):
        assert logical.startswith(f"_triggers/{PROJECT}/")
        s3.head_object(Bucket=BUCKET, Key=_full(logical))


def test_trigger_marker_carries_tenant(s3, app_cfg, tmp_path):
    from price_forecast import publisher

    dataset = tmp_path / "dataset.parquet"
    params = tmp_path / "params.yaml"
    dataset.write_bytes(b"data")
    params.write_text("k: v\n", encoding="utf-8")

    publisher.load_config = lambda: app_cfg
    trigger_id, _ = publisher.publish_trigger(dataset, params, model_family="regression")

    body = s3.get_object(Bucket=BUCKET, Key=_full(trigger_metadata_key(PROJECT, trigger_id)))[
        "Body"
    ].read()
    marker = TriggerFile.from_dict(json.loads(body))
    assert marker.category == CATEGORY
    assert marker.project == PROJECT
    assert marker.model_name == MODEL
    assert marker.dataset_format == "parquet"


def test_csv_trigger_uses_csv_key(s3, app_cfg, tmp_path):
    from price_forecast import publisher

    dataset = tmp_path / "cleaned.csv"
    params = tmp_path / "params.yaml"
    dataset.write_text("id,y\n1,2.5\n3,4.5\n", encoding="utf-8")
    params.write_text("dataset:\n  target_column: y\n", encoding="utf-8")

    publisher.load_config = lambda: app_cfg
    trigger_id, _ = publisher.publish_trigger(dataset, params, model_family="regression")

    s3.head_object(Bucket=BUCKET, Key=_full(trigger_dataset_key(PROJECT, trigger_id, "csv")))
    body = s3.get_object(Bucket=BUCKET, Key=_full(trigger_metadata_key(PROJECT, trigger_id)))[
        "Body"
    ].read()
    marker = TriggerFile.from_dict(json.loads(body))
    assert marker.dataset_format == "csv"


def test_two_projects_publish_to_independent_trigger_dirs(s3, monkeypatch, tmp_path):
    from price_forecast import publisher

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    dataset = tmp_path / "d.parquet"
    params = tmp_path / "p.yaml"
    dataset.write_bytes(b"d")
    params.write_text("k: v\n", encoding="utf-8")

    publisher.load_config = lambda: _make_cfg(PROJECT)
    t1, uri1 = publisher.publish_trigger(dataset, params, model_family="r")

    publisher.load_config = lambda: _make_cfg(PROJECT_OTHER)
    t2, uri2 = publisher.publish_trigger(dataset, params, model_family="r")

    assert f"/_triggers/{PROJECT}/" in uri1
    assert f"/_triggers/{PROJECT_OTHER}/" in uri2
    s3.head_object(Bucket=BUCKET, Key=_full(trigger_metadata_key(PROJECT, t1)))
    s3.head_object(Bucket=BUCKET, Key=_full(trigger_metadata_key(PROJECT_OTHER, t2)))


class _ToyModel:
    def predict(self, X):
        try:
            return [float(len(X))] * len(X)
        except TypeError:
            return [0.0]


def _publish_artifact_set(s3, version: int, project: str = PROJECT, run_id: str = "run-abc"):
    schema_contract = {
        "schema_version": "1.0",
        "feature_columns": ["a", "b"],
        "nullable_columns": [],
    }
    schema_bytes = json.dumps(schema_contract, indent=2).encode("utf-8")
    _put(s3, schema_contract_key(project, MODEL, version), schema_bytes, "application/json")

    req_bytes = b"scikit-learn==1.4.0\n"
    _put(s3, requirements_key(project, MODEL, version), req_bytes, "text/plain")

    model_bytes = pickle.dumps(_ToyModel())
    _put(s3, model_pkl_key(project, MODEL, version), model_bytes)

    manifest = ArtifactManifest(
        category=CATEGORY,
        project=project,
        model_name=MODEL,
        version=version,
        run_id=run_id,
        registry_version="5",
        model_type="toy",
        schema_hash="hash-abc",
        schema_contract=schema_contract,
        artifact_checksums={
            "schema_contract.json": _sha256_bytes(schema_bytes),
            "requirements.lock": _sha256_bytes(req_bytes),
            "model.pkl": _sha256_bytes(model_bytes),
        },
        published_at="2026-05-14T10:00:00+00:00",
    )
    _put_json(s3, manifest_key(project, MODEL, version), manifest.to_dict())

    pointer = PointerFile(
        category=CATEGORY,
        project=project,
        model_name=MODEL,
        version=version,
        version_id=f"v{version}",
        run_id=run_id,
        registry_version="5",
        manifest_uri=f"s3://{BUCKET}/{_full(manifest_key(project, MODEL, version))}",
        status="stable",
        promoted_at="2026-05-14T10:01:00+00:00",
    )
    _put_json(s3, pointer_key(project, MODEL, "stable"), pointer.to_dict())
    return manifest, pointer


def test_loader_reads_full_artifact_set(s3, app_cfg):
    _publish_artifact_set(s3, version=1)
    store = _make_store(app_cfg, s3)
    loaded = store.reload()

    assert loaded.version_id == "v1"
    assert loaded.manifest.project == PROJECT
    assert loaded.manifest.model_name == MODEL
    assert loaded.model.predict([{"a": 1, "b": 2}, {"a": 3, "b": 4}]) == [2.0, 2.0]


def test_loader_refuses_cross_project_pointer(s3, app_cfg):
    """Pointer at PROJECT's path but claiming project=PROJECT_OTHER must be refused."""
    _publish_artifact_set(s3, version=1, project=PROJECT_OTHER, run_id="r2")
    body = s3.get_object(Bucket=BUCKET, Key=_full(pointer_key(PROJECT_OTHER, MODEL, "stable")))[
        "Body"
    ].read()
    _put_json(s3, pointer_key(PROJECT, MODEL, "stable"), json.loads(body))

    store = _make_store(app_cfg, s3)
    with pytest.raises(RuntimeError, match="scope mismatch"):
        store.reload()


def test_loader_refuses_cross_project_manifest(s3, app_cfg):
    _publish_artifact_set(s3, version=1)
    bad_manifest = ArtifactManifest(
        category=CATEGORY,
        project=PROJECT_OTHER,
        model_name=MODEL,
        version=1,
        run_id="r",
        registry_version="5",
        model_type="toy",
        schema_hash="h",
        schema_contract={"feature_columns": ["a"]},
        artifact_checksums={"model.pkl": "x" * 64},
    )
    _put_json(s3, manifest_key(PROJECT, MODEL, 1), bad_manifest.to_dict())

    store = _make_store(app_cfg, s3)
    with pytest.raises(RuntimeError, match="scope mismatch"):
        store.reload()


def test_loader_refuses_corrupted_pkl(s3, app_cfg):
    _publish_artifact_set(s3, version=2)
    _put(s3, model_pkl_key(PROJECT, MODEL, 2), b"corrupted-bytes")
    store = _make_store(app_cfg, s3)
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        store.reload()


def test_loader_swaps_atomically_on_new_pointer(s3, app_cfg):
    _publish_artifact_set(s3, version=1, run_id="run-1")
    store = _make_store(app_cfg, s3)
    first = store.reload()
    assert first.version_id == "v1"

    _publish_artifact_set(s3, version=2, run_id="run-2")
    second = store.reload()
    assert second.version_id == "v2"
    assert second.manifest.run_id == "run-2"


def test_loader_skips_reload_when_pointer_unchanged(s3, app_cfg):
    _publish_artifact_set(s3, version=1)
    store = _make_store(app_cfg, s3)
    first = store.reload()
    second = store.reload()
    assert first is second


def test_loader_raises_lookup_error_when_pointer_missing(s3, app_cfg):
    store = _make_store(app_cfg, s3)
    with pytest.raises(LookupError, match="not found"):
        store.reload()


def test_loader_concurrent_reload_is_safe(s3, app_cfg):
    _publish_artifact_set(s3, version=1)
    store = _make_store(app_cfg, s3)
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


def test_loader_in_project2_ignores_project1_artifacts(s3, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")

    _publish_artifact_set(s3, version=1, project=PROJECT)

    store2 = _make_store(_make_cfg(PROJECT_OTHER), s3)
    with pytest.raises(LookupError):
        store2.reload()


class _FailingStore:
    """ArtifactStore wrapper that fails the Nth upload_file/put_bytes call."""

    def __init__(self, real_store, *, fail_on_call: int):
        self._real = real_store
        self._call_count = 0
        self._fail_on_call = fail_on_call

    def _maybe_fail(self):
        self._call_count += 1
        if self._call_count == self._fail_on_call:
            raise RuntimeError(f"simulated S3 failure on upload call {self._call_count}")

    def upload_file(self, *args, **kwargs):
        self._maybe_fail()
        return self._real.upload_file(*args, **kwargs)

    def put_bytes(self, *args, **kwargs):
        self._maybe_fail()
        return self._real.put_bytes(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _NoopOrchestrator:
    calls: list = []

    def dispatch_training(self, **kwargs):
        self.calls.append(kwargs)


class _FailingOrchestrator:
    def dispatch_training(self, **_kwargs):
        raise RuntimeError("simulated dispatch failure")


def test_publish_trigger_rolls_back_dataset_when_params_upload_fails(s3, app_cfg, tmp_path):
    from price_forecast import publisher
    from price_forecast.adapters.s3_store import S3ReadStore

    dataset = tmp_path / "dataset.parquet"
    params = tmp_path / "params.yaml"
    dataset.write_bytes(b"PAR1fake-parquet")
    params.write_text("k: v\n", encoding="utf-8")

    failing_store = _FailingStore(
        S3ReadStore(bucket=BUCKET, prefix=PREFIX, client=s3),
        fail_on_call=2,
    )

    with pytest.raises(RuntimeError, match="simulated S3 failure"):
        publisher.publish_trigger(
            dataset,
            params,
            model_family="regression",
            cfg=app_cfg,
            store=failing_store,
            orchestrator=_NoopOrchestrator(),
        )

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"_triggers/{PROJECT}/")
    orphaned = [obj["Key"] for obj in resp.get("Contents", [])]
    assert not orphaned, f"Expected empty S3 after rollback, found: {orphaned}"


def test_publish_trigger_dispatch_skipped_when_no_training_repo(s3, app_cfg, tmp_path):
    from price_forecast import publisher
    from price_forecast.adapters.s3_store import S3ReadStore

    dataset = tmp_path / "d.parquet"
    params = tmp_path / "p.yaml"
    dataset.write_bytes(b"data")
    params.write_text("k: v\n", encoding="utf-8")

    orchestrator = _NoopOrchestrator()
    trigger_id, _ = publisher.publish_trigger(
        dataset,
        params,
        model_family="regression",
        cfg=app_cfg,
        store=S3ReadStore(bucket=BUCKET, prefix=PREFIX, client=s3),
        orchestrator=orchestrator,
    )

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"_triggers/{PROJECT}/{trigger_id}/")
    keys = [obj["Key"] for obj in resp.get("Contents", [])]
    assert len(keys) == 3, f"Expected 3 trigger keys, found: {keys}"
    assert len(orchestrator.calls) == 1
    assert orchestrator.calls[0]["trigger_id"] == trigger_id


def test_dispatch_failure_writes_failed_marker_to_s3(s3, app_cfg, tmp_path):
    from price_forecast import publisher
    from price_forecast.adapters.s3_store import S3ReadStore
    from price_forecast.layout import trigger_failure_key

    dataset = tmp_path / "d.parquet"
    params = tmp_path / "p.yaml"
    dataset.write_bytes(b"data")
    params.write_text("k: v\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="simulated dispatch failure"):
        publisher.publish_trigger(
            dataset,
            params,
            model_family="regression",
            cfg=app_cfg,
            store=S3ReadStore(bucket=BUCKET, prefix=PREFIX, client=s3),
            orchestrator=_FailingOrchestrator(),
        )

    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"_triggers/{PROJECT}/")
    all_keys = [obj["Key"] for obj in resp.get("Contents", [])]
    trigger_ids = set()
    for key in all_keys:
        parts = key.split(f"_triggers/{PROJECT}/")
        if len(parts) == 2:
            trigger_ids.add(parts[1].split("/")[0])
    assert len(trigger_ids) == 1
    tid = trigger_ids.pop()

    failure_full_key = trigger_failure_key(PROJECT, tid)
    body = s3.get_object(Bucket=BUCKET, Key=failure_full_key)["Body"].read()
    payload = json.loads(body)
    assert payload["status"] == "failed"
