"""Unit tests for the /trigger-status/<trigger_id> endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from price_forecast.app import create_app
from price_forecast.config import AppConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_CFG = AppConfig(
    app_id="testapp",
    bucket="test-bucket",
    stack_id="MLOPS",
    model_name="price_forecast",
    channel="stable",
    aws_region="us-east-1",
    admin_token="secret",
    reload_interval_s=0,  # disable background reloader in tests
    host="0.0.0.0",
    port=8000,
    max_batch_size=1000,
    max_request_bytes=1048576,
    cors_allowed_origins="",
    env="dev",
    log_format="",
    strict_schema=True,
    startup_grace_seconds=120,
    training_repo="",
    training_repo_token="",
    training_auto_promote=False,
)

_VALID_TRIGGER_ID = "20260519T120000Z_abcdef12"
_TRIGGER_JSON = {"trigger_id": _VALID_TRIGGER_ID, "app_id": "testapp"}
_STABLE_V1 = {
    "version_id": "v1",
    "updated_at": "2026-05-19T11:00:00+00:00",
}
_STABLE_V2 = {
    "version_id": "v2",
    "updated_at": "2026-05-19T13:00:00+00:00",  # after trigger creation
}
_ADMIN_HDR = {"X-Admin-Token": "secret"}


@pytest.fixture()
def mock_store():
    store = MagicMock()
    store.current_or_none.return_value = None
    store.try_reload.return_value = None
    return store


@pytest.fixture()
def client(mock_store):
    app = create_app(cfg=_BASE_CFG, store=mock_store)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, mock_store


# ---------------------------------------------------------------------------
# Format validation
# ---------------------------------------------------------------------------

def test_invalid_trigger_id_returns_400(client):
    c, _ = client
    resp = c.get("/trigger-status/bad-id!!", headers=_ADMIN_HDR)
    assert resp.status_code == 400
    assert "invalid trigger_id format" in resp.get_json()["error"]


def test_missing_trigger_returns_404(client):
    c, store = client
    # trigger.json absent → _get_json returns None for trigger_metadata_key
    store._get_json.return_value = None
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}", headers=_ADMIN_HDR)
    assert resp.status_code == 404
    assert _VALID_TRIGGER_ID in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# Status: failed
# ---------------------------------------------------------------------------

def test_returns_failed_when_failure_marker_exists(client):
    c, store = client
    failure_payload = {"status": "failed", "reason": "pipeline blew up", "trigger_id": _VALID_TRIGGER_ID}

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        if "failed.json" in key:
            return failure_payload
        return None

    store._get_json.side_effect = _get_json
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}", headers=_ADMIN_HDR)
    data = resp.get_json()
    assert resp.status_code == 200
    assert data["status"] == "failed"
    assert data["reason"] == "pipeline blew up"
    assert data["current_version_id"] is None


def test_failed_marker_overrides_completed_pointer(client):
    """Even if a pointer exists, failure marker wins — the pointer belongs to a different trigger."""
    c, store = client

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        if "failed.json" in key:
            return {"reason": "rank_compare failed"}
        if "stable.json" in key:
            return _STABLE_V2
        return None

    store._get_json.side_effect = _get_json
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}", headers=_ADMIN_HDR)
    assert resp.get_json()["status"] == "failed"


# ---------------------------------------------------------------------------
# Status: running
# ---------------------------------------------------------------------------

def test_returns_running_when_running_marker_exists_no_pointer(client):
    c, store = client

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        if "failed.json" in key:
            return None
        if "running.json" in key:
            return {"status": "running", "trigger_id": _VALID_TRIGGER_ID}
        return None  # no stable.json yet

    store._get_json.side_effect = _get_json
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}", headers=_ADMIN_HDR)
    data = resp.get_json()
    assert resp.status_code == 200
    assert data["status"] == "running"


def test_returns_running_when_pointer_predates_trigger(client):
    """running.json present + pointer exists but hasn't moved → still running."""
    c, store = client

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        if "failed.json" in key:
            return None
        if "running.json" in key:
            return {"status": "running"}
        if "stable.json" in key:
            return _STABLE_V1  # updated_at 11:00 < trigger 12:00
        return None

    store._get_json.side_effect = _get_json
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}", headers=_ADMIN_HDR)
    assert resp.get_json()["status"] == "running"


def test_failed_overrides_running(client):
    """If both failed.json and running.json are present (shouldn't happen but be defensive), failed wins."""
    c, store = client

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        if "failed.json" in key:
            return {"reason": "rank_compare crashed"}
        if "running.json" in key:
            return {"status": "running"}
        return None

    store._get_json.side_effect = _get_json
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}", headers=_ADMIN_HDR)
    assert resp.get_json()["status"] == "failed"


# ---------------------------------------------------------------------------
# Status: pending
# ---------------------------------------------------------------------------

def test_returns_pending_when_no_pointer(client):
    c, store = client

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        return None  # failed.json and stable.json both absent

    store._get_json.side_effect = _get_json
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}", headers=_ADMIN_HDR)
    data = resp.get_json()
    assert resp.status_code == 200
    assert data["status"] == "pending"
    assert data["current_version_id"] is None


def test_returns_pending_with_baseline_when_version_unchanged(client):
    c, store = client

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        if "failed.json" in key:
            return None
        if "stable.json" in key:
            return _STABLE_V1
        return None

    store._get_json.side_effect = _get_json
    resp = c.get(
        f"/trigger-status/{_VALID_TRIGGER_ID}?baseline=v1",
        headers=_ADMIN_HDR,
    )
    data = resp.get_json()
    assert data["status"] == "pending"
    assert data["current_version_id"] == "v1"


# ---------------------------------------------------------------------------
# Status: completed
# ---------------------------------------------------------------------------

def test_returns_completed_with_baseline_when_version_changed(client):
    c, store = client

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        if "failed.json" in key:
            return None
        if "stable.json" in key:
            return _STABLE_V2
        return None

    store._get_json.side_effect = _get_json
    resp = c.get(
        f"/trigger-status/{_VALID_TRIGGER_ID}?baseline=v1",
        headers=_ADMIN_HDR,
    )
    data = resp.get_json()
    assert data["status"] == "completed"
    assert data["current_version_id"] == "v2"


def test_returns_completed_via_timestamp_fallback(client):
    """Without baseline, completion is inferred from pointer.updated_at > trigger creation time."""
    c, store = client
    # Trigger was created at 2026-05-19T12:00:00Z; pointer updated_at is 13:00:00Z (after).

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        if "failed.json" in key:
            return None
        if "stable.json" in key:
            return _STABLE_V2
        return None

    store._get_json.side_effect = _get_json
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}", headers=_ADMIN_HDR)
    assert resp.get_json()["status"] == "completed"


def test_returns_pending_via_timestamp_fallback_when_pointer_older(client):
    """Without baseline, pending when pointer.updated_at <= trigger creation time."""
    c, store = client
    # Trigger created at 2026-05-19T12:00:00Z; pointer at 11:00:00Z (before).

    def _get_json(key):
        if "trigger.json" in key:
            return _TRIGGER_JSON
        if "failed.json" in key:
            return None
        if "stable.json" in key:
            return _STABLE_V1
        return None

    store._get_json.side_effect = _get_json
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}", headers=_ADMIN_HDR)
    assert resp.get_json()["status"] == "pending"


# ---------------------------------------------------------------------------
# Admin auth
# ---------------------------------------------------------------------------

def test_trigger_status_requires_admin_token(client):
    c, _ = client
    resp = c.get(f"/trigger-status/{_VALID_TRIGGER_ID}")
    assert resp.status_code == 401
