from __future__ import annotations

import json
import urllib.error
from dataclasses import replace
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from price_forecast.config import AppConfig
from price_forecast.publisher import _dispatch_training

# ---------------------------------------------------------------------------
# Minimal AppConfig fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def base_cfg() -> AppConfig:
    return AppConfig(
        app_id="testapp",
        bucket="test-bucket",
        stack_id="MLOPS",
        model_name="price_forecast",
        channel="stable",
        aws_region="us-east-1",
        admin_token="secret",
        reload_interval_s=30,
        host="0.0.0.0",
        port=8000,
        max_batch_size=1000,
        max_request_bytes=1048576,
        cors_allowed_origins="",
        env="dev",
        log_format="",
        strict_schema=True,
        startup_grace_seconds=120,
        training_repo="my-org/mlops",
        training_repo_token="ghp_testtoken",
        training_auto_promote=False,
    )


# ---------------------------------------------------------------------------
# _dispatch_training — skip path
# ---------------------------------------------------------------------------

def test_dispatch_skipped_when_no_repo(base_cfg, caplog):
    cfg = replace(base_cfg, training_repo="", training_repo_token="ghp_x")
    with patch("urllib.request.urlopen") as mock_open:
        _dispatch_training("tid-001", cfg)
        mock_open.assert_not_called()


def test_dispatch_skipped_when_no_token(base_cfg, caplog):
    cfg = replace(base_cfg, training_repo="org/repo", training_repo_token="")
    with patch("urllib.request.urlopen") as mock_open:
        _dispatch_training("tid-002", cfg)
        mock_open.assert_not_called()


# ---------------------------------------------------------------------------
# _dispatch_training — success path
# ---------------------------------------------------------------------------

def test_dispatch_sends_correct_payload(base_cfg):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 204

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        _dispatch_training("tid-123", base_cfg)

    mock_open.assert_called_once()
    req = mock_open.call_args[0][0]

    # Correct URL
    assert req.full_url == "https://api.github.com/repos/my-org/mlops/dispatches"

    # Authorization header
    assert req.get_header("Authorization") == "Bearer ghp_testtoken"
    assert req.get_header("Accept") == "application/vnd.github+json"

    # JSON payload
    body = json.loads(req.data.decode("utf-8"))
    assert body["event_type"] == "train-model"
    assert body["client_payload"]["trigger_id"] == "tid-123"
    assert body["client_payload"]["auto_promote"] is False


def test_dispatch_auto_promote_flag_forwarded(base_cfg):
    cfg = replace(base_cfg, training_auto_promote=True)
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 204

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        _dispatch_training("tid-456", cfg)
    body = json.loads(mock_open.call_args[0][0].data.decode("utf-8"))
    assert body["client_payload"]["auto_promote"] is True


# ---------------------------------------------------------------------------
# _dispatch_training — error paths
# ---------------------------------------------------------------------------

def test_dispatch_raises_on_http_error(base_cfg):
    http_err = urllib.error.HTTPError(
        url="https://api.github.com/repos/my-org/mlops/dispatches",
        code=401,
        msg="Unauthorized",
        hdrs=MagicMock(),  # type: ignore[arg-type]
        fp=BytesIO(b""),
    )
    with patch("urllib.request.urlopen", side_effect=http_err), pytest.raises(
        RuntimeError, match="GitHub dispatch failed: HTTP 401"
    ):
        _dispatch_training("tid-789", base_cfg)


def test_dispatch_raises_on_network_error(base_cfg):
    with patch(
        "urllib.request.urlopen", side_effect=OSError("connection refused")
    ), pytest.raises(RuntimeError, match="GitHub dispatch failed after retry"):
        _dispatch_training("tid-000", base_cfg)


def test_dispatch_raises_on_unexpected_status(base_cfg):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 500

    with patch("urllib.request.urlopen", return_value=mock_resp), pytest.raises(
        RuntimeError, match="unexpected status 500"
    ):
        _dispatch_training("tid-999", base_cfg)
