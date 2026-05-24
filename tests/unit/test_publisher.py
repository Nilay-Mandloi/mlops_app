from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from price_forecast.adapters.github_dispatch import GitHubDispatchAdapter, NoopDispatchAdapter


@pytest.fixture()
def adapter() -> GitHubDispatchAdapter:
    return GitHubDispatchAdapter(
        training_repo="my-org/mlops",
        training_repo_token="ghp_testtoken",
        retry_pause_s=0,
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


def test_adapter_rejects_missing_repo():
    with pytest.raises(ValueError, match="training_repo and training_repo_token"):
        GitHubDispatchAdapter(training_repo="", training_repo_token="x")


def test_adapter_rejects_missing_token():
    with pytest.raises(ValueError, match="training_repo and training_repo_token"):
        GitHubDispatchAdapter(training_repo="org/repo", training_repo_token="")


# ---------------------------------------------------------------------------
# Noop adapter — dev/test default
# ---------------------------------------------------------------------------


def test_noop_dispatch_is_a_silent_logger():
    NoopDispatchAdapter().dispatch_training(
        trigger_id="tid-noop",
        category="mlops",
        project="product_dq",
        model_name="price_forecast",
        bucket="test-bucket",
        prefix="",
        auto_promote=False,
    )


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_dispatch_sends_correct_payload(adapter):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 204

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        adapter.dispatch_training(
            trigger_id="tid-123",
            category="mlops",
            project="product_dq",
            model_name="price_forecast",
            bucket="test-bucket",
            prefix="",
            auto_promote=False,
        )

    mock_open.assert_called_once()
    req = mock_open.call_args[0][0]

    assert req.full_url == "https://api.github.com/repos/my-org/mlops/dispatches"
    assert req.get_header("Authorization") == "Bearer ghp_testtoken"
    assert req.get_header("Accept") == "application/vnd.github+json"

    body = json.loads(req.data.decode("utf-8"))
    assert body["event_type"] == "train-model"
    payload = body["client_payload"]
    assert payload["trigger_id"] == "tid-123"
    assert payload["auto_promote"] is False
    assert payload["category"] == "mlops"
    assert payload["project"] == "product_dq"
    assert payload["model_name"] == "price_forecast"
    assert payload["artifact_store_bucket"] == "test-bucket"
    assert payload["artifact_store_prefix"] == ""


def test_dispatch_auto_promote_flag_forwarded(adapter):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 204

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_open:
        adapter.dispatch_training(
            trigger_id="tid-456",
            category="mlops",
            project="product_dq",
            model_name="price_forecast",
            bucket="test-bucket",
            prefix="",
            auto_promote=True,
        )
    body = json.loads(mock_open.call_args[0][0].data.decode("utf-8"))
    assert body["client_payload"]["auto_promote"] is True


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_dispatch_raises_on_http_error(adapter):
    http_err = urllib.error.HTTPError(
        url="https://api.github.com/repos/my-org/mlops/dispatches",
        code=401,
        msg="Unauthorized",
        hdrs=MagicMock(),  # type: ignore[arg-type]
        fp=BytesIO(b""),
    )
    with (
        patch("urllib.request.urlopen", side_effect=http_err),
        pytest.raises(RuntimeError, match="GitHub dispatch failed: HTTP 401"),
    ):
        adapter.dispatch_training(
            trigger_id="tid-789",
            category="mlops",
            project="product_dq",
            model_name="price_forecast",
            bucket="test-bucket",
            prefix="",
            auto_promote=False,
        )


def test_dispatch_raises_on_network_error(adapter):
    with (
        patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
        pytest.raises(RuntimeError, match="GitHub dispatch failed after retry"),
    ):
        adapter.dispatch_training(
            trigger_id="tid-000",
            category="mlops",
            project="product_dq",
            model_name="price_forecast",
            bucket="test-bucket",
            prefix="",
            auto_promote=False,
        )


def test_dispatch_raises_on_unexpected_status(adapter):
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 500

    with (
        patch("urllib.request.urlopen", return_value=mock_resp),
        pytest.raises(RuntimeError, match="unexpected status 500"),
    ):
        adapter.dispatch_training(
            trigger_id="tid-999",
            category="mlops",
            project="product_dq",
            model_name="price_forecast",
            bucket="test-bucket",
            prefix="",
            auto_promote=False,
        )
