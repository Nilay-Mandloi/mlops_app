from __future__ import annotations

import pytest

from price_forecast.config import load_config


def _set_required(monkeypatch):
    monkeypatch.setenv("CATEGORY", "mlops")
    monkeypatch.setenv("PROJECT", "product_dq")
    monkeypatch.setenv("MODEL_NAME", "price_forecast")


def test_load_config_rejects_invalid_channel(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("APP_CHANNEL", "beta")
    monkeypatch.setenv("ENV", "test")
    with pytest.raises(OSError, match="APP_CHANNEL must be one of"):
        load_config()


def test_load_config_rejects_non_stable_channel_in_prod(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("APP_CHANNEL", "latest")
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("APP_ADMIN_TOKEN", "token")
    monkeypatch.setenv("APP_CORS_ALLOWED_ORIGINS", "https://example.com")
    monkeypatch.setenv("TRAINING_REPO", "org/repo")
    monkeypatch.setenv("TRAINING_REPO_TOKEN", "tok")
    monkeypatch.setenv("TRIGGER_DATA_ROOT", "/tmp/x")
    with pytest.raises(OSError, match="APP_CHANNEL must be 'stable' in prod"):
        load_config()


def test_load_config_defaults_bucket_to_category(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("ENV", "test")
    monkeypatch.delenv("ARTIFACT_STORE_BUCKET", raising=False)
    cfg = load_config()
    assert cfg.bucket == "mlops-artifacts"
    assert cfg.project == "product_dq"
    assert cfg.model_name == "price_forecast"
