from __future__ import annotations

import pytest

from price_forecast.config import load_config


def test_load_config_rejects_invalid_channel(monkeypatch):
    monkeypatch.setenv("APP_ID", "app1")
    monkeypatch.setenv("APP_S3_BUCKET", "bucket")
    monkeypatch.setenv("APP_MODEL_NAME", "price_forecast")
    monkeypatch.setenv("APP_CHANNEL", "beta")
    monkeypatch.setenv("ENV", "test")

    with pytest.raises(OSError, match="APP_CHANNEL must be one of"):
        load_config()


def test_load_config_rejects_non_stable_channel_in_prod(monkeypatch):
    monkeypatch.setenv("APP_ID", "app1")
    monkeypatch.setenv("APP_S3_BUCKET", "bucket")
    monkeypatch.setenv("APP_MODEL_NAME", "price_forecast")
    monkeypatch.setenv("APP_CHANNEL", "latest")
    monkeypatch.setenv("ENV", "prod")
    monkeypatch.setenv("APP_ADMIN_TOKEN", "token")
    monkeypatch.setenv("APP_CORS_ALLOWED_ORIGINS", "https://example.com")

    with pytest.raises(OSError, match="APP_CHANNEL must be 'stable' in prod"):
        load_config()
