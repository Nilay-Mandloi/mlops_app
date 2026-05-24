"""Layout sanity tests — confirm key shapes match the training repo's contract."""

from __future__ import annotations

import pytest

from price_forecast import layout

P = "product_dq"
M = "sentiment_analysis"


def test_model_pkl_key_shape():
    assert layout.model_pkl_key(P, M, 42) == "product_dq/sentiment_analysis/v42/model.pkl"
    assert layout.model_pkl_key(P, M, "7") == "product_dq/sentiment_analysis/v7/model.pkl"
    assert layout.model_pkl_key(P, M, "v9") == "product_dq/sentiment_analysis/v9/model.pkl"


def test_pointer_key_shape():
    assert layout.pointer_key(P, M, "stable") == "product_dq/sentiment_analysis/stable.json"


def test_trigger_dataset_key_default_parquet():
    assert (
        layout.trigger_dataset_key(P, "2026-05-14T10-22Z_abc12345")
        == "_triggers/product_dq/2026-05-14T10-22Z_abc12345/dataset.parquet"
    )


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("parquet", "_triggers/product_dq/t/dataset.parquet"),
        ("csv", "_triggers/product_dq/t/dataset.csv"),
    ],
)
def test_trigger_dataset_key_format_picks_extension(fmt, expected):
    assert layout.trigger_dataset_key(P, "t", fmt) == expected


def test_trigger_dataset_key_rejects_unsupported_format():
    with pytest.raises(ValueError, match="dataset_format"):
        layout.trigger_dataset_key(P, "t", "json")


def test_trigger_running_key_shape():
    assert (
        layout.trigger_running_key(P, "20260519T120000Z_abcdef12")
        == "_triggers/product_dq/20260519T120000Z_abcdef12/running.json"
    )


def test_trigger_failure_key_shape():
    assert (
        layout.trigger_failure_key(P, "20260519T120000Z_abcdef12")
        == "_triggers/product_dq/20260519T120000Z_abcdef12/failed.json"
    )


def test_trigger_metadata_key_shape():
    assert layout.trigger_metadata_key(P, "tid") == "_triggers/product_dq/tid/trigger.json"


def test_bucket_for():
    assert layout.bucket_for("mlops") == "mlops-artifacts"


@pytest.mark.parametrize("bad", ["", " ", " app", "a/b", "..", ".x"])
def test_unsafe_project_rejected(bad):
    with pytest.raises(ValueError):
        layout.model_pkl_key(bad, M, 1)
