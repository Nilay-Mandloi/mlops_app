"""Layout sanity tests — confirm key shapes match the training repo's contract.

If these tests fail after updating layout.py, the contract has changed and
the training repo's layout.py must be updated to match. Bump SCHEMA_VERSION
in contracts.py if the change is breaking.
"""

from __future__ import annotations

import pytest

from price_forecast import layout

APP = "app1"


def test_artifact_model_pkl_key_shape():
    assert (
        layout.artifact_model_pkl_key(APP, "42") == "output/artifacts/app1/v42/champion/model.pkl"
    )
    assert layout.artifact_model_pkl_key(APP, 7) == "output/artifacts/app1/v7/champion/model.pkl"
    assert layout.artifact_model_pkl_key(APP, "v9") == "output/artifacts/app1/v9/champion/model.pkl"


def test_pointer_key_shape():
    assert (
        layout.pointer_key(APP, "price_forecast", "stable")
        == "output/registry/app1/price_forecast/pointers/stable.json"
    )


def test_trigger_dataset_key_default_parquet():
    assert (
        layout.trigger_dataset_key(APP, "2026-05-14T10-22Z_abc12345")
        == "triggers/app1/2026-05-14T10-22Z_abc12345/dataset.parquet"
    )


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("parquet", "triggers/app1/t/dataset.parquet"),
        ("csv", "triggers/app1/t/dataset.csv"),
    ],
)
def test_trigger_dataset_key_format_picks_extension(fmt, expected):
    assert layout.trigger_dataset_key(APP, "t", fmt) == expected


def test_trigger_dataset_key_rejects_unsupported_format():
    with pytest.raises(ValueError, match="dataset_format"):
        layout.trigger_dataset_key(APP, "t", "json")


def test_trigger_running_key_shape():
    assert (
        layout.trigger_running_key(APP, "20260519T120000Z_abcdef12")
        == "triggers/app1/20260519T120000Z_abcdef12/running.json"
    )


def test_trigger_failure_key_shape():
    assert (
        layout.trigger_failure_key(APP, "20260519T120000Z_abcdef12")
        == "triggers/app1/20260519T120000Z_abcdef12/failed.json"
    )


def test_trigger_metadata_key_shape():
    assert layout.trigger_metadata_key(APP, "tid") == "triggers/app1/tid/trigger.json"


@pytest.mark.parametrize("bad", ["", " ", " app", "a/b", "..", ".x"])
def test_unsafe_app_id_rejected(bad):
    with pytest.raises(ValueError):
        layout.artifact_model_pkl_key(bad, "v1")
