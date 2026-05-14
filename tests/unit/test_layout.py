"""Layout sanity tests — confirm key shapes match the training repo's contract.

If these tests fail after updating layout.py, the contract has changed and
the training repo's layout.py must be updated to match. Bump SCHEMA_VERSION
in contracts.py if the change is breaking.
"""

from __future__ import annotations

from price_forecast import layout


def test_artifact_model_pkl_key_shape():
    assert layout.artifact_model_pkl_key("42") == "output/artifacts/v42/champion/model.pkl"
    assert layout.artifact_model_pkl_key(7) == "output/artifacts/v7/champion/model.pkl"
    assert layout.artifact_model_pkl_key("v9") == "output/artifacts/v9/champion/model.pkl"


def test_pointer_key_shape():
    assert (
        layout.pointer_key("price_forecast", "stable")
        == "output/registry/price_forecast/pointers/stable.json"
    )


def test_trigger_dataset_key_shape():
    assert (
        layout.trigger_dataset_key("2026-05-14T10-22Z_abc12345")
        == "triggers/2026-05-14T10-22Z_abc12345/dataset.parquet"
    )
