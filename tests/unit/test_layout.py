"""Layout sanity tests — confirm key shapes match the training repo's contract.

If these tests fail after updating layout.py, the contract has changed and
the training repo's layout.py must be updated to match. Bump SCHEMA_VERSION
in contracts.py if the change is breaking.
"""

from __future__ import annotations

import pytest

from price_forecast import layout


APP = "APP1"


def test_artifact_model_pkl_key_shape():
    assert layout.artifact_model_pkl_key(APP, "42") == "output/artifacts/APP1/v42/champion/model.pkl"
    assert layout.artifact_model_pkl_key(APP, 7) == "output/artifacts/APP1/v7/champion/model.pkl"
    assert layout.artifact_model_pkl_key(APP, "v9") == "output/artifacts/APP1/v9/champion/model.pkl"


def test_pointer_key_shape():
    assert (
        layout.pointer_key(APP, "price_forecast", "stable")
        == "output/registry/APP1/price_forecast/pointers/stable.json"
    )


def test_trigger_dataset_key_shape():
    assert (
        layout.trigger_dataset_key(APP, "2026-05-14T10-22Z_abc12345")
        == "triggers/APP1/2026-05-14T10-22Z_abc12345/dataset.parquet"
    )


@pytest.mark.parametrize("bad", ["", " ", " app", "a/b", "..", ".x"])
def test_unsafe_app_id_rejected(bad):
    with pytest.raises(ValueError):
        layout.artifact_model_pkl_key(bad, "v1")
