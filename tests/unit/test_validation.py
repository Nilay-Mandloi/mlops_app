"""Schema validation behavior."""

from __future__ import annotations

import pytest

from price_forecast.validation import SchemaValidationError, validate_features


def _contract() -> dict:
    return {
        "schema_version": "1.0",
        "feature_columns": ["a", "b", "c"],
        "nullable_columns": ["c"],
    }


def test_validate_ok():
    validate_features({"a": 1, "b": 2.0, "c": "x"}, _contract())


def test_validate_missing_column():
    with pytest.raises(SchemaValidationError) as exc:
        validate_features({"a": 1, "b": 2.0}, _contract())
    assert "b" not in exc.value.missing
    assert exc.value.missing == ["c"]


def test_validate_unexpected_strict():
    with pytest.raises(SchemaValidationError) as exc:
        validate_features({"a": 1, "b": 2.0, "c": "x", "z": 99}, _contract())
    assert exc.value.unexpected == ["z"]


def test_validate_unexpected_lax():
    """Non-strict mode tolerates extra columns."""
    validate_features({"a": 1, "b": 2.0, "c": "x", "z": 99}, _contract(), strict=False)


def test_validate_null_in_required():
    with pytest.raises(SchemaValidationError) as exc:
        validate_features({"a": None, "b": 2.0, "c": "x"}, _contract())
    assert "a" in exc.value.null_required


def test_validate_null_in_nullable_is_fine():
    validate_features({"a": 1, "b": 2.0, "c": None}, _contract())


def test_validate_catches_python_nan():
    with pytest.raises(SchemaValidationError) as exc:
        validate_features({"a": float("nan"), "b": 2.0, "c": "x"}, _contract())
    assert "a" in exc.value.null_required


def test_validate_catches_numpy_nan_scalars():
    """Regression: NaN values from numpy scalars must also be rejected.
    Previously the isinstance(v, float) gate let numpy.float32(NaN) through."""
    np = pytest.importorskip("numpy")
    for nan_val in (np.float64("nan"), np.float32("nan"), np.nan):
        with pytest.raises(SchemaValidationError) as exc:
            validate_features({"a": nan_val, "b": 2.0, "c": "x"}, _contract())
        assert "a" in exc.value.null_required


def test_validate_non_numeric_values_not_treated_as_null():
    """Strings, bools, lists must not be misclassified as null by the
    try/except around math.isnan."""
    validate_features({"a": "hello", "b": True, "c": [1, 2]}, _contract())


def test_validate_skips_when_contract_empty():
    """Legacy artifacts with no contract should not block predictions."""
    validate_features({"a": 1}, {})
    validate_features({"a": 1}, {"feature_columns": []})


def test_to_dict_includes_buckets():
    exc = SchemaValidationError("test", missing=["a"], unexpected=["z"], null_required=["b"])
    out = exc.to_dict()
    assert out["error"] == "test"
    assert out["missing"] == ["a"]
    assert out["unexpected"] == ["z"]
    assert out["null_required"] == ["b"]
