"""Request-side feature validation against the manifest's schema_contract.

The training repo's contracts.ArtifactManifest carries a ``schema_contract``
dict that lists the feature_columns the model was trained on (plus optional
dtype hints). When a prediction request comes in, we cross-check the request
features against this contract BEFORE handing them to model.predict().

This catches three classes of bugs at the API boundary instead of inside
sklearn (where they become cryptic ValueError stack traces):

  - Missing feature column -> 400 with explicit "missing: [...]" payload.
  - Extra unknown column   -> 400 with explicit "unexpected: [...]" payload.
  - None / NaN feature when the contract says required -> 400.

Type coercion is intentionally NOT done here. If the contract specifies
dtypes the loader's preprocessor handles them; this module just validates
the shape of the incoming dict.
"""

from __future__ import annotations

import math
from typing import Any


class SchemaValidationError(ValueError):
    """Raised when a request payload does not match the model schema."""

    def __init__(
        self,
        message: str,
        *,
        missing: list[str] | None = None,
        unexpected: list[str] | None = None,
        null_required: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.missing = missing or []
        self.unexpected = unexpected or []
        self.null_required = null_required or []

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"error": str(self)}
        if self.missing:
            out["missing"] = self.missing
        if self.unexpected:
            out["unexpected"] = self.unexpected
        if self.null_required:
            out["null_required"] = self.null_required
        return out


def _contract_feature_columns(schema_contract: dict[str, Any]) -> list[str] | None:
    """Pull the canonical feature list out of the manifest's schema_contract.

    Returns None when the contract is empty (legacy artifacts) — caller
    should skip validation rather than reject every request.
    """
    if not schema_contract:
        return None
    cols = schema_contract.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        return None
    return [str(c) for c in cols]


def _nullable_columns(schema_contract: dict[str, Any]) -> set[str]:
    raw = schema_contract.get("nullable_columns") or []
    return {str(c) for c in raw if isinstance(c, str)}


def validate_features(
    features: dict[str, Any],
    schema_contract: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """Raise SchemaValidationError on shape mismatch.

    Args:
        features: request payload (single row).
        schema_contract: manifest.schema_contract dict.
        strict: if False, only check 'missing'; ignore extras.
    """
    expected = _contract_feature_columns(schema_contract)
    if expected is None:
        return  # legacy artifact w/o contract — skip
    expected_set = set(expected)
    got_set = set(features.keys())

    missing = sorted(expected_set - got_set)
    unexpected = sorted(got_set - expected_set) if strict else []

    nullable = _nullable_columns(schema_contract)

    def _is_null(v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, (str, bool, list, dict, tuple, bytes)):
            return False
        try:
            return bool(math.isnan(v))
        except (TypeError, ValueError):
            return False

    null_required = sorted(
        col for col in expected_set & got_set if _is_null(features[col]) and col not in nullable
    )

    if missing or unexpected or null_required:
        bits = []
        if missing:
            bits.append(f"missing={missing}")
        if unexpected:
            bits.append(f"unexpected={unexpected}")
        if null_required:
            bits.append(f"null_required={null_required}")
        raise SchemaValidationError(
            "feature payload does not match schema_contract: " + ", ".join(bits),
            missing=missing,
            unexpected=unexpected,
            null_required=null_required,
        )
