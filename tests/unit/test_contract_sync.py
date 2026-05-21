"""Verify vendored contracts.py and layout.py are in sync with the training repo.

These files are copy-and-pinned from mlops/src/quantity_forecast/ into this
repo. If they diverge the serving layer will mis-interpret payloads the
training repo produces. This test fails whenever drift is detected so the
developer is forced to sync both copies before merging.

The test auto-discovers the training repo as a sibling directory (../mlops)
or via the MLOPS_ROOT env var. Skipped when neither is present — typical on
serving-repo-only CI that checks out mlops_app alone.

To run locally (repos checked out as siblings):
    pytest tests/unit/test_contract_sync.py
or:
    MLOPS_ROOT=/path/to/mlops pytest tests/unit/test_contract_sync.py

To sync manually:
    cp ../mlops/src/quantity_forecast/contracts.py src/price_forecast/contracts.py
    cp ../mlops/src/quantity_forecast/layout.py     src/price_forecast/layout.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[2]

_MLOPS_ROOT: Path | None = None
if os.environ.get("MLOPS_ROOT"):
    _MLOPS_ROOT = Path(os.environ["MLOPS_ROOT"])
else:
    candidate = _REPO_ROOT.parent / "mlops"
    if candidate.is_dir():
        _MLOPS_ROOT = candidate

_SKIP = pytest.mark.skipif(
    _MLOPS_ROOT is None,
    reason="Training repo (mlops) not found. Set MLOPS_ROOT or check out repos as siblings.",
)

# Strip both styles of vendor header:
#   # Vendored copy ... (comment block followed by blank line)
_VENDOR_COMMENT_RE = re.compile(r"^(?:#[^\n]*\n)+\n")


def _strip_vendor_header(text: str) -> str:
    return _VENDOR_COMMENT_RE.sub("", text, count=1)


def _normalise(text: str) -> str:
    return _strip_vendor_header(text).strip()


@_SKIP
def test_contracts_py_in_sync():
    training = (_MLOPS_ROOT / "src/quantity_forecast/contracts.py").read_text(encoding="utf-8")
    serving = (_REPO_ROOT / "src/price_forecast/contracts.py").read_text(encoding="utf-8")
    assert _normalise(training) == _normalise(serving), (
        "contracts.py has drifted from the training repo.\n"
        "Sync with:\n"
        "  cp ../mlops/src/quantity_forecast/contracts.py src/price_forecast/contracts.py"
    )


@_SKIP
def test_layout_py_in_sync():
    training = (_MLOPS_ROOT / "src/quantity_forecast/layout.py").read_text(encoding="utf-8")
    serving = (_REPO_ROOT / "src/price_forecast/layout.py").read_text(encoding="utf-8")
    assert _normalise(training) == _normalise(serving), (
        "layout.py has drifted from the training repo.\n"
        "Sync with:\n"
        "  cp ../mlops/src/quantity_forecast/layout.py src/price_forecast/layout.py"
    )
