"""Verification gate: the dossier schema, and that a malformed document is rejected.

Hermetic, no network. The per-match predictions A, B, and C each validate in their
own gate test (test_llm_prediction_a_gate, test_prediction_b_gate,
test_prediction_c_gate); this file covers the committed dossier contract and proves
the schema is actually constraining by rejecting a malformed document.
"""

from __future__ import annotations

import jsonschema
import pytest

from architect_wc.llm import schemas, smoke


def test_dossier_validates() -> None:
    schemas.validate_document(smoke.CLEAN_DOSSIER, "dossier")


def test_malformed_prediction_is_rejected() -> None:
    broken = {"prediction": "A", "round": "R32"}  # missing required fields
    with pytest.raises(jsonschema.ValidationError):
        schemas.validate_document(broken, "prediction_a")
