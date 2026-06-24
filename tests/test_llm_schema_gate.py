"""Verification gate: the dossier and Prediction A, B, C schemas.

Hermetic, no network. Validates the canonical fixtures against their published
schemas, confirms each prediction's three-way is a valid distribution, confirms
every nonzero factor in Prediction B cites evidence, and proves a malformed
document is rejected so the schema is actually constraining.
"""

from __future__ import annotations

import jsonschema
import pytest

from architect_wc.llm import schemas, score_llm, smoke, validity


def test_prediction_a_validates() -> None:
    schemas.validate_document(smoke.FIXTURE_A, "A")


def test_prediction_b_validates() -> None:
    schemas.validate_document(smoke.FIXTURE_B, "B")


def test_prediction_c_validates() -> None:
    schemas.validate_document(smoke.FIXTURE_C, "C")


def test_dossier_validates() -> None:
    schemas.validate_document(smoke.CLEAN_DOSSIER, "dossier")


def test_three_way_distributions_are_valid() -> None:
    for kind, document in (
        ("A", smoke.FIXTURE_A),
        ("B", smoke.FIXTURE_B),
        ("C", smoke.FIXTURE_C),
    ):
        keys = score_llm.keys_for(kind)
        for tie in document["ties"]:
            validity.assert_prediction_valid(tie, keys)


def test_nonzero_factors_are_cited() -> None:
    for tie in smoke.FIXTURE_B["ties"]:
        assert validity.nonzero_factors_are_cited(tie["factor_scores"])


def test_uncited_nonzero_factor_is_rejected() -> None:
    bad = [{"factor": "recent_form", "score": 2, "evidence": []}]
    assert not validity.nonzero_factors_are_cited(bad)


def test_malformed_prediction_is_rejected() -> None:
    broken = {"prediction": "A", "round": "R32"}  # missing required fields
    with pytest.raises(jsonschema.ValidationError):
        schemas.validate_document(broken, "A")
