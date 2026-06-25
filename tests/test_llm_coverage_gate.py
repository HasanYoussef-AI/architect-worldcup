"""Verification gate: the research coverage manifest (Refinement A).

Hermetic, no network. Proves the status derivation, the prediction-time
insufficient-evidence link, and that assert_coverage_complete accepts a consistent
manifest and fails loud on every inconsistency it is meant to catch: a missing
factor, an admissible count that disagrees with the factor array, a raw count
below the admissible count, and a reported status that the counts contradict. The
quarantined-and-filtered cell must not pass as a clean neutral.
"""

from __future__ import annotations

import pytest

from architect_wc.llm import coverage
from architect_wc.llm.weights import FACTORS


def _cell(factor, n_raw, n_admissible, researchable, status, note=None):
    return {
        "factor": factor,
        "n_raw_hits": n_raw,
        "n_admissible_findings": n_admissible,
        "researchable": researchable,
        "status": status,
        "note": note,
    }


def _dossier(findings_per_factor, coverage_cells):
    return {
        "factors": {f: findings_per_factor.get(f, []) for f in FACTORS},
        "coverage": coverage_cells,
    }


def _finding():
    return {"claim": "x", "source": "uefa.com", "source_tier": 1}


def test_derive_status_covers_every_case() -> None:
    assert coverage.derive_status(0, 0, False) == coverage.STATUS_UNRESEARCHABLE
    assert coverage.derive_status(5, 0, False) == coverage.STATUS_UNRESEARCHABLE
    assert coverage.derive_status(3, 2, True) == coverage.STATUS_HAS_FINDINGS
    assert coverage.derive_status(2, 0, True) == coverage.STATUS_QUARANTINED_OR_FILTERED
    assert coverage.derive_status(0, 0, True) == coverage.STATUS_QUERIED_NO_FINDINGS


def test_insufficient_evidence_link() -> None:
    # Only queried_no_findings and unresearchable permit an insufficient-evidence
    # neutral; a quarantined-and-filtered cell does not.
    assert coverage.insufficient_evidence_allowed(coverage.STATUS_QUERIED_NO_FINDINGS)
    assert coverage.insufficient_evidence_allowed(coverage.STATUS_UNRESEARCHABLE)
    assert not coverage.insufficient_evidence_allowed(
        coverage.STATUS_QUARANTINED_OR_FILTERED
    )
    assert not coverage.insufficient_evidence_allowed(coverage.STATUS_HAS_FINDINGS)


def _complete_manifest():
    findings = {f: [_finding()] for f in FACTORS}
    cells = [_cell(f, 1, 1, True, "has_findings") for f in FACTORS]
    return _dossier(findings, cells)


def test_complete_consistent_manifest_passes() -> None:
    coverage.assert_coverage_complete(_complete_manifest())


def test_quarantined_cell_is_not_a_clean_neutral() -> None:
    # The Refinement A case: raw hits came back, none admissible. The status must
    # be quarantined_or_filtered, and the manifest is valid in that state.
    findings = {f: [_finding()] for f in FACTORS}
    findings["psychological_momentum"] = []
    cells = [
        _cell(f, 1, 1, True, "has_findings")
        for f in FACTORS
        if f != "psychological_momentum"
    ]
    cells.append(_cell("psychological_momentum", 2, 0, True, "quarantined_or_filtered"))
    coverage.assert_coverage_complete(_dossier(findings, cells))
    # Labelling that same cell queried_no_findings (a clean neutral) is rejected.
    bad = [c for c in cells if c["factor"] != "psychological_momentum"]
    bad.append(_cell("psychological_momentum", 2, 0, True, "queried_no_findings"))
    with pytest.raises(coverage.CoverageError):
        coverage.assert_coverage_complete(_dossier(findings, bad))


def test_missing_factor_cell_fails() -> None:
    manifest = _complete_manifest()
    manifest["coverage"] = manifest["coverage"][:-1]
    with pytest.raises(coverage.CoverageError):
        coverage.assert_coverage_complete(manifest)


def test_admissible_count_must_match_factor_array() -> None:
    findings = {f: [_finding()] for f in FACTORS}
    # recent_form array has one entry, but the manifest claims two admissible.
    cells = [
        _cell(f, 2, 2, True, "has_findings")
        if f == "recent_form"
        else _cell(f, 1, 1, True, "has_findings")
        for f in FACTORS
    ]
    with pytest.raises(coverage.CoverageError):
        coverage.assert_coverage_complete(_dossier(findings, cells))


def test_raw_below_admissible_fails() -> None:
    findings = {f: [_finding()] for f in FACTORS}
    cells = [
        _cell(f, 0, 1, True, "has_findings")
        if f == "recent_form"
        else _cell(f, 1, 1, True, "has_findings")
        for f in FACTORS
    ]
    with pytest.raises(coverage.CoverageError):
        coverage.assert_coverage_complete(_dossier(findings, cells))


def test_unresearchable_factors_are_surfaced() -> None:
    findings = {f: [_finding()] for f in FACTORS}
    findings["coaching_staff"] = []
    cells = [
        _cell(f, 1, 1, True, "has_findings") for f in FACTORS if f != "coaching_staff"
    ]
    cells.append(
        _cell(
            "coaching_staff", 0, 0, False, "unresearchable", "no allow-listed coverage"
        )
    )
    dossier = _dossier(findings, cells)
    coverage.assert_coverage_complete(dossier)
    assert coverage.unresearchable_factors(dossier) == ["coaching_staff"]
