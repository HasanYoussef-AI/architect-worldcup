"""The research coverage manifest and its completeness gate (Refinement A).

A per-match dossier records, for each of the seven factor cells, how many raw
search hits came back (n_raw_hits) and how many survived the quarantine gate and
the tier filter to become admissible findings (n_admissible_findings). Recording
the two separately is the point: a cell that returned content which was then
quarantined or tier-filtered shows n_raw_hits greater than zero with
n_admissible_findings zero, and must never be silently treated as a clean neutral.

The status of each cell is derived purely from those two counts and a researchable
flag, so the gate recomputes it rather than trusting the writer:

  - unresearchable          researchable is false: no admissible source covered the
                            factor for this fixture. A stated blind spot.
  - has_findings            at least one admissible finding.
  - quarantined_or_filtered raw hits came back but none survived. NOT a clean
                            neutral.
  - queried_no_findings     the cell was searched and nothing came back.

Prediction-time link: a prediction may mark a factor insufficient_evidence only
when its cell is queried_no_findings or unresearchable. A quarantined_or_filtered
cell is not a free neutral, and a has_findings cell has evidence to weigh.
"""

from __future__ import annotations

from typing import Any

from architect_wc.llm.weights import FACTORS

STATUS_HAS_FINDINGS = "has_findings"
STATUS_QUERIED_NO_FINDINGS = "queried_no_findings"
STATUS_QUARANTINED_OR_FILTERED = "quarantined_or_filtered"
STATUS_UNRESEARCHABLE = "unresearchable"

# The only two cell statuses under which a prediction may mark insufficient
# evidence. The link the strategy chat asked to keep.
INSUFFICIENT_EVIDENCE_STATUSES = frozenset(
    {STATUS_QUERIED_NO_FINDINGS, STATUS_UNRESEARCHABLE}
)


class CoverageError(Exception):
    """Raised, fail loud, when the coverage manifest is incomplete or inconsistent."""


def derive_status(n_raw_hits: int, n_admissible: int, researchable: bool) -> str:
    """Derive a cell's status purely from its counts and researchable flag.

    The single source of truth for what a cell means, recomputed by the gate so a
    writer cannot label a quarantined cell as a clean neutral.
    """
    if not researchable:
        return STATUS_UNRESEARCHABLE
    if n_admissible > 0:
        return STATUS_HAS_FINDINGS
    if n_raw_hits > 0:
        return STATUS_QUARANTINED_OR_FILTERED
    return STATUS_QUERIED_NO_FINDINGS


def insufficient_evidence_allowed(status: str) -> bool:
    """True if a prediction may mark insufficient_evidence for a cell in this status.

    The prediction-time link: only queried_no_findings and unresearchable qualify.
    A quarantined_or_filtered cell is not a clean neutral, so it does not.
    """
    return status in INSUFFICIENT_EVIDENCE_STATUSES


def unresearchable_factors(dossier: dict[str, Any]) -> list[str]:
    """Return the factors whose cell is unresearchable, the dossier's blind spots.

    These must be surfaced in the match rationale at prediction time so the blind
    spot is stated rather than hidden.
    """
    coverage = dossier.get("coverage", []) or []
    return [
        cell.get("factor")
        for cell in coverage
        if cell.get("status") == STATUS_UNRESEARCHABLE
    ]


def assert_coverage_complete(dossier: dict[str, Any]) -> None:
    """Raise CoverageError unless the coverage manifest is complete and consistent.

    Checks: exactly one cell per factor and no extras; each cell's counts are
    non-negative with raw a superset of admissible; n_admissible_findings equals
    the length of that factor's admissible array in factors; and the reported
    status matches the status derived from the counts. The gate never repairs the
    manifest, it fails loud, because a silently-wrong manifest would let a
    researched-and-quarantined cell pass as a clean neutral.
    """
    coverage = dossier.get("coverage")
    if not isinstance(coverage, list):
        raise CoverageError("Dossier has no coverage manifest.")

    factors = dossier.get("factors", {}) or {}
    seen: dict[str, dict[str, Any]] = {}
    for cell in coverage:
        factor = cell.get("factor")
        if factor not in FACTORS:
            raise CoverageError(f"Coverage cell for unknown factor {factor!r}.")
        if factor in seen:
            raise CoverageError(f"Duplicate coverage cell for factor {factor!r}.")
        seen[factor] = cell

    missing = [factor for factor in FACTORS if factor not in seen]
    if missing:
        raise CoverageError(f"Coverage manifest is missing factors: {missing}.")

    for factor in FACTORS:
        cell = seen[factor]
        n_raw = int(cell["n_raw_hits"])
        n_admissible = int(cell["n_admissible_findings"])
        researchable = bool(cell["researchable"])
        status = cell["status"]

        if n_raw < 0 or n_admissible < 0:
            raise CoverageError(f"Negative coverage count for {factor!r}.")
        if n_admissible > n_raw:
            raise CoverageError(
                f"Coverage for {factor!r} claims {n_admissible} admissible findings "
                f"from only {n_raw} raw hits; raw hits must be a superset."
            )

        actual_admissible = len(factors.get(factor, []) or [])
        if n_admissible != actual_admissible:
            raise CoverageError(
                f"Coverage for {factor!r} reports {n_admissible} admissible findings "
                f"but the factor array holds {actual_admissible}."
            )

        expected = derive_status(n_raw, n_admissible, researchable)
        if status != expected:
            raise CoverageError(
                f"Coverage for {factor!r} reports {status!r} but counts raw="
                f"{n_raw}, admissible={n_admissible}, researchable={researchable} "
                f"imply {expected!r}."
            )
