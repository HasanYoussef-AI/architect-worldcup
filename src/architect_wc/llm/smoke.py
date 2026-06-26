"""Offline plumbing smoke test for the LLM phase.

Proves the plumbing before any live round, with no API call, so it lives in the
normal pytest suite and runs in CI. It proves three things, exactly the contract
the phase requires:

  1. The quarantine gate catches a known planted target-round result, while a
     clean dossier of pre-cutoff group-stage facts passes untouched.
  2. The committed dossier validates against its schema and its coverage manifest
     is complete; the per-match predictions have their own gate tests.
  3. The scoring harness reproduces RPS against a hand-checked value, reusing the
     project's single RPS implementation.

The fixtures here are canonical Python literals, not a scored forecast, so the
smoke is fully hermetic and the same data drives both the wc-llm-smoke CLI and the
pytest gates. The planted result is a target-round result, so it exercises the
round-aware gate correctly.
"""

from __future__ import annotations

from typing import Any

from architect_wc import calibrate
from architect_wc.llm import coverage, quarantine, schemas, score_llm

TARGET_ROUND = "R32"

# Two fixtures for the smoke. Nominal home is the first-listed team, a labelling
# convention only since the venue is neutral.
FIXTURES: list[dict[str, Any]] = [
    {"match": 73, "home_team": "Spain", "away_team": "Portugal"},
    {"match": 74, "home_team": "Argentina", "away_team": "Norway"},
]

# A clean dossier: pre-cutoff group-stage facts that must be preserved. It names
# fixture opponents without results (tactical), carries a group-stage scoreline
# (recent form), reaches the target round (allowed), and holds a historical
# head-to-head between the two fixture teams (allowed, the head-to-head factor).
CLEAN_DOSSIER: dict[str, Any] = {
    "round": TARGET_ROUND,
    "match": 73,
    "home_team": "Spain",
    "away_team": "Portugal",
    "as_of_date": "2026-06-26",
    "committed_at": None,
    "git_sha": "smoke",
    "model_version": "0.0.0",
    "factors": {
        "squad_availability": [
            {
                "claim": (
                    "Spain midfielder Rodri returned to full training this "
                    "week after a minor knock."
                ),
                "source": "marca.com",
                "source_tier": 2,
                "team": "Spain",
            },
            {
                "claim": (
                    "Portugal forward Goncalo Ramos trained fully and is available."
                ),
                "source": "espn.com",
                "source_tier": 2,
                "team": "Portugal",
            },
        ],
        "recent_form": [
            {
                "claim": (
                    "Spain won all three Group H matches, including a 2-0 win "
                    "over Uruguay in the group stage."
                ),
                "source": "uefa.com",
                "source_tier": 1,
                "team": "Spain",
            },
            {
                "claim": "Portugal took four points from their first two group games.",
                "source": "uefa.com",
                "source_tier": 1,
                "team": "Portugal",
            },
        ],
        "tactical_matchup": [
            {
                "claim": (
                    "Spain dominate possession while Portugal are most "
                    "dangerous in transition."
                ),
                "source": "espn.com",
                "source_tier": 2,
                "team": None,
            }
        ],
        "coaching_staff": [
            {
                "claim": (
                    "Spain's head coach has kept a settled backroom staff through 2026."
                ),
                "source": "as.com",
                "source_tier": 2,
                "team": "Spain",
            }
        ],
        # Searched and nothing came back: a clean neutral, insufficient evidence
        # may be marked here.
        "strategic_incentives": [],
        # Raw hits came back but all were filtered: NOT a clean neutral, so
        # insufficient evidence may not be marked here.
        "psychological_momentum": [],
        "historical_head_to_head": [
            {
                "claim": (
                    "Spain and Portugal last met in 2022, a 1-1 draw in the "
                    "Nations League."
                ),
                "source": "uefa.com",
                "source_tier": 1,
                "team": None,
                "meeting_date": "2022-09-27",
            }
        ],
    },
    "coverage": [
        {
            "factor": "squad_availability",
            "n_raw_hits": 3,
            "n_admissible_findings": 2,
            "researchable": True,
            "status": "has_findings",
            "note": None,
        },
        {
            "factor": "recent_form",
            "n_raw_hits": 3,
            "n_admissible_findings": 2,
            "researchable": True,
            "status": "has_findings",
            "note": None,
        },
        {
            "factor": "tactical_matchup",
            "n_raw_hits": 1,
            "n_admissible_findings": 1,
            "researchable": True,
            "status": "has_findings",
            "note": None,
        },
        {
            "factor": "coaching_staff",
            "n_raw_hits": 2,
            "n_admissible_findings": 1,
            "researchable": True,
            "status": "has_findings",
            "note": None,
        },
        {
            "factor": "strategic_incentives",
            "n_raw_hits": 0,
            "n_admissible_findings": 0,
            "researchable": True,
            "status": "queried_no_findings",
            "note": "Searched the allow-list; no incentive or rotation news surfaced.",
        },
        {
            "factor": "psychological_momentum",
            "n_raw_hits": 2,
            "n_admissible_findings": 0,
            "researchable": True,
            "status": "quarantined_or_filtered",
            "note": (
                "Two hits framed momentum via a forbidden later-round result; filtered."
            ),
        },
        {
            "factor": "historical_head_to_head",
            "n_raw_hits": 1,
            "n_admissible_findings": 1,
            "researchable": True,
            "status": "has_findings",
            "note": None,
        },
    ],
}

# The poisoned dossier is the clean one plus one planted target-round result. The
# gate must catch it loud.
_PLANTED_RESULT = {
    "claim": "Spain beat Portugal 2-0 in the round of 32 to reach the last 16.",
    "source": "leaked feed",
    "source_tier": 3,
    "team": "Spain",
}


def poisoned_dossier() -> dict[str, Any]:
    """The clean dossier with one planted target-round result added to recent form."""
    dossier = {
        "round": CLEAN_DOSSIER["round"],
        "as_of_date": CLEAN_DOSSIER["as_of_date"],
        "committed_at": None,
        "git_sha": CLEAN_DOSSIER["git_sha"],
        "model_version": CLEAN_DOSSIER["model_version"],
        "factors": {
            factor: list(entries)
            for factor, entries in CLEAN_DOSSIER["factors"].items()
        },
    }
    dossier["factors"]["recent_form"] = [
        *dossier["factors"]["recent_form"],
        dict(_PLANTED_RESULT),
    ]
    return dossier


# Hand-checked RPS values. For the ordered three-way (0.5, 0.3, 0.2) on a home
# win, cumulative predictions are (0.5, 0.8) against the observed (1, 1), so
# RPS = ((0.5-1)^2 + (0.8-1)^2) / 2 = (0.25 + 0.04) / 2 = 0.145. For the binary
# advance, p = 0.7 with the home side advancing gives (0.7-1)^2 = 0.09.
HAND_CHECKED_THREE_WAY = ((0.5, 0.3, 0.2), calibrate.HOME_WIN, 0.145)
HAND_CHECKED_ADVANCE = (0.7, True, 0.09)
RPS_TOLERANCE = 1e-9


def run_smoke() -> dict[str, Any]:
    """Run all offline plumbing checks and return a structured report.

    Does not raise: every check is recorded as passed or failed with a detail, so
    the CLI can print a full report and the pytest gates can assert on the parts
    they care about. all_passed is the single bottom-line.
    """
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    # 1. Quarantine: clean passes, planted target-round result is caught.
    try:
        quarantine.assert_no_dossier_leakage(CLEAN_DOSSIER, TARGET_ROUND, FIXTURES)
        record("quarantine_clean_passes", True, "clean dossier accepted")
    except quarantine.LeakageError as error:
        record("quarantine_clean_passes", False, f"clean dossier rejected: {error}")

    try:
        quarantine.assert_no_dossier_leakage(poisoned_dossier(), TARGET_ROUND, FIXTURES)
        record(
            "quarantine_catches_planted_result",
            False,
            "planted target-round result was NOT caught",
        )
    except quarantine.LeakageError as error:
        reason = error.violations[0].reason
        caught_round = "R32" in reason or "target round" in reason
        record(
            "quarantine_catches_planted_result",
            caught_round,
            f"caught: {reason}",
        )

    # 2. The committed dossier validates against its schema and its coverage manifest
    # is complete. The per-match predictions A, B, and C have their own gate tests.
    try:
        schemas.validate_document(CLEAN_DOSSIER, "dossier")
        record("schema_dossier_validates", True, "valid against schema")
    except Exception as error:  # noqa: BLE001
        record("schema_dossier_validates", False, str(error))

    try:
        coverage.assert_coverage_complete(CLEAN_DOSSIER)
        record(
            "coverage_manifest_complete",
            True,
            "coverage manifest complete and consistent with the factor arrays",
        )
    except coverage.CoverageError as error:
        record("coverage_manifest_complete", False, str(error))

    # 3. RPS reproduces hand-checked values, reusing the project's scorer.
    three_way, outcome, expected_tw = HAND_CHECKED_THREE_WAY
    got_tw = score_llm.score_three_way(three_way, outcome)
    record(
        "rps_three_way_hand_checked",
        abs(got_tw - expected_tw) <= RPS_TOLERANCE,
        f"score_three_way{three_way} = {got_tw:.4f}, expected {expected_tw}",
    )

    p_adv, advanced, expected_adv = HAND_CHECKED_ADVANCE
    got_adv = score_llm.score_advance(p_adv, advanced)
    record(
        "rps_advance_hand_checked",
        abs(got_adv - expected_adv) <= RPS_TOLERANCE,
        f"score_advance({p_adv}, advanced={advanced}) = {got_adv:.4f}, "
        f"expected {expected_adv}",
    )

    all_passed = all(check["passed"] for check in checks)
    return {"checks": checks, "all_passed": all_passed}
