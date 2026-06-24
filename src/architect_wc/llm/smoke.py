"""Offline plumbing smoke test for the LLM phase.

Proves the plumbing before any live round, with no API call, so it lives in the
normal pytest suite and runs in CI. It proves three things, exactly the contract
the phase requires:

  1. The quarantine gate catches a known planted target-round result, while a
     clean dossier of pre-cutoff group-stage facts passes untouched.
  2. Predictions A, B, and C validate against their published schemas and form
     valid probability distributions.
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
from architect_wc.llm import quarantine, schemas, score_llm, validity

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
                "source": "Marca",
                "source_tier": 1,
                "team": "Spain",
            },
            {
                "claim": (
                    "Norway striker Haaland is fit and started their third "
                    "group-stage match."
                ),
                "source": "NRK",
                "source_tier": 1,
                "team": "Norway",
            },
        ],
        "recent_form": [
            {
                "claim": (
                    "Spain won all three Group H matches, including a 2-0 win "
                    "over Uruguay in the group stage."
                ),
                "source": "UEFA match centre",
                "source_tier": 1,
                "team": "Spain",
            },
            {
                "claim": "Argentina topped Group J with three wins from three.",
                "source": "AFA",
                "source_tier": 1,
                "team": "Argentina",
            },
        ],
        "tactical_matchup": [
            {
                "claim": (
                    "Argentina favour a high press while Norway often sit "
                    "deeper and counter."
                ),
                "source": "The Athletic",
                "source_tier": 2,
                "team": None,
            }
        ],
        "coaching_staff": [
            {
                "claim": (
                    "Spain's head coach has kept a settled backroom staff through 2026."
                ),
                "source": "El Pais",
                "source_tier": 2,
                "team": "Spain",
            }
        ],
        "strategic_incentives": [
            {
                "claim": (
                    "Spain rotated heavily in their last group game, "
                    "prioritising freshness."
                ),
                "source": "El Pais",
                "source_tier": 2,
                "team": "Spain",
            }
        ],
        "psychological_momentum": [
            {
                "claim": "Norway are on a five-match unbeaten run across 2026.",
                "source": "NRK",
                "source_tier": 2,
                "team": "Norway",
            }
        ],
        "historical_head_to_head": [
            {
                "claim": (
                    "Spain and Portugal last met in 2022, a 1-1 draw in the "
                    "Nations League."
                ),
                "source": "UEFA",
                "source_tier": 1,
                "team": None,
                "meeting_date": "2022-09-27",
            }
        ],
    },
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


def _provenance_b() -> dict[str, Any]:
    return {
        "model": "claude-opus-4-8",
        "effort": "xhigh",
        "temperature": "not set; removed on Opus 4.8, determinism best effort",
        "cutoff": "2026-06-26",
        "dossier_ref": "smoke://dossier",
        "committed_at": None,
    }


def _factor(name: str, lens: str, score: int, cited: bool) -> dict[str, Any]:
    evidence = (
        [{"claim": f"{name} evidence", "source_tier": 1, "dossier_ref": "smoke"}]
        if cited
        else []
    )
    return {
        "factor": name,
        "lens": lens,
        "score": score,
        "insufficient_evidence": score == 0 and not cited,
        "evidence": evidence,
    }


# Prediction A, the math call, two ties. Internally consistent: advance equals
# p_home_win_90 + p_draw_90 * shootout_lean.
FIXTURE_A: dict[str, Any] = {
    "prediction": "A",
    "round": TARGET_ROUND,
    "as_of_date": "2026-06-26",
    "committed_at": None,
    "git_sha": "smoke",
    "model_version": "0.0.0",
    "seed": 42,
    "ties": [
        {
            "match": 73,
            "home_team": "Spain",
            "away_team": "Portugal",
            "p_home_win_90": 0.50,
            "p_draw_90": 0.28,
            "p_away_win_90": 0.22,
            "shootout_lean": 0.62,
            "p_advance_home": 0.6736,
            "source": "dixon_coles_three_way+elo_shootout",
        },
        {
            "match": 74,
            "home_team": "Argentina",
            "away_team": "Norway",
            "p_home_win_90": 0.55,
            "p_draw_90": 0.26,
            "p_away_win_90": 0.19,
            "shootout_lean": 0.66,
            "p_advance_home": 0.7216,
            "source": "dixon_coles_three_way+elo_shootout",
        },
    ],
}

FIXTURE_B: dict[str, Any] = {
    "prediction": "B",
    "round": TARGET_ROUND,
    "as_of_date": "2026-06-26",
    "committed_at": None,
    "git_sha": "smoke",
    "model_version": "0.0.0",
    "dossier_ref": "smoke://dossier",
    "ties": [
        {
            "match": 73,
            "home_team": "Spain",
            "away_team": "Portugal",
            "factor_scores": [
                _factor("squad_availability", "club scout", 1, True),
                _factor("recent_form", "match analyst", 2, True),
                _factor("tactical_matchup", "match analyst", 0, False),
                _factor("coaching_staff", "tournament strategist", 1, True),
                _factor("strategic_incentives", "tournament strategist", 0, False),
                _factor("psychological_momentum", "sports psychologist", 1, True),
                _factor("historical_head_to_head", "analyst", -1, True),
            ],
            "anchor_signal": 0.78,
            "p_home_win_90": 0.52,
            "p_draw_90": 0.27,
            "p_away_win_90": 0.21,
            "shootout_lean": 0.62,
            "p_advance_home": 0.6874,
            "anchor_departure": 0.0,
            "departure_justification": None,
            "rationale": (
                "Spain stronger across squad and form; head-to-head a mild "
                "pull the other way."
            ),
            "provenance": _provenance_b(),
        },
        {
            "match": 74,
            "home_team": "Argentina",
            "away_team": "Norway",
            "factor_scores": [
                _factor("squad_availability", "club scout", 1, True),
                _factor("recent_form", "match analyst", 1, True),
                _factor("tactical_matchup", "match analyst", 1, True),
                _factor("coaching_staff", "tournament strategist", 0, False),
                _factor("strategic_incentives", "tournament strategist", 0, False),
                _factor("psychological_momentum", "sports psychologist", 1, True),
                _factor("historical_head_to_head", "analyst", 0, False),
            ],
            "anchor_signal": 0.67,
            "p_home_win_90": 0.56,
            "p_draw_90": 0.25,
            "p_away_win_90": 0.19,
            "shootout_lean": 0.66,
            "p_advance_home": 0.725,
            "anchor_departure": 0.0,
            "departure_justification": None,
            "rationale": (
                "Argentina favoured on squad and form; Norway's momentum "
                "keeps it from a blowout."
            ),
            "provenance": _provenance_b(),
        },
    ],
}

FIXTURE_C: dict[str, Any] = {
    "prediction": "C",
    "round": TARGET_ROUND,
    "as_of_date": "2026-06-26",
    "committed_at": None,
    "git_sha": "smoke",
    "model_version": "0.0.0",
    "a_ref": "smoke://A",
    "b_ref": "smoke://B",
    "dossier_ref": "smoke://dossier",
    "ties": [
        {
            "match": 73,
            "home_team": "Spain",
            "away_team": "Portugal",
            "took_from_a": "the calibrated three-way magnitude",
            "took_from_b": "the head-to-head caution",
            "reconciled_p_home_win_90": 0.51,
            "reconciled_p_draw_90": 0.275,
            "reconciled_p_away_win_90": 0.215,
            "reconciled_shootout_lean": 0.62,
            "reconciled_p_advance_home": 0.6805,
            "rationale": "Blend A's grid with B's read; net close to A.",
            "provenance": {
                "model": "claude-opus-4-8",
                "effort": "xhigh",
                "temperature": "not set; removed on Opus 4.8, determinism best effort",
                "cutoff": "2026-06-26",
                "a_ref": "smoke://A",
                "b_ref": "smoke://B",
                "dossier_ref": "smoke://dossier",
                "committed_at": None,
            },
        },
        {
            "match": 74,
            "home_team": "Argentina",
            "away_team": "Norway",
            "took_from_a": "the calibrated three-way magnitude",
            "took_from_b": "the momentum read on Norway",
            "reconciled_p_home_win_90": 0.555,
            "reconciled_p_draw_90": 0.255,
            "reconciled_p_away_win_90": 0.19,
            "reconciled_shootout_lean": 0.66,
            "reconciled_p_advance_home": 0.7233,
            "rationale": "A and B agree closely; reconciled call sits between them.",
            "provenance": {
                "model": "claude-opus-4-8",
                "effort": "xhigh",
                "temperature": "not set; removed on Opus 4.8, determinism best effort",
                "cutoff": "2026-06-26",
                "a_ref": "smoke://A",
                "b_ref": "smoke://B",
                "dossier_ref": "smoke://dossier",
                "committed_at": None,
            },
        },
    ],
}

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

    # 2. Schemas and probability validity for A, B, C.
    for kind, document in (("A", FIXTURE_A), ("B", FIXTURE_B), ("C", FIXTURE_C)):
        try:
            schemas.validate_document(document, kind)
            record(f"schema_{kind}_validates", True, "valid against schema")
        except Exception as error:  # noqa: BLE001 - report any validation failure
            record(f"schema_{kind}_validates", False, str(error))

    try:
        schemas.validate_document(CLEAN_DOSSIER, "dossier")
        record("schema_dossier_validates", True, "valid against schema")
    except Exception as error:  # noqa: BLE001
        record("schema_dossier_validates", False, str(error))

    valid_dists = True
    for kind, document in (("A", FIXTURE_A), ("B", FIXTURE_B), ("C", FIXTURE_C)):
        keys = score_llm.keys_for(kind)
        for tie in document["ties"]:
            try:
                validity.assert_prediction_valid(tie, keys)
            except ValueError:
                valid_dists = False
    record("three_way_distributions_valid", valid_dists, "all three-ways sum to one")

    cited = all(
        validity.nonzero_factors_are_cited(tie["factor_scores"])
        for tie in FIXTURE_B["ties"]
    )
    record("nonzero_factors_cited", cited, "every nonzero factor cites evidence")

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
