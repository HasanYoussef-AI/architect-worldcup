"""Verification gate: the round-aware quarantine, the LLM-phase leakage guard.

Hermetic, no network. Proves the gate preserves legitimate pre-cutoff group-stage
inputs while failing loud on results and advancement for the target round and
later, and on betting or prediction-market content. The planted target-round
result is the same one the smoke test uses.
"""

from __future__ import annotations

import pytest

from architect_wc.llm import quarantine, smoke


def _dossier(factor: str, claim: str, **fields) -> dict:
    entry = {"claim": claim, "source": "x", "source_tier": 1, **fields}
    return {"factors": {factor: [entry]}}


def test_clean_dossier_passes() -> None:
    quarantine.assert_no_dossier_leakage(smoke.CLEAN_DOSSIER, "R32", smoke.FIXTURES)


def test_planted_target_round_result_is_caught() -> None:
    with pytest.raises(quarantine.LeakageError) as excinfo:
        quarantine.assert_no_dossier_leakage(
            smoke.poisoned_dossier(), "R32", smoke.FIXTURES
        )
    assert any("R32" in v.reason for v in excinfo.value.violations)


def test_pre_cutoff_group_scoreline_is_preserved() -> None:
    # A group-stage result from before the cutoff is a legitimate form input.
    dossier = _dossier("recent_form", "Spain won 2-0 over Uruguay in the group stage.")
    assert quarantine.find_violations(dossier, "R32") == []


def test_pre_cutoff_head_to_head_between_fixture_teams_is_preserved() -> None:
    # A genuine meeting dated strictly before the cutoff is the head-to-head
    # factor's legitimate content and is preserved.
    dossier = _dossier(
        "historical_head_to_head",
        "Spain and Portugal drew 1-1 in 2022.",
        meeting_date="2022-09-27",
    )
    violations = quarantine.find_violations(
        dossier, "R32", smoke.FIXTURES, cutoff="2026-06-26"
    )
    assert violations == []


def test_head_to_head_dated_on_or_after_cutoff_is_caught() -> None:
    # A meeting between the two fixture teams dated on the cutoff is not a past
    # meeting; it could be the current-tournament result framed as head-to-head.
    dossier = _dossier(
        "historical_head_to_head",
        "Spain and Portugal drew 1-1.",
        meeting_date="2026-06-26",
    )
    violations = quarantine.find_violations(
        dossier, "R32", smoke.FIXTURES, cutoff="2026-06-26"
    )
    assert violations and "head-to-head" in violations[0].reason


def test_head_to_head_with_no_date_between_fixture_teams_is_caught() -> None:
    # A head-to-head result with no meeting date cannot be shown to be a past
    # meeting, so it is flagged.
    dossier = _dossier(
        "historical_head_to_head", "Spain and Portugal drew 1-1 recently."
    )
    violations = quarantine.find_violations(
        dossier, "R32", smoke.FIXTURES, cutoff="2026-06-26"
    )
    assert violations and "head-to-head" in violations[0].reason


def test_head_to_head_does_not_smuggle_a_current_knockout_result() -> None:
    # The case the date bound exists for: two later-round fixture teams with a real
    # past meeting and a current-tournament meeting. The pre-cutoff one is
    # preserved; the on-or-after-cutoff one is caught even though both name the
    # fixture pair.
    fixtures = [{"match": 101, "home_team": "Spain", "away_team": "France"}]
    past = _dossier(
        "historical_head_to_head",
        "Spain and France drew 1-1 in 2024.",
        meeting_date="2024-06-01",
    )
    assert quarantine.find_violations(past, "SF", fixtures, cutoff="2026-07-14") == []
    current = _dossier(
        "historical_head_to_head",
        "Spain and France drew 1-1.",
        meeting_date="2026-07-14",
    )
    assert quarantine.find_violations(current, "SF", fixtures, cutoff="2026-07-14")


def test_reaching_the_target_round_is_allowed() -> None:
    dossier = _dossier(
        "recent_form", "Portugal reached the round of 32 as group runners-up."
    )
    assert quarantine.find_violations(dossier, "R32") == []


def test_advancement_to_a_later_round_is_caught() -> None:
    dossier = _dossier("recent_form", "Spain advanced to the round of 16.")
    violations = quarantine.find_violations(dossier, "R32")
    assert violations and "R16" in violations[0].reason


def test_target_round_result_is_caught_without_fixtures() -> None:
    dossier = _dossier(
        "tactical_matchup", "Brazil knocked out Morocco in the round of 32."
    )
    violations = quarantine.find_violations(dossier, "R32")
    assert violations and "R32" in violations[0].reason


def test_bare_fixture_matchup_result_is_caught_via_fixtures() -> None:
    # No round word, but naming both teams of a tie with a result is that tie's
    # outcome. The fixtures supply the context.
    dossier = _dossier("recent_form", "Argentina beat Norway 1-0.")
    violations = quarantine.find_violations(dossier, "R32", smoke.FIXTURES)
    assert violations


def test_market_content_is_always_caught() -> None:
    dossier = _dossier(
        "recent_form", "Polymarket has Spain at an implied probability of 0.62."
    )
    violations = quarantine.find_violations(dossier, "R32")
    assert violations and "market" in violations[0].reason


def test_a_later_target_round_allows_earlier_round_results() -> None:
    # When predicting the quarter-finals, a round-of-32 result is legitimate history.
    dossier = _dossier("recent_form", "Spain beat Portugal 2-0 in the round of 32.")
    assert quarantine.find_violations(dossier, "QF") == []
    # But a quarter-final result is forbidden for a quarter-final prediction.
    qf_leak = _dossier("recent_form", "Spain won the quarter-final 1-0.")
    assert quarantine.find_violations(qf_leak, "QF")


# --- GROUP target: a group-stage fixture is a valid research target. -------------

_GROUP_FIXTURES = [{"match": 64, "home_team": "Uruguay", "away_team": "Spain"}]


def test_group_target_preserves_other_pre_cutoff_group_form() -> None:
    # Predicting a group match, an earlier group result for a different pairing is
    # legitimate form, not leakage.
    dossier = _dossier("recent_form", "Spain beat Cape Verde 3-0 in their opener.")
    assert (
        quarantine.find_violations(
            dossier, "GROUP", _GROUP_FIXTURES, cutoff="2026-06-25"
        )
        == []
    )


def test_group_target_catches_the_fixtures_own_result() -> None:
    # The target group match's own result is leakage, caught by the fixture pair.
    dossier = _dossier("recent_form", "Uruguay beat Spain 1-0.")
    violations = quarantine.find_violations(
        dossier, "GROUP", _GROUP_FIXTURES, cutoff="2026-06-25"
    )
    assert violations


def test_group_target_catches_any_knockout_result() -> None:
    # Every knockout round is strictly later than the group stage, so a knockout
    # result or advancement reveals a future outcome and is caught.
    leak = _dossier("psychological_momentum", "Spain reached the round of 16.")
    violations = quarantine.find_violations(leak, "GROUP")
    assert violations and "later round" in violations[0].reason


# --- Tournament and date scoping, plus the source-attribution fix. ---------------


def test_historical_titles_and_other_competitions_are_preserved() -> None:
    # A coach's pedigree across other competitions and past years is legitimate
    # analyst content, not a 2026 result, even though it names finals and titles.
    dossier = _dossier(
        "coaching_staff",
        "De la Fuente led Spain to the 2023 Nations League and EURO 2024 titles "
        "before a penalty-shootout defeat in the 2025 Nations League final.",
    )
    assert (
        quarantine.find_violations(
            dossier, "GROUP", _GROUP_FIXTURES, cutoff="2026-06-25"
        )
        == []
    )


def test_final_group_game_phrase_is_not_the_world_cup_final() -> None:
    # "final group game" is the last group match, not the World Cup Final, and a
    # group win in it is legitimate pre-cutoff form.
    dossier = _dossier(
        "strategic_incentives",
        "Spain entered the final group game having taken four points from a win "
        "over Saudi Arabia and a draw with Cabo Verde.",
    )
    assert (
        quarantine.find_violations(
            dossier, "GROUP", _GROUP_FIXTURES, cutoff="2026-06-25"
        )
        == []
    )


def test_group_result_with_opponent_in_source_url_is_preserved() -> None:
    # The source-attribution bug: a Uruguay group result cited from an ESPN URL
    # whose slug contains "spain" must not read as the Uruguay-Spain tie's result.
    dossier = _dossier(
        "recent_form",
        "Uruguay drew 1-1 with Saudi Arabia in their opening Group H match.",
        source="https://www.espn.com/soccer/team/_/id/164/spain",
        team="Uruguay",
    )
    assert (
        quarantine.find_violations(
            dossier, "GROUP", _GROUP_FIXTURES, cutoff="2026-06-25"
        )
        == []
    )


def test_2026_world_cup_final_reference_is_still_caught() -> None:
    # The real leak the scoping must preserve: a current-tournament final reference.
    reached = _dossier(
        "psychological_momentum", "Spain reached the 2026 World Cup final."
    )
    assert quarantine.find_violations(reached, "GROUP")
    result = _dossier("recent_form", "Spain won the 2026 World Cup final 2-1.")
    assert quarantine.find_violations(result, "GROUP")


def test_ambiguous_current_forward_reference_stays_flagged() -> None:
    # No historical or other-competition marker: treated as current and flagged,
    # the conservative error for an ambiguous 2026 forward reference.
    dossier = _dossier("psychological_momentum", "Spain reached the final.")
    assert quarantine.find_violations(dossier, "GROUP")
