"""Verification gate: the real World Cup 2026 tournament logic, structural only.

Hermetic, mostly on constructed cases so the tiebreaker and bracket logic is
proven exactly. The real-data name-resolution check is guarded to run wherever
the snapshot exists. These gates cover the parts where a subtle bug would
silently corrupt the bracket: head-to-head before overall goal difference, the
three-way re-assessment, the third-place ladder, the round-of-32 assignment, and
that every group team resolves to a rating.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from architect_wc import ingest, simulate

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
STRUCTURE_PATH = REPO_ROOT / "data" / "tournament" / "wc2026_groups_2026-06-10.csv"
ASSIGNMENT_PATH = (
    REPO_ROOT / "data" / "tournament" / "r32_thirdplace_assignment_2026.csv"
)

GROUPS = "ABCDEFGHIJKL"


def _context() -> dict:
    return {"ratings": {}}


def test_head_to_head_beats_overall_goal_difference() -> None:
    # X and Y finish level on points. X has the better overall goal difference,
    # but Y beat X head-to-head, so the ladder must rank Y above X.
    results = [
        ("Y", "X", 1, 0),  # Y beats X head-to-head
        ("X", "Z", 5, 0),  # X piles up overall goal difference
        ("X", "W", 3, 0),
        ("Z", "Y", 1, 0),
        ("Y", "W", 1, 0),
        ("W", "Z", 2, 0),
    ]
    ranked = simulate.rank_group(
        ["X", "Y", "Z", "W"], results, _context(), np.random.default_rng(0)
    )
    # X has overall GD +7, Y only +1; sorting by overall GD first would put X top.
    assert ranked.index("Y") < ranked.index("X")
    assert ranked[0] == "Y"


def test_three_way_tie_reassesses_after_one_team_separates() -> None:
    # A, B, C all finish on 6 points. Head-to-head separates A first. B and C
    # remain level on every head-to-head aggregate, so the criteria are reapplied
    # to just B and C, where B beat C, ranking B above C, even though C has the
    # better overall goal difference.
    results = [
        ("A", "B", 3, 0),
        ("C", "A", 2, 1),
        ("B", "C", 2, 0),
        ("A", "D", 1, 0),
        ("B", "D", 1, 0),
        ("C", "D", 5, 0),  # C inflates its overall goal difference
    ]
    ranked = simulate.rank_group(
        ["A", "B", "C", "D"], results, _context(), np.random.default_rng(0)
    )
    assert ranked[0] == "A"
    # Overall GD would order C before B; the head-to-head re-assessment must not.
    assert ranked.index("B") < ranked.index("C")
    assert ranked[3] == "D"


def test_third_place_ladder_has_no_head_to_head_and_selects_eight() -> None:
    # Twelve third-placed teams with distinct overall stats, ranked with no
    # head-to-head. The best eight by points, goal difference, goals advance.
    third_teams = [(g, f"3_{g}") for g in GROUPS]
    overall_stats = {f"3_{g}": (4 - i // 4, 5 - i, 9 - i) for i, g in enumerate(GROUPS)}
    ranked = simulate.rank_third_place(
        third_teams, overall_stats, _context(), np.random.default_rng(0)
    )
    assert len(ranked) == 12
    # Ranking matches a plain sort by (points, gd, goals), no head-to-head.
    expected = sorted(third_teams, key=lambda e: overall_stats[e[1]], reverse=True)
    assert ranked == expected
    advancing = ranked[:8]
    assert len(advancing) == 8
    assert len({team for _, team in advancing}) == 8


def test_round_of_32_assignment_is_valid_for_every_combination() -> None:
    table = simulate.load_assignment_table(ASSIGNMENT_PATH)
    winners = {g: f"W_{g}" for g in GROUPS}
    runners = {g: f"RU_{g}" for g in GROUPS}

    def group_of(team: str) -> str:
        return team.split("_")[1]

    checked = 0
    for combo in itertools.combinations(GROUPS, 8):
        qualifying = [(g, f"3_{g}") for g in combo]
        matches = simulate.assign_round_of_32(winners, runners, qualifying, table)
        assert len(matches) == 16  # every slot filled, no slot empty
        teams = [team for pair in matches.values() for team in pair]
        assert len(teams) == 32
        assert len(set(teams)) == 32  # no team appears twice
        for team_a, team_b in matches.values():
            assert group_of(team_a) != group_of(team_b)  # no group-stage rematch
        checked += 1
    assert checked == 495


def test_all_group_teams_resolve_to_a_rating() -> None:
    structure = simulate.load_structure(STRUCTURE_PATH)
    teams = [team for group in structure.values() for team in group]
    assert len(teams) == 48
    snapshots = sorted(RAW_DIR.glob("results_snapshot_*.csv"))
    if not snapshots:
        pytest.skip("no local results snapshot to resolve names against")
    df = ingest.load_raw(snapshots[-1])
    rated = set(df["home_team"]) | set(df["away_team"])
    unmatched = [team for team in teams if team not in rated]
    assert not unmatched, f"group teams with no rating: {unmatched}"


def _rating_score_fn(ratings: dict[str, float]) -> simulate.ScoreFn:
    def score_fn(home, away, neutral, rng):
        edge = 0.0 if neutral else 60.0
        diff = ratings.get(home, 1500.0) - ratings.get(away, 1500.0) + edge
        lam_home = max(0.15, 1.4 + diff / 400.0)
        lam_away = max(0.15, 1.4 - diff / 400.0)
        return int(rng.poisson(lam_home)), int(rng.poisson(lam_away))

    return score_fn


def _small_field() -> tuple[dict, dict]:
    structure = simulate.load_structure(STRUCTURE_PATH)
    teams = [team for group in structure.values() for team in group]
    # A clear strength gradient so the simulation is well-behaved.
    ratings = {team: 2100.0 - 8.0 * index for index, team in enumerate(teams)}
    return structure, ratings


def test_simulation_probabilities_are_valid_and_reproducible() -> None:
    structure, ratings = _small_field()
    table = simulate.load_assignment_table(ASSIGNMENT_PATH)
    score_fn = _rating_score_fn(ratings)
    first, meta = simulate.run_simulations(
        structure, score_fn, ratings, table, n_sims=120, seed=42
    )
    second, _ = simulate.run_simulations(
        structure, score_fn, ratings, table, n_sims=120, seed=42
    )
    assert first == second  # deterministic under a fixed seed

    stages = [
        "p_round_of_32",
        "p_round_of_16",
        "p_quarter_final",
        "p_semi_final",
        "p_final",
        "p_win",
    ]
    for record in first:
        for field in stages:
            assert 0.0 <= record[field] <= 1.0
        for earlier, later in zip(stages, stages[1:], strict=False):
            assert record[earlier] >= record[later] - 1e-12
    assert abs(sum(record["p_win"] for record in first) - 1.0) <= 1e-9
    assert 0.0 <= meta["fair_play_fallback_fraction"] <= 1.0


def test_play_group_anchors_on_known_results() -> None:
    # A known result is used verbatim and never simulated; unknown fixtures fall
    # to the score_fn. A score_fn that raises proves the known fixtures bypass it.
    teams = ["P", "Q", "R", "S"]
    known = {frozenset(("P", "Q")): ("P", "Q", 3, 0)}

    def exploding_score_fn(home, away, neutral, rng):
        raise AssertionError("known fixtures must not be simulated")

    # All six fixtures known, so the exploding score_fn is never called.
    all_known = {
        frozenset((a, b)): (a, b, 1, 0) for a, b in itertools.combinations(teams, 2)
    }
    results = simulate.play_group(
        teams, exploding_score_fn, np.random.default_rng(0), known_results=all_known
    )
    assert len(results) == 6
    assert ("P", "Q", 3, 0) in simulate.play_group(
        ["P", "Q"], exploding_score_fn, np.random.default_rng(0), known_results=known
    )


def test_current_standings_reflect_real_results() -> None:
    structure = {"A": ["P", "Q", "R", "S"]}
    known = {
        frozenset(("P", "Q")): ("P", "Q", 2, 0),
        frozenset(("R", "S")): ("R", "S", 1, 1),
    }
    standings = simulate.current_standings(structure, known)["A"]
    by_team = {row[0]: row for row in standings}
    # P won, so it leads on points; R and S drew; Q lost.
    assert by_team["P"][2] == 3  # points
    assert by_team["P"][1] == 1  # played
    assert by_team["R"][2] == 1 and by_team["S"][2] == 1
    assert standings[0][0] == "P"


def test_load_structure_skips_comment_lines(tmp_path) -> None:
    path = tmp_path / "groups.csv"
    path.write_text("# note\ngroup,team\nA,W\nA,X\n", encoding="utf-8")
    assert simulate.load_structure(path) == {"A": ["W", "X"]}
