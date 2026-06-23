"""Layer 5, Monte Carlo, the real World Cup 2026 tournament.

Owns the tournament simulator with the actual 2026 format and logic. Group
matches are simulated as scorelines, since goals are needed for the tiebreaker
ladder, then teams are ranked within each group by the exact FIFA ladder,
head-to-head before overall goal difference. The eight best third-placed teams
are ranked on a separate ladder with no head-to-head and mapped into the round of
32 by the official FIFA assignment table. The knockout is single elimination from
the round of 32 to the final.

Sources, all reproduced into committed inputs rather than scraped at run time: the
group draw in data/tournament, the 495-row round-of-32 third-place assignment
table in data/tournament/r32_thirdplace_assignment_2026.csv parsed from the
published FIFA combinations table, and the bracket progression encoded below from
the published match schedule.

Venue rule: hosts Mexico, Canada, and United States play their group matches at
home; every other group match and every knockout match is neutral.

Fair play tiebreaker: the model cannot predict cards, so for a simulated tie that
reaches the fair-play step there is no card data and it falls back to the FIFA
ranking proxy and then to a random draw, the drawing of lots. How often this
fallback is reached is counted and reported, so we know how rarely it matters.
This is a deliberate, honest design choice, not a fabricated card model.

Knockout tiebreaker: a 90-minute result is simulated from the model, and a draw is
resolved as extra time then penalties by a single strength-weighted coin flip on
the two teams' ratings, kept deliberately simple and documented.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

# Hosts play their group matches at home. Every other match is neutral.
HOSTS = ("United States", "Canada", "Mexico")

GROUP_COLUMN = "group"
TEAM_COLUMN = "team"

WIN_POINTS = 3
DRAW_POINTS = 1

# Round of 32, from the published schedule. Each match is (number, slot, slot).
# A slot is ("W", group) winner, ("RU", group) runner-up, or ("3W", group) the
# third-place team assigned to that group winner by the assignment table.
ROUND_OF_32 = [
    (73, ("RU", "A"), ("RU", "B")),
    (74, ("W", "E"), ("3W", "E")),
    (75, ("W", "F"), ("RU", "C")),
    (76, ("W", "C"), ("RU", "F")),
    (77, ("W", "I"), ("3W", "I")),
    (78, ("RU", "E"), ("RU", "I")),
    (79, ("W", "A"), ("3W", "A")),
    (80, ("W", "L"), ("3W", "L")),
    (81, ("W", "D"), ("3W", "D")),
    (82, ("W", "G"), ("3W", "G")),
    (83, ("RU", "K"), ("RU", "L")),
    (84, ("W", "H"), ("RU", "J")),
    (85, ("W", "B"), ("3W", "B")),
    (86, ("W", "J"), ("RU", "H")),
    (87, ("W", "K"), ("3W", "K")),
    (88, ("RU", "D"), ("RU", "G")),
]

# The eight group winners that face a third-placed team, in match order.
WINNERS_WITH_THIRD = ["E", "I", "A", "L", "D", "G", "B", "K"]

# Knockout bracket: (match, source match 1, source match 2). Winners feed forward.
BRACKET = [
    (89, 74, 77),
    (90, 73, 75),
    (91, 76, 78),
    (92, 79, 80),
    (93, 83, 84),
    (94, 81, 82),
    (95, 86, 88),
    (96, 85, 87),
    (97, 89, 90),
    (98, 93, 94),
    (99, 91, 92),
    (100, 95, 96),
    (101, 97, 98),
    (102, 99, 100),
    (104, 101, 102),
]

# Round size reached by the participants of each bracket match, for stage stats.
PARTICIPANT_ROUND = {
    **{number: 32 for number, _, _ in ROUND_OF_32},
    89: 16,
    90: 16,
    91: 16,
    92: 16,
    93: 16,
    94: 16,
    95: 16,
    96: 16,
    97: 8,
    98: 8,
    99: 8,
    100: 8,
    101: 4,
    102: 4,
    104: 2,
}
FINAL_MATCH = 104

STAGE_FIELDS = {
    32: "p_round_of_32",
    16: "p_round_of_16",
    8: "p_quarter_final",
    4: "p_semi_final",
    2: "p_final",
    1: "p_win",
}

# score_fn(home, away, neutral, rng) -> (home_goals, away_goals).
ScoreFn = Callable[[str, str, bool, Any], tuple[int, int]]
Rng = Any


def load_structure(path: Any) -> dict[str, list[str]]:
    """Read the dated group draw and return group to teams, comment lines skipped."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tournament structure not found: {path}. Expected a committed, dated "
            f"CSV with columns {GROUP_COLUMN} and {TEAM_COLUMN}."
        )
    df = pd.read_csv(path, comment="#")
    missing = [
        column for column in (GROUP_COLUMN, TEAM_COLUMN) if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Tournament structure {path} is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}."
        )
    groups: dict[str, list[str]] = {}
    for group, team in zip(df[GROUP_COLUMN], df[TEAM_COLUMN], strict=True):
        groups.setdefault(str(group).strip(), []).append(str(team).strip())
    return groups


def load_assignment_table(path: Any) -> dict[str, dict[str, str]]:
    """Read the round-of-32 third-place assignment table.

    Returns a lookup from a sorted eight-letter combination of qualifying
    third-place groups to a dict mapping each winner group to the third-place
    group assigned to it. This is the official FIFA Annex C table.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Round of 32 assignment table not found: {path}.")
    df = pd.read_csv(path)
    table: dict[str, dict[str, str]] = {}
    for row in df.itertuples(index=False):
        combo = str(row.thirds).strip()
        table[combo] = {
            winner: str(getattr(row, f"winner_{winner}")).strip()
            for winner in WINNERS_WITH_THIRD
        }
    return table


def _is_neutral(value: Any) -> bool:
    """Interpret the neutral column robustly across bool and text encodings."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1")
    return bool(value)


def _match_venue(
    team_a: str, team_b: str, hosts: tuple[str, ...]
) -> tuple[str, str, bool]:
    """Return (home, away, neutral) for a group match, hosts at home."""
    a_host = team_a in hosts
    b_host = team_b in hosts
    if a_host and not b_host:
        return team_a, team_b, False
    if b_host and not a_host:
        return team_b, team_a, False
    return team_a, team_b, True


def play_group(
    teams: list[str],
    score_fn: ScoreFn,
    rng: Rng,
    hosts: tuple[str, ...] = HOSTS,
    known_results: dict[frozenset, tuple[str, str, int, int]] | None = None,
) -> list[tuple[str, str, int, int]]:
    """Play a group round robin and return the match results as scorelines.

    A fixture already played in reality, keyed in known_results by the two teams,
    keeps its real result, so a live forecast is anchored to the matches that have
    happened. Every unplayed fixture is simulated. With no known results this is a
    full pre-tournament simulation.
    """
    known_results = known_results or {}
    results = []
    for team_a, team_b in itertools.combinations(teams, 2):
        known = known_results.get(frozenset((team_a, team_b)))
        if known is not None:
            results.append(known)
            continue
        home, away, neutral = _match_venue(team_a, team_b, hosts)
        home_goals, away_goals = score_fn(home, away, neutral, rng)
        results.append((home, away, int(home_goals), int(away_goals)))
    return results


def build_known_results(
    matches: pd.DataFrame, structure: dict[str, list[str]]
) -> dict[frozenset, tuple[str, str, int, int]]:
    """Extract the real, played World Cup 2026 group results from the data.

    Returns a lookup keyed by the two teams to the real scoreline, for matches in
    the 2026 finals between two teams in the same group. These anchor the live
    forecast. Empty before the tournament, since no match has been played.
    """
    team_to_group = {
        team: group for group, teams in structure.items() for team in teams
    }
    played = matches.dropna(subset=["home_score", "away_score"]).copy()
    dates = pd.to_datetime(played["date"])
    finals = played[
        (played["tournament"] == "FIFA World Cup")
        & (dates >= pd.Timestamp("2026-06-01"))
    ]
    known: dict[frozenset, tuple[str, str, int, int]] = {}
    for row in finals.itertuples(index=False):
        home, away = row.home_team, row.away_team
        group_home = team_to_group.get(home)
        if group_home is not None and group_home == team_to_group.get(away):
            known[frozenset((home, away))] = (
                home,
                away,
                int(row.home_score),
                int(row.away_score),
            )
    return known


def current_standings(
    structure: dict[str, list[str]],
    known_results: dict[frozenset, tuple[str, str, int, int]],
) -> dict[str, list[tuple[str, int, int, int, int]]]:
    """Group standings from the real results so far: team, played, points, GD, GF.

    A snapshot of where the groups actually stand at the cutoff, ranked by points,
    goal difference, and goals over the matches played. Partial, since not every
    fixture has been played.
    """
    standings: dict[str, list[tuple[str, int, int, int, int]]] = {}
    for group, teams in structure.items():
        results = [
            known_results[frozenset((team_a, team_b))]
            for team_a, team_b in itertools.combinations(teams, 2)
            if frozenset((team_a, team_b)) in known_results
        ]
        ranked = sorted(teams, key=lambda t: team_stats(t, results), reverse=True)
        rows = []
        for team in ranked:
            played = sum(1 for result in results if team in (result[0], result[1]))
            points, goal_difference, goals_for = team_stats(team, results)
            rows.append((team, played, points, goal_difference, goals_for))
        standings[group] = rows
    return standings


def team_stats(
    team: str,
    results: list[tuple[str, str, int, int]],
    among: set[str] | None = None,
) -> tuple[int, int, int]:
    """Points, goal difference, and goals scored for a team over matches.

    When among is given, only matches between teams in that set count, which is
    how the head-to-head criteria are computed.
    """
    points = goal_difference = goals_for = 0
    for home, away, home_goals, away_goals in results:
        if among is not None and (home not in among or away not in among):
            continue
        if team == home:
            scored, conceded = home_goals, away_goals
        elif team == away:
            scored, conceded = away_goals, home_goals
        else:
            continue
        goal_difference += scored - conceded
        goals_for += scored
        if scored > conceded:
            points += WIN_POINTS
        elif scored == conceded:
            points += DRAW_POINTS
    return points, goal_difference, goals_for


def _group_by_key(teams: list[str], key: Callable[[str], Any]) -> list[list[str]]:
    """Split a list, already sorted by key descending, into equal-key runs."""
    runs: list[list[str]] = []
    for team in teams:
        if runs and key(runs[-1][0]) == key(team):
            runs[-1].append(team)
        else:
            runs.append([team])
    return runs


def _residual_order(teams: list[str], context: dict[str, Any], rng: Rng) -> list[str]:
    """Resolve a tie that reached the fair-play step: fair play, FIFA, lots.

    The model has no card data for simulated matches, so fair play is skipped and
    the tie falls back to the FIFA ranking, proxied by the team rating, then to a
    random draw. Reaching here is recorded so the fallback frequency can be
    reported.
    """
    context["fair_play_reached"] = True
    ratings = context.get("ratings", {})
    # FIFA ranking proxy is the team rating, nearly unique, so the random draw
    # almost never decides. The random key is the drawing of lots.
    return sorted(
        teams, key=lambda team: (ratings.get(team, 0.0), rng.random()), reverse=True
    )


def _resolve_overall(
    teams: list[str],
    results: list[tuple[str, str, int, int]],
    context: dict[str, Any],
    rng: Rng,
) -> list[str]:
    """Rank tied teams by overall goal difference, goals, then the residual ladder."""
    ordered = sorted(teams, key=lambda t: team_stats(t, results)[1:], reverse=True)
    out: list[str] = []
    for run in _group_by_key(ordered, key=lambda t: team_stats(t, results)[1:]):
        if len(run) == 1:
            out.extend(run)
        else:
            out.extend(_residual_order(run, context, rng))
    return out


def _resolve_points_tie(
    tied: list[str],
    results: list[tuple[str, str, int, int]],
    context: dict[str, Any],
    rng: Rng,
) -> list[str]:
    """Resolve teams level on points via the head-to-head ladder, then overall.

    Head-to-head points, goal difference, and goals among the tied teams come
    first. Whenever a subset stays level, the head-to-head criteria are reapplied
    to just that subset, the re-assessment from the top of the ladder. Only when
    head-to-head separates nobody does it fall through to overall goal difference,
    which is why head-to-head precedes overall goal difference, not the reverse.
    """
    if len(tied) == 1:
        return tied
    among = set(tied)
    key = lambda t: team_stats(t, results, among=among)  # noqa: E731
    ordered = sorted(tied, key=key, reverse=True)
    runs = _group_by_key(ordered, key=key)
    if len(runs) == 1:
        # Head-to-head separated nobody; fall through to the overall criteria.
        return _resolve_overall(tied, results, context, rng)
    out: list[str] = []
    for run in runs:
        if len(run) == 1:
            out.extend(run)
        else:
            # Still level: reapply the head-to-head ladder among only these teams.
            out.extend(_resolve_points_tie(run, results, context, rng))
    return out


def rank_group(
    teams: list[str],
    results: list[tuple[str, str, int, int]],
    context: dict[str, Any],
    rng: Rng,
) -> list[str]:
    """Rank a group: overall points first, then the tiebreaker ladder for ties."""
    points = lambda t: team_stats(t, results)[0]  # noqa: E731
    ordered = sorted(teams, key=points, reverse=True)
    out: list[str] = []
    for run in _group_by_key(ordered, key=points):
        if len(run) == 1:
            out.extend(run)
        else:
            out.extend(_resolve_points_tie(run, results, context, rng))
    return out


def rank_third_place(
    third_teams: list[tuple[str, str]],
    overall_stats: dict[str, tuple[int, int, int]],
    context: dict[str, Any],
    rng: Rng,
) -> list[tuple[str, str]]:
    """Rank the twelve third-placed teams on their own ladder, no head-to-head.

    Ranks by overall group points, goal difference, goals scored, then the
    residual ladder of fair play, FIFA ranking, and lots. These teams never met,
    so head-to-head does not apply. Returns the teams ordered best first.
    """

    def stat_key(entry: tuple[str, str]) -> tuple[int, int, int]:
        return overall_stats[entry[1]]

    ordered = sorted(third_teams, key=stat_key, reverse=True)
    out: list[tuple[str, str]] = []
    for run in _group_by_key(ordered, key=lambda e: stat_key(e)):  # type: ignore[arg-type]
        if len(run) == 1:
            out.extend(run)
        else:
            teams = [team for _, team in run]
            order = _residual_order(teams, context, rng)
            by_team = {team: group for group, team in run}
            out.extend((by_team[team], team) for team in order)
    return out


def _resolve_slot(
    slot: tuple[str, str],
    winners: dict[str, str],
    runners: dict[str, str],
    assignment: dict[str, str],
    third_by_group: dict[str, str],
) -> str:
    """Resolve a round-of-32 slot spec to a team."""
    kind, group = slot
    if kind == "W":
        return winners[group]
    if kind == "RU":
        return runners[group]
    # ("3W", winner_group): the third-place team assigned to that winner.
    return third_by_group[assignment[group]]


def assign_round_of_32(
    winners: dict[str, str],
    runners: dict[str, str],
    qualifying_thirds: list[tuple[str, str]],
    assignment_table: dict[str, dict[str, str]],
) -> dict[int, tuple[str, str]]:
    """Build the sixteen round-of-32 matchups from the official assignment table.

    qualifying_thirds is the eight best third-placed teams as (group, team). The
    sorted combination of their groups keys into the assignment table, which says
    which third-place group each winner faces. Returns match number to (team, team).
    """
    combo = "".join(sorted(group for group, _ in qualifying_thirds))
    if combo not in assignment_table:
        raise ValueError(f"No round-of-32 assignment for third-place combo {combo}.")
    assignment = assignment_table[combo]
    third_by_group = {group: team for group, team in qualifying_thirds}

    matches: dict[int, tuple[str, str]] = {}
    for number, slot_a, slot_b in ROUND_OF_32:
        team_a = _resolve_slot(slot_a, winners, runners, assignment, third_by_group)
        team_b = _resolve_slot(slot_b, winners, runners, assignment, third_by_group)
        matches[number] = (team_a, team_b)
    return matches


def _knockout_winner(
    team_a: str, team_b: str, score_fn: ScoreFn, rng: Rng, context: dict[str, Any]
) -> str:
    """Play a neutral knockout match, resolving a draw by a strength-weighted flip."""
    home_goals, away_goals = score_fn(team_a, team_b, True, rng)
    if home_goals > away_goals:
        return team_a
    if away_goals > home_goals:
        return team_b
    # Extra time then penalties, modelled as one strength-weighted coin flip.
    ratings = context.get("ratings", {})
    rating_a = ratings.get(team_a, 0.0)
    rating_b = ratings.get(team_b, 0.0)
    p_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    return team_a if rng.random() < p_a else team_b


def simulate_tournament(
    structure: dict[str, list[str]],
    score_fn: ScoreFn,
    assignment_table: dict[str, dict[str, str]],
    context: dict[str, Any],
    rng: Rng,
    hosts: tuple[str, ...] = HOSTS,
) -> dict[str, Any]:
    """Simulate one full tournament and return a detailed result.

    Plays every group, ranks each by the tiebreaker ladder, ranks the third-placed
    teams, maps the eight best into the round of 32 by the assignment table, and
    plays the knockout to a champion. Returns the standings, the qualifying
    thirds, the round-of-32 matchups, the knockout winners, the champion, each
    team's furthest round, and whether the fair-play fallback was reached.
    """
    context["fair_play_reached"] = False
    known_results = context.get("known_results") or {}

    standings: dict[str, list[str]] = {}
    overall_stats: dict[str, tuple[int, int, int]] = {}
    third_teams: list[tuple[str, str]] = []
    for group in sorted(structure):
        teams = structure[group]
        results = play_group(teams, score_fn, rng, hosts, known_results)
        for team in teams:
            overall_stats[team] = team_stats(team, results)
        ranked = rank_group(teams, results, context, rng)
        standings[group] = ranked
        third_teams.append((group, ranked[2]))

    winners = {group: standings[group][0] for group in structure}
    runners = {group: standings[group][1] for group in structure}

    ranked_thirds = rank_third_place(third_teams, overall_stats, context, rng)
    qualifying_thirds = ranked_thirds[:8]

    round_of_32 = assign_round_of_32(
        winners, runners, qualifying_thirds, assignment_table
    )

    furthest: dict[str, int] = {}
    match_winner: dict[int, str] = {}
    for number, (team_a, team_b) in round_of_32.items():
        furthest[team_a] = 32
        furthest[team_b] = 32
        match_winner[number] = _knockout_winner(team_a, team_b, score_fn, rng, context)

    for number, source_a, source_b in BRACKET:
        team_a = match_winner[source_a]
        team_b = match_winner[source_b]
        size = PARTICIPANT_ROUND[number]
        furthest[team_a] = size
        furthest[team_b] = size
        match_winner[number] = _knockout_winner(team_a, team_b, score_fn, rng, context)

    champion = match_winner[FINAL_MATCH]
    furthest[champion] = 1

    return {
        "standings": standings,
        "overall_stats": overall_stats,
        "ranked_thirds": ranked_thirds,
        "qualifying_thirds": qualifying_thirds,
        "round_of_32": round_of_32,
        "match_winner": match_winner,
        "champion": champion,
        "furthest": furthest,
        "fair_play_reached": context["fair_play_reached"],
    }


def run_simulations(
    structure: dict[str, list[str]],
    score_fn: ScoreFn,
    ratings: dict[str, float],
    assignment_table: dict[str, dict[str, str]],
    *,
    n_sims: int = 10000,
    seed: int = 42,
    hosts: tuple[str, ...] = HOSTS,
    fairplay: dict[str, float] | None = None,
    known_results: dict[frozenset, tuple[str, str, int, int]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run n_sims tournaments and aggregate per-team stage probabilities.

    known_results anchors the simulation to the real matches already played, so
    the live forecast simulates only the remaining matches. Returns the per-team
    probability records and a meta dict that includes how often a simulation
    reached the fair-play fallback, so we know how rarely it matters. Deterministic
    given the seed, score_fn, and inputs.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    context = {
        "ratings": ratings,
        "fairplay": fairplay,
        "known_results": known_results or {},
    }
    teams = [team for group in sorted(structure) for team in structure[group]]
    round_sizes = list(STAGE_FIELDS)

    reach_counts = {team: dict.fromkeys(round_sizes, 0) for team in teams}
    fallback_sims = 0
    for _ in range(n_sims):
        result = simulate_tournament(
            structure, score_fn, assignment_table, context, rng, hosts
        )
        if result["fair_play_reached"]:
            fallback_sims += 1
        for team, reached in result["furthest"].items():
            for size in round_sizes:
                if size >= reached:
                    reach_counts[team][size] += 1

    records: list[dict[str, Any]] = []
    for team in teams:
        record: dict[str, Any] = {"team": team}
        for size in round_sizes:
            record[STAGE_FIELDS[size]] = reach_counts[team][size] / n_sims
        records.append(record)

    meta = {
        "n_sims": n_sims,
        "fair_play_fallback_sims": fallback_sims,
        "fair_play_fallback_fraction": fallback_sims / n_sims,
    }
    return records, meta
