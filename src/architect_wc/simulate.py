"""Layer 5, Monte Carlo.

Owns the tournament simulator. It plays the World Cup many times on the
squad-adjusted team strengths and the Dixon-Coles goal model, then aggregates
how often each team reaches each knockout round and wins. The pure simulation
logic is kept separate from any I/O and from penaltyblog, so the core rules are
unit-testable on a synthetic field with no network and no model fit.

Match outcomes enter through a prob_fn callable: given (home, away, neutral) it
returns the home win, draw, and away win probabilities. run_simulations wires
the real goal model into that callable; tests pass a synthetic one. Keeping the
simulator behind this seam is what lets the verification gate run fast.

Venue rule: World Cup 2026 matches are at neutral venues, the explicit exception
being a group match involving a host nation, the United States, Canada, or
Mexico, which that host plays at home. Neutral is the default and host-home is
the exception, so no phantom home advantage biases the bracket. Knockout matches
are treated as neutral.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Host nations play their group matches at home. Every other match is neutral.
HOSTS = ("United States", "Canada", "Mexico")

GROUP_COLUMN = "group"
TEAM_COLUMN = "team"

WIN_POINTS = 3
DRAW_POINTS = 1

# Knockout round sizes, from the round of 32 down to the champion, mapped to the
# output-contract field names. p_final is reaching the final, p_win is winning
# it. Smaller fields apply when a smaller synthetic bracket is simulated.
STAGE_FIELDS = {
    32: "p_round_of_32",
    16: "p_round_of_16",
    8: "p_quarter_final",
    4: "p_semi_final",
    2: "p_final",
    1: "p_win",
}

# prob_fn(home, away, neutral) -> (p_home_win, p_draw, p_away_win).
ProbFn = Callable[[str, str, bool], tuple[float, float, float]]
Rng = Any


def load_structure(path: Any) -> dict[str, list[str]]:
    """Read a dated tournament structure and return group to teams.

    The CSV has a group column and a team column, with comment lines starting
    with a hash skipped. Raise a clear error if the file or either column is
    missing. Group order and within-group order follow the file.
    """
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


def simulate_match(
    prob_fn: ProbFn,
    team_a: str,
    team_b: str,
    rng: Rng,
    *,
    neutral: bool = True,
    knockout: bool = False,
) -> str | None:
    """Simulate one match and return the winning team, or None for a draw.

    prob_fn(home, away, neutral) gives the home win, draw, and away win
    probabilities. team_a is treated as the home side, which only matters when
    neutral is False. A single uniform draw from rng decides the result. In the
    group stage a draw is a valid outcome and returns None. In a knockout match
    there are no draws: the draw mass is removed and a winner is always carried
    through. Pure given prob_fn and rng.
    """
    p_home, p_draw, p_away = prob_fn(team_a, team_b, neutral)
    roll = float(rng.random())
    if knockout:
        decisive = p_home + p_away
        if decisive <= 0.0:
            return team_a if roll < 0.5 else team_b
        return team_a if roll < p_home / decisive else team_b
    if roll < p_home:
        return team_a
    if roll < p_home + p_draw:
        return None
    return team_b


def _match_venue(
    team_a: str, team_b: str, hosts: tuple[str, ...]
) -> tuple[str, str, bool]:
    """Return (home, away, neutral) for a group match.

    A group match involving exactly one host is played at that host's home, so
    the host is the home side and the match is not neutral. Every other match,
    including the unlikely host against host, is neutral.
    """
    a_host = team_a in hosts
    b_host = team_b in hosts
    if a_host and not b_host:
        return team_a, team_b, False
    if b_host and not a_host:
        return team_b, team_a, False
    return team_a, team_b, True


def simulate_group(
    teams: list[str],
    prob_fn: ProbFn,
    rng: Rng,
    ratings: dict[str, float],
    hosts: tuple[str, ...] = HOSTS,
) -> list[tuple[str, int]]:
    """Play a group as a round robin and return its standings.

    Every pair in the group plays once. A win is worth three points and a draw
    one. The standings are sorted by points, then by pre-tournament rating as a
    deterministic tiebreaker, highest first. Returns a list of (team, points)
    from first to last. Pure given prob_fn and rng.

    Goal difference is not tracked in this outcome-based simulation, so the
    rating tiebreaker stands in for it. That keeps the real run fast and is a
    natural place to deepen later.
    """
    points = dict.fromkeys(teams, 0)
    for team_a, team_b in itertools.combinations(teams, 2):
        home, away, neutral = _match_venue(team_a, team_b, hosts)
        winner = simulate_match(
            prob_fn, home, away, rng, neutral=neutral, knockout=False
        )
        if winner is None:
            points[team_a] += DRAW_POINTS
            points[team_b] += DRAW_POINTS
        else:
            points[winner] += WIN_POINTS
    ranked = sorted(
        teams, key=lambda team: (points[team], ratings.get(team, 0.0)), reverse=True
    )
    return [(team, points[team]) for team in ranked]


def _next_power_of_two(value: int) -> int:
    """Smallest power of two greater than or equal to value."""
    power = 1
    while power < value:
        power *= 2
    return power


def _bracket_seed_order(size: int) -> list[int]:
    """Standard single-elimination seed order for a bracket of given size.

    Returns seeds 1..size arranged so that pairing adjacent entries keeps the
    top seeds in opposite halves for as long as possible, the usual tournament
    bracket. size must be a power of two.
    """
    order = [1]
    while len(order) < size:
        total = len(order) * 2 + 1
        expanded: list[int] = []
        for seed in order:
            expanded.append(seed)
            expanded.append(total - seed)
        order = expanded
    return order


def simulate_tournament(
    groups: dict[str, list[str]],
    prob_fn: ProbFn,
    rng: Rng,
    ratings: dict[str, float],
    hosts: tuple[str, ...] = HOSTS,
) -> tuple[dict[str, int], str]:
    """Simulate one tournament and return each team's furthest round and the winner.

    Plays every group, takes the top two of each group plus the best
    third-placed teams needed to fill the knockout bracket to a power of two,
    seeds them, and plays single-elimination rounds to a champion. The first
    return value maps each qualifier to the size of the furthest round it
    reached, 32 for the round of 32 down to 1 for the champion. Group-stage
    casualties are absent. The second return value is the champion. Pure given
    prob_fn and rng.
    """
    winners: list[str] = []
    runners_up: list[str] = []
    thirds: list[tuple[str, int]] = []
    for group in sorted(groups):
        standings = simulate_group(groups[group], prob_fn, rng, ratings, hosts)
        winners.append(standings[0][0])
        runners_up.append(standings[1][0])
        if len(standings) >= 3:
            thirds.append(standings[2])

    bracket_size = _next_power_of_two(2 * len(groups))
    n_thirds = bracket_size - 2 * len(groups)
    if n_thirds > len(thirds):
        raise ValueError(
            f"Structure yields {len(thirds)} third-placed teams but the bracket "
            f"needs {n_thirds} to reach size {bracket_size}."
        )
    best_thirds = sorted(
        thirds, key=lambda item: (item[1], ratings.get(item[0], 0.0)), reverse=True
    )
    qualifier_thirds = [team for team, _ in best_thirds[:n_thirds]]

    # Seed the qualifiers: winners, then runners-up, then thirds, each ordered by
    # rating, so group winners take the higher seeds. This mirrors the advantage
    # of winning a group and gives a deterministic, reproducible bracket.
    seeded = (
        sorted(winners, key=lambda team: ratings.get(team, 0.0), reverse=True)
        + sorted(runners_up, key=lambda team: ratings.get(team, 0.0), reverse=True)
        + qualifier_thirds
    )
    order = _bracket_seed_order(bracket_size)
    round_teams = [seeded[seed - 1] for seed in order]

    furthest = dict.fromkeys(seeded, bracket_size)
    size = bracket_size
    while size > 1:
        next_round: list[str] = []
        for index in range(0, size, 2):
            winner = simulate_match(
                prob_fn,
                round_teams[index],
                round_teams[index + 1],
                rng,
                neutral=True,
                knockout=True,
            )
            next_round.append(winner)
        size //= 2
        for team in next_round:
            furthest[team] = size
        round_teams = next_round

    return furthest, round_teams[0]


def run_simulations(
    groups: dict[str, list[str]],
    prob_fn: ProbFn,
    ratings: dict[str, float],
    *,
    n_sims: int = 10000,
    seed: int = 42,
    hosts: tuple[str, ...] = HOSTS,
) -> list[dict[str, Any]]:
    """Run n_sims tournaments from a fixed seed and aggregate stage probabilities.

    Returns one record per team in the structure, each with the team name, its
    probability of reaching each knockout round, and its probability of winning,
    every value a count over n_sims. Deterministic: the same groups, prob_fn,
    ratings, seed, and n_sims give identical results.
    """
    rng = np.random.default_rng(seed)
    teams = [team for group in sorted(groups) for team in groups[group]]
    bracket_size = _next_power_of_two(2 * len(groups))
    round_sizes = [size for size in STAGE_FIELDS if size <= bracket_size]

    reach_counts = {team: dict.fromkeys(round_sizes, 0) for team in teams}
    for _ in range(n_sims):
        furthest, _champion = simulate_tournament(groups, prob_fn, rng, ratings, hosts)
        for team, reached_size in furthest.items():
            for size in round_sizes:
                if size >= reached_size:
                    reach_counts[team][size] += 1

    results: list[dict[str, Any]] = []
    for team in teams:
        record: dict[str, Any] = {"team": team}
        for size in round_sizes:
            record[STAGE_FIELDS[size]] = reach_counts[team][size] / n_sims
        results.append(record)
    return results
