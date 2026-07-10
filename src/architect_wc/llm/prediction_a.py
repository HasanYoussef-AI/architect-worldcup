"""Prediction A, the math model per-match call (per-match shape).

A reads the 90-minute three-way and advance probability straight from the existing
math model: the Dixon-Coles analytic three-way at a neutral venue plus the Elo
shootout lean. It needs no Monte Carlo, since for a single fixture with both teams
known the three-way is the exact marginal of the same goal grid the simulator
samples, with zero sampling noise, and the shootout lean is the exact value the
simulator uses to resolve a drawn tie (simulate.shootout_lean_from_elo).

A is analytic and has no prose reasoning, so a small drivers explainer travels in
A's own output: A's three-way, A's lean, and the strength differentials that
produced them, the squad-adjusted Elo difference, the Dixon-Coles expected-goals
difference, and the squad-value adjustment. This lets Prediction C understand A's
number rather than reading a bare probability. The explainer lives only in A's
output, so B stays blind to A.

The pure assembly (advance_probability, expected_goals_from_grid, a_core) is
separated from the data work (build_prediction_a) so the math is unit-testable with
no model fit and no network. A is per-match, the same shape Predictions B and C use.
"""

from __future__ import annotations

from typing import Any

from architect_wc import simulate

SOURCE = "dixon_coles_three_way+elo_shootout"
SCHEMA_VERSION = "1.0.0"


def advance_probability(
    p_home_win_90: float, p_draw_90: float, shootout_lean: float
) -> float:
    """Probability the nominal home side advances: win in 90, or draw then shootout.

    p_advance_home = p_home_win_90 + p_draw_90 * shootout_lean. Pure function.
    """
    return p_home_win_90 + p_draw_90 * shootout_lean


def expected_goals_from_grid(matrix: Any) -> tuple[float, float]:
    """Home and away expected goals from a scoreline probability grid. Pure.

    The grid entry (i, j) is the probability of i home goals and j away goals, so
    the home expectation sums i weighted by each row's mass and the away
    expectation sums j weighted by each column's mass.
    """
    import numpy as np

    grid = np.asarray(matrix, dtype=float)
    rows = np.arange(grid.shape[0])
    cols = np.arange(grid.shape[1])
    home_xg = float((rows * grid.sum(axis=1)).sum())
    away_xg = float((cols * grid.sum(axis=0)).sum())
    return home_xg, away_xg


def a_core(
    three_way: dict[str, float],
    rating_home: float,
    rating_away: float,
    home_xg: float,
    away_xg: float,
    squad_adjustment_home: float,
    squad_adjustment_away: float,
) -> dict[str, Any]:
    """Assemble A's three-way, shootout lean, advance, and drivers. Pure, no I/O.

    The shootout lean and the elo difference come from the squad-adjusted Elo, the
    identical ratings the bracket simulator uses, so A stays consistent with the
    bracket. The drivers name the strength differentials behind A's number.
    """
    lean = simulate.shootout_lean_from_elo(rating_home, rating_away)
    p_home = float(three_way["p_home_win"])
    p_draw = float(three_way["p_draw"])
    p_away = float(three_way["p_away_win"])
    return {
        "three_way": {"p_home": p_home, "p_draw": p_draw, "p_away": p_away},
        "shootout_lean": lean,
        "advance_probability": advance_probability(p_home, p_draw, lean),
        "drivers": {
            "elo_diff": rating_home - rating_away,
            "ability_diff": home_xg - away_xg,
            "squad_value_adjustment": squad_adjustment_home - squad_adjustment_away,
        },
    }


def build_prediction_a(
    config: dict[str, Any],
    fixture: dict[str, Any],
    as_of_date: str,
    round_code: str,
) -> dict[str, Any]:
    """Build the full per-match Prediction A document for one fixture.

    Fits the existing pipeline at the cutoff (the same chain as the forecast:
    leakage-guarded matches, Elo, squad adjustment, Dixon-Coles), asserts no leakage
    at the cutoff boundary, reads the three-way and the shootout lean, and builds the
    drivers explainer from the squad-adjusted Elo difference, the Dixon-Coles
    expected-goals difference, and the squad-value adjustment. No Anthropic call: A
    is pure math. Returns the in-memory dict; the artifact layer writes it.
    """
    import pandas as pd

    from architect_wc import artifact, ingest, model, ratings, squad

    fit_config = dict(config)
    fit_config["as_of_date"] = as_of_date

    matches, _provenance = ingest.load_matches(fit_config)
    as_of = pd.Timestamp(as_of_date)
    boundary = as_of + pd.Timedelta(days=1)
    training_max = pd.to_datetime(matches["date"]).max()
    if training_max >= boundary:
        raise ValueError(
            f"Leakage at Prediction A cutoff: training max {training_max.date()} "
            f"is not before the boundary {boundary.date()}."
        )

    team_ratings = ratings.compute_elo(matches, fit_config)
    raw_elo = dict(team_ratings)
    squad_values = squad.load_squad_values(fit_config["squad"]["snapshot"])
    adjusted = dict(squad.adjust_ratings(team_ratings, squad_values, fit_config))
    goal_model = model.fit_model(matches, fit_config)

    home = fixture["home_team"]
    away = fixture["away_team"]
    three_way = model.match_probabilities(goal_model, home, away, neutral=True)
    grid = goal_model.predict(home, away, neutral_venue=True)
    home_xg, away_xg = expected_goals_from_grid(grid.goal_matrix)

    core = a_core(
        three_way,
        adjusted.get(home, 0.0),
        adjusted.get(away, 0.0),
        home_xg,
        away_xg,
        adjusted.get(home, 0.0) - raw_elo.get(home, 0.0),
        adjusted.get(away, 0.0) - raw_elo.get(away, 0.0),
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "prediction": "A",
        "round": round_code,
        "match": int(fixture["match"]),
        "home_team": home,
        "away_team": away,
        "nominal_home": home,
        "cutoff": as_of_date,
        "committed_at": None,
        "seed": int(config.get("random_seed", 42)),
        "provenance": {
            "source": SOURCE,
            "model_version": artifact.MODEL_VERSION,
            "code_commit": artifact.get_git_sha(),
            "timestamp": None,
        },
        **core,
    }
