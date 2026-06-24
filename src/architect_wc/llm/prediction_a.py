"""Prediction A, the math model knockout call.

A reads the per-tie 90-minute three-way and advance probability straight from the
existing math model: the Dixon-Coles analytic three-way at a neutral venue plus
the Elo shootout lean. It needs no Monte Carlo, since for a single tie with both
teams known the three-way is the exact marginal of the same goal grid the
simulator samples, with zero sampling noise, and the shootout lean is the exact
value the simulator uses to resolve a drawn tie (simulate.shootout_lean).

A is bit-reproducible from a fixed seed and dated data, and it is frozen: B and C
never edit it.

The pure assembly (advance_probability, tie_record) is separated from the data
work (build_prediction_a) so the math is unit-testable with no model fit and no
network.
"""

from __future__ import annotations

from typing import Any

from architect_wc import simulate

SOURCE = "dixon_coles_three_way+elo_shootout"


def advance_probability(
    p_home_win_90: float, p_draw_90: float, shootout_lean: float
) -> float:
    """Probability the nominal home side advances: win in 90, or draw then shootout.

    p_advance_home = p_home_win_90 + p_draw_90 * shootout_lean. Pure function.
    """
    return p_home_win_90 + p_draw_90 * shootout_lean


def tie_record(
    match: int,
    home_team: str,
    away_team: str,
    three_way: dict[str, float],
    rating_home: float,
    rating_away: float,
) -> dict[str, Any]:
    """Build one Prediction A tie record from a three-way and the two ratings.

    three_way carries p_home_win, p_draw, p_away_win for the nominal home side at a
    neutral venue, exactly model.match_probabilities(..., neutral=True). The
    shootout lean and advance probability are derived from the same ratings the
    simulator uses, so A is internally consistent with the bracket simulation.
    Pure function with no I/O.
    """
    lean = simulate.shootout_lean(rating_home, rating_away)
    p_home_win_90 = float(three_way["p_home_win"])
    p_draw_90 = float(three_way["p_draw"])
    p_away_win_90 = float(three_way["p_away_win"])
    return {
        "match": int(match),
        "home_team": home_team,
        "away_team": away_team,
        "p_home_win_90": p_home_win_90,
        "p_draw_90": p_draw_90,
        "p_away_win_90": p_away_win_90,
        "shootout_lean": lean,
        "p_advance_home": advance_probability(p_home_win_90, p_draw_90, lean),
        "source": SOURCE,
    }


def build_prediction_a(
    config: dict[str, Any],
    round_code: str,
    as_of_date: str,
    fixtures: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full Prediction A document for a round from the math layers.

    Fits the existing pipeline at the round cutoff (the same chain as the forecast:
    leakage-guarded matches, Elo, squad adjustment, Dixon-Coles), asserts no
    leakage at the cutoff boundary, and reads the per-tie three-way and shootout
    lean for each real fixture. No Anthropic call: A is pure math. The heavy
    document is written by the artifact layer; this returns the in-memory dict.
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

    # adjusted is the squad-adjusted Elo rating per team, the identical object the
    # bracket simulator passes as context["ratings"]; the shootout lean is computed
    # from it, while the three-way is the Dixon-Coles analytic marginal, exactly the
    # two-model pairing the simulator uses, so A matches the bracket.
    team_ratings = ratings.compute_elo(matches, fit_config)
    squad_values = squad.load_squad_values(fit_config["squad"]["snapshot"])
    adjusted = dict(squad.adjust_ratings(team_ratings, squad_values, fit_config))
    goal_model = model.fit_model(matches, fit_config)

    ties = []
    for fixture in fixtures:
        home = fixture["home_team"]
        away = fixture["away_team"]
        three_way = model.match_probabilities(goal_model, home, away, neutral=True)
        ties.append(
            tie_record(
                fixture["match"],
                home,
                away,
                three_way,
                adjusted.get(home, 0.0),
                adjusted.get(away, 0.0),
            )
        )

    return {
        "prediction": "A",
        "round": round_code,
        "as_of_date": as_of_date,
        "committed_at": None,
        "git_sha": artifact.get_git_sha(),
        "model_version": artifact.MODEL_VERSION,
        "seed": int(config.get("random_seed", 42)),
        "ties": ties,
    }
