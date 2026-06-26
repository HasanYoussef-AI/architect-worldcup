"""Verification gate: per-match Prediction A math, the drivers explainer, and the
offline smoke.

Hermetic, no network, no model fit. Proves the advance-probability formula, that
a_core derives the shootout lean and advance from the shared
simulate.shootout_lean_from_elo and names the strength differentials, that the
shootout lean equals the neutral Elo expectation, that a per-match A document
validates against its schema and is internally consistent, that the expected-goals
helper is correct, that the frozen weights load and aggregate, and that the offline
smoke driver reports all checks passing.

Migrated from the per-round-with-ties shape to per-match: test_tie_record_uses... and
test_prediction_a_fixture_is_internally_consistent became test_a_core_uses... and
test_per_match_a_document_validates_and_is_consistent, with the same assertions on
the shootout lean and the advance formula but on a single per-match record.
"""

from __future__ import annotations

import numpy as np

from architect_wc import pipeline, ratings, simulate
from architect_wc.llm import prediction_a, schemas, smoke, weights


def test_advance_probability_formula() -> None:
    assert prediction_a.advance_probability(0.5, 0.3, 0.6) == 0.5 + 0.3 * 0.6


def test_shootout_lean_is_the_neutral_elo_expectation() -> None:
    # The shootout lean is exactly the Elo expected score at a neutral venue, so A
    # and the bracket simulation resolve a drawn tie by the identical math.
    assert simulate.shootout_lean_from_elo(1600.0, 1500.0) == ratings.expected_score(
        1600.0, 1500.0, 0.0
    )


def test_a_core_uses_the_shared_elo_shootout_lean_and_names_drivers() -> None:
    three_way = {"p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2}
    core = prediction_a.a_core(three_way, 1600.0, 1500.0, 1.6, 1.1, 80.0, -5.0)
    lean = simulate.shootout_lean_from_elo(1600.0, 1500.0)
    assert core["shootout_lean"] == lean
    assert abs(core["advance_probability"] - (0.5 + 0.3 * lean)) <= 1e-12
    assert core["three_way"] == {"p_home": 0.5, "p_draw": 0.3, "p_away": 0.2}
    assert core["drivers"]["elo_diff"] == 100.0
    assert abs(core["drivers"]["ability_diff"] - (1.6 - 1.1)) <= 1e-12
    assert core["drivers"]["squad_value_adjustment"] == 85.0


def test_expected_goals_from_grid() -> None:
    grid = np.array([[0.25, 0.25], [0.25, 0.25]])
    home_xg, away_xg = prediction_a.expected_goals_from_grid(grid)
    assert abs(home_xg - 0.5) <= 1e-12
    assert abs(away_xg - 0.5) <= 1e-12
    # A grid concentrated on (2, 0) gives home 2, away 0.
    skew = np.zeros((3, 3))
    skew[2, 0] = 1.0
    home_xg, away_xg = prediction_a.expected_goals_from_grid(skew)
    assert home_xg == 2.0 and away_xg == 0.0


def test_per_match_a_document_validates_and_is_consistent() -> None:
    core = prediction_a.a_core(
        {"p_home_win": 0.5, "p_draw": 0.28, "p_away_win": 0.22},
        1600.0,
        1520.0,
        1.7,
        1.3,
        60.0,
        -10.0,
    )
    document = {
        "schema_version": "1.0.0",
        "prediction": "A",
        "round": "GROUP",
        "match": 64,
        "home_team": "Uruguay",
        "away_team": "Spain",
        "nominal_home": "Uruguay",
        "cutoff": "2026-06-25",
        "committed_at": None,
        "seed": 42,
        "provenance": {
            "source": prediction_a.SOURCE,
            "model_version": "0.0.0",
            "code_commit": "test",
            "timestamp": None,
        },
        **core,
    }
    schemas.validate_document(document, "prediction_a")
    tw = document["three_way"]
    expected = tw["p_home"] + tw["p_draw"] * document["shootout_lean"]
    assert abs(document["advance_probability"] - expected) <= 1e-12


def test_frozen_weights_load_and_sum_to_one() -> None:
    config = pipeline.load_config()
    loaded = weights.load_weights(config)
    assert set(loaded) == set(weights.FACTORS)
    assert abs(sum(loaded.values()) - 1.0) <= weights.WEIGHT_SUM_TOLERANCE


def test_aggregate_is_the_fixed_weighted_sum() -> None:
    config = pipeline.load_config()
    loaded = weights.load_weights(config)
    scores = dict.fromkeys(weights.FACTORS, 0.0)
    scores["squad_availability"] = 1.0
    anchor = weights.aggregate(scores, loaded)
    assert abs(anchor - loaded["squad_availability"]) <= 1e-12


def test_offline_smoke_reports_all_passed() -> None:
    report = smoke.run_smoke()
    failed = [c["name"] for c in report["checks"] if not c["passed"]]
    assert report["all_passed"], f"smoke checks failed: {failed}"
