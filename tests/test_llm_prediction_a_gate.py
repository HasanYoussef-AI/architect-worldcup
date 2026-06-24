"""Verification gate: the analytic Prediction A math and the smoke driver.

Hermetic, no network. Proves the advance-probability formula, that tie_record
derives the shootout lean and advance from the shared simulate.shootout_lean,
that the shootout lean is exactly the neutral Elo expectation, that the frozen
Prediction A fixture is internally consistent, the frozen weights load and
aggregate, and that the full offline smoke driver reports all checks passing.
"""

from __future__ import annotations

from architect_wc import pipeline, ratings, simulate
from architect_wc.llm import prediction_a, smoke, weights


def test_advance_probability_formula() -> None:
    assert prediction_a.advance_probability(0.5, 0.3, 0.6) == 0.5 + 0.3 * 0.6


def test_tie_record_uses_shared_shootout_lean() -> None:
    three_way = {"p_home_win": 0.5, "p_draw": 0.3, "p_away_win": 0.2}
    record = prediction_a.tie_record(1, "A", "B", three_way, 1600.0, 1500.0)
    lean = simulate.shootout_lean(1600.0, 1500.0)
    assert record["shootout_lean"] == lean
    assert abs(record["p_advance_home"] - (0.5 + 0.3 * lean)) <= 1e-12


def test_shootout_lean_is_the_neutral_elo_expectation() -> None:
    # The shootout lean is exactly the Elo expected score at a neutral venue, so A
    # and the bracket simulation resolve a drawn tie by the identical math.
    assert simulate.shootout_lean(1600.0, 1500.0) == ratings.expected_score(
        1600.0, 1500.0, 0.0
    )


def test_prediction_a_fixture_is_internally_consistent() -> None:
    for tie in smoke.FIXTURE_A["ties"]:
        expected = tie["p_home_win_90"] + tie["p_draw_90"] * tie["shootout_lean"]
        assert abs(tie["p_advance_home"] - expected) <= 1e-9


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
