"""Verification gate: the RPS scorer.

Hermetic test, no network, small. This gate proves the scorer itself is correct,
which has to be trustworthy before any number it produces means anything. It
checks that a perfect confident prediction scores zero, that a worse prediction
scores higher than a better one on the same outcome, that the score respects the
ordering of the three outcomes, that the score stays within its valid bounds,
that a confident correct prediction beats a uniform guess, and that scoring is
deterministic. It also checks the base-rate baseline helper.
"""

from __future__ import annotations

import pandas as pd

from architect_wc import calibrate

# Ordered probabilities are (home win, draw, away win); outcome index matches.
PERFECT_HOME = (1.0, 0.0, 0.0)
UNIFORM = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
TOLERANCE = 1e-9


def test_perfect_confident_prediction_scores_zero() -> None:
    assert calibrate.rps(PERFECT_HOME, calibrate.HOME_WIN) == 0.0


def test_worse_prediction_scores_higher_on_same_outcome() -> None:
    # Both predict a home win that happens; the less confident one scores worse.
    better = calibrate.rps((0.7, 0.2, 0.1), calibrate.HOME_WIN)
    worse = calibrate.rps((0.4, 0.3, 0.3), calibrate.HOME_WIN)
    assert worse > better


def test_score_respects_outcome_ordering() -> None:
    # A home win happens. Putting the wrong mass on the adjacent draw is less bad
    # than putting it on the far away win, which only an ordered score captures.
    on_draw = calibrate.rps((0.1, 0.8, 0.1), calibrate.HOME_WIN)
    on_away = calibrate.rps((0.1, 0.1, 0.8), calibrate.HOME_WIN)
    assert on_away > on_draw


def test_score_stays_within_bounds() -> None:
    cases = [
        ((1.0, 0.0, 0.0), calibrate.HOME_WIN),
        ((0.0, 0.0, 1.0), calibrate.HOME_WIN),
        ((0.2, 0.5, 0.3), calibrate.DRAW),
        (UNIFORM, calibrate.AWAY_WIN),
    ]
    for probabilities, outcome in cases:
        score = calibrate.rps(probabilities, outcome)
        assert 0.0 <= score <= 1.0


def test_confident_correct_beats_uniform_guess() -> None:
    confident = calibrate.rps((0.9, 0.05, 0.05), calibrate.HOME_WIN)
    uniform = calibrate.rps(UNIFORM, calibrate.HOME_WIN)
    assert confident < uniform


def test_scoring_is_deterministic() -> None:
    first = calibrate.rps((0.55, 0.25, 0.20), calibrate.DRAW)
    second = calibrate.rps((0.55, 0.25, 0.20), calibrate.DRAW)
    assert first == second


def test_mean_rps_matches_single_scores() -> None:
    probabilities = [[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]]
    outcomes = [calibrate.HOME_WIN, calibrate.AWAY_WIN]
    expected = (
        calibrate.rps(tuple(probabilities[0]), outcomes[0])
        + calibrate.rps(tuple(probabilities[1]), outcomes[1])
    ) / 2.0
    assert abs(calibrate.mean_rps(probabilities, outcomes) - expected) <= TOLERANCE


def test_outcome_index_reads_results() -> None:
    assert calibrate.outcome_index(3, 1) == calibrate.HOME_WIN
    assert calibrate.outcome_index(1, 1) == calibrate.DRAW
    assert calibrate.outcome_index(0, 2) == calibrate.AWAY_WIN


def test_base_rates_are_valid_frequencies() -> None:
    matches = pd.DataFrame(
        {
            "home_score": [2, 1, 0, 3],
            "away_score": [0, 1, 1, 3],
        }
    )
    home, draw, away = calibrate.base_rates(matches)
    assert (home, draw, away) == (0.25, 0.5, 0.25)
    assert abs((home + draw + away) - 1.0) <= TOLERANCE
