"""Verification gate: Elo ratings sanity.

Hermetic test, no network, on a small synthetic set of matches. Checks the
core Elo properties: a calibrated expectation, a bounded expectation, monotonic
response to a win and a loss, zero-sum updates, all-wins beating all-losses, and
determinism.
"""

from __future__ import annotations

import pandas as pd

from architect_wc import ratings

CONFIG = {"elo": {"base_rating": 1500, "k_factor": 32, "home_advantage": 65}}
BASE = 1500.0

COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def _match(date, home, away, home_score, away_score, neutral=False) -> dict:
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_score": home_score,
        "away_score": away_score,
        "tournament": "Friendly",
        "city": "Citytown",
        "country": "Countryland",
        "neutral": neutral,
    }


def _matches(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_expected_score_equal_ratings_no_home_advantage() -> None:
    assert ratings.expected_score(1500, 1500, 0) == 0.5


def test_expected_score_within_unit_interval() -> None:
    cases = [
        (1500, 1500, 0),
        (1500, 1500, 65),
        (2200, 900, 65),
        (900, 2200, 0),
    ]
    for rating_for, rating_against, home_advantage in cases:
        value = ratings.expected_score(rating_for, rating_against, home_advantage)
        assert 0.0 < value < 1.0


def test_winner_rises_and_loser_falls() -> None:
    df = _matches([_match("2020-01-01", "Winner", "Loser", 3, 0)])
    result = dict(ratings.compute_elo(df, CONFIG))
    assert result["Winner"] > BASE
    assert result["Loser"] < BASE


def test_single_update_is_zero_sum() -> None:
    df = _matches([_match("2020-01-01", "Winner", "Loser", 2, 1)])
    result = dict(ratings.compute_elo(df, CONFIG))
    gain = result["Winner"] - BASE
    loss = result["Loser"] - BASE
    # The winner gains exactly what the loser loses, so the net change is zero.
    assert gain == -loss
    assert gain + loss == 0.0


def test_all_wins_beats_all_losses() -> None:
    df = _matches(
        [
            _match("2020-01-01", "A", "B", 2, 0),
            _match("2020-01-02", "A", "C", 1, 0),
            _match("2020-01-03", "A", "D", 3, 1),
            _match("2020-01-04", "B", "D", 2, 0),
            _match("2020-01-05", "C", "D", 1, 0),
        ]
    )
    result = dict(ratings.compute_elo(df, CONFIG))
    # A wins every match it plays; D loses every match it plays.
    assert result["A"] > result["D"]


def test_compute_elo_is_deterministic() -> None:
    df = _matches(
        [
            _match("2020-01-01", "A", "B", 2, 0),
            _match("2020-01-02", "B", "C", 1, 1),
            _match("2020-01-03", "C", "A", 0, 2),
        ]
    )
    first = ratings.compute_elo(df, CONFIG)
    second = ratings.compute_elo(df, CONFIG)
    assert first == second
