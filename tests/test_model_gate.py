"""Verification gate: Dixon-Coles goal model.

Hermetic test, no network, on a small synthetic league where one team clearly
outscores another across several matches. Fits penaltyblog's Dixon-Coles and
checks that the dominant team is favoured, that outcome probabilities are valid
and sum to 1, and that fitting twice gives the same probabilities.
"""

from __future__ import annotations

import pandas as pd

from architect_wc import model

CONFIG = {"as_of_date": "2020-12-31", "dixon_coles": {"xi": 0.0018}}
SUM_TOLERANCE = 1e-6

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

# One block of a small league. Strong outscores everyone, Weak is outscored.
_BLOCK = [
    ("Strong", "Weak", 4, 0),
    ("Weak", "Strong", 0, 3),
    ("Strong", "Mid", 3, 1),
    ("Mid", "Strong", 1, 3),
    ("Mid", "Weak", 2, 0),
    ("Weak", "Mid", 0, 2),
]


def _league(repeats: int = 5) -> pd.DataFrame:
    rows = []
    fixtures = _BLOCK * repeats
    dates = pd.date_range("2019-01-01", periods=len(fixtures), freq="D")
    for (home, away, home_score, away_score), date in zip(fixtures, dates, strict=True):
        rows.append(
            {
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": "League",
                "city": "Citytown",
                "country": "Countryland",
                "neutral": False,
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def test_dominant_team_is_favoured() -> None:
    fitted = model.fit_model(_league(), CONFIG)
    probs = model.match_probabilities(fitted, "Strong", "Weak")
    assert probs["p_home_win"] > probs["p_away_win"]


def test_outcome_probabilities_are_valid() -> None:
    fitted = model.fit_model(_league(), CONFIG)
    for home, away in [("Strong", "Weak"), ("Mid", "Strong"), ("Weak", "Mid")]:
        probs = model.match_probabilities(fitted, home, away)
        total = probs["p_home_win"] + probs["p_draw"] + probs["p_away_win"]
        assert abs(total - 1.0) <= SUM_TOLERANCE
        for value in probs.values():
            assert 0.0 <= value <= 1.0


def test_fit_is_deterministic() -> None:
    first = model.match_probabilities(
        model.fit_model(_league(), CONFIG), "Strong", "Weak"
    )
    second = model.match_probabilities(
        model.fit_model(_league(), CONFIG), "Strong", "Weak"
    )
    for key in first:
        assert abs(first[key] - second[key]) <= 1e-9
