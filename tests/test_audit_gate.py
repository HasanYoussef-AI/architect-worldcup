"""Verification gate: leakage invariant and probability validity.

Hermetic, on synthetic data. These gates close two reviewer concerns
permanently. The first is the real leakage invariant, stronger than a literal
duplicate count: no match, identified as a (date, home_team, away_team) triple,
appears in both the training set and the holdout set of the same walk-forward
window. The second is probability validity: every vector the model emits is a
valid distribution. Both run for every future change, including the ensemble.

The duplicate detector itself stays a reported diagnostic in wc-audit, not a
gate, because the loaded data carries two characterized, benign duplicates that
cannot leak. Its detection logic is unit-tested here so the diagnostic stays
trustworthy.
"""

from __future__ import annotations

import pandas as pd

from architect_wc import audit, calibrate, ingest, ratings

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

_BLOCK = [
    ("A", "B", 2, 0),
    ("A", "C", 3, 1),
    ("A", "D", 2, 0),
    ("B", "A", 1, 2),
    ("B", "C", 2, 0),
    ("B", "D", 3, 1),
    ("C", "A", 1, 3),
    ("C", "B", 0, 2),
    ("C", "D", 2, 0),
    ("D", "A", 0, 2),
    ("D", "B", 1, 3),
    ("D", "C", 0, 2),
]


def _matches(repeats: int = 12) -> pd.DataFrame:
    fixtures = _BLOCK * repeats
    dates = pd.date_range("2024-01-01", periods=len(fixtures), freq="D")
    rows = [
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
        for (home, away, home_score, away_score), date in zip(
            fixtures, dates, strict=True
        )
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def _matches_with_duplicate() -> pd.DataFrame:
    # Inject an exact duplicate of a recent fixture to stress the leakage
    # invariant: a duplicate shares a date, so it must stay on one side of every
    # window boundary, exactly the situation the real data presents.
    df = _matches()
    duplicate = df.iloc[[-5]].copy()
    return pd.concat([df, duplicate], ignore_index=True)


CONFIG = {
    "as_of_date": "2025-06-01",
    "random_seed": 42,
    "dixon_coles": {"xi": 0.0018},
    "calibrate": {"backtest_size": 12},
}
WALK_CONFIG = {**CONFIG, "calibrate": {"backtest_size": 12, "walk_windows": 3}}


def _triples(df: pd.DataFrame) -> set[tuple[str, str, str]]:
    return {
        (pd.Timestamp(date).date().isoformat(), home, away)
        for date, home, away in zip(
            df["date"], df["home_team"], df["away_team"], strict=True
        )
    }


def test_no_match_in_train_and_holdout_of_same_window() -> None:
    matches = _matches_with_duplicate()
    windows = list(calibrate.walk_forward_windows(matches, WALK_CONFIG))
    # Tested on more than one window, not just the anchor.
    assert len(windows) >= 2
    for window in windows:
        train_triples = _triples(window.training)
        holdout_triples = {
            (match.date, match.home_team, match.away_team) for match in window.matches
        }
        # The real leakage concern: no match trains and is also scored against in
        # the same window. A duplicate cannot break this because it shares a date
        # and the split is purely by date.
        assert train_triples.isdisjoint(holdout_triples)


def test_duplicate_matches_detects_a_duplicate() -> None:
    df = _matches(repeats=2)
    duplicate = df.iloc[[0]].copy()
    df_with_dup = pd.concat([df, duplicate], ignore_index=True)
    found = ingest.duplicate_matches(df_with_dup)
    assert len(found) == 2
    assert set(found["home_team"]) == {df.iloc[0]["home_team"]}


def test_duplicate_matches_none_when_unique() -> None:
    assert len(ingest.duplicate_matches(_matches())) == 0


def test_valid_probability_vector_accepts_a_distribution() -> None:
    assert audit.valid_probability_vector([0.5, 0.3, 0.2])
    assert audit.valid_probability_vector([1.0, 0.0, 0.0])


def test_valid_probability_vector_rejects_bad_vectors() -> None:
    assert not audit.valid_probability_vector([-0.01, 0.5, 0.51])
    assert not audit.valid_probability_vector([1.2, 0.0, 0.0])
    assert not audit.valid_probability_vector([0.5, 0.3, 0.1])


def test_dixon_coles_emits_valid_probabilities() -> None:
    window = calibrate.prepare_backtest(_matches(), CONFIG)
    prob_vectors, _outcomes = calibrate.window_predictions(window, CONFIG)
    assert prob_vectors
    for vector in prob_vectors:
        assert audit.valid_probability_vector(vector)


def test_elo_generator_emits_valid_probabilities() -> None:
    # Sweep rating gaps, venues, and tie parameters covering the Elo config path.
    for diff in (-600.0, -150.0, 0.0, 150.0, 600.0):
        for advantage in (0.0, 65.0):
            for nu in (0.0, 0.5, 2.0):
                vector = list(
                    ratings.elo_win_draw_loss(1500.0 + diff, 1500.0, advantage, nu)
                )
                assert audit.valid_probability_vector(vector)
