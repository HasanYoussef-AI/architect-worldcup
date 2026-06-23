"""Verification gate: the walk-forward backtest, structural only.

Hermetic test, no network, on a small synthetic league. The walk-forward harness
is the measurement instrument every later change is judged against, so these
gates prove it is trustworthy without gating the empirical spread or ordering.
They check that the most recent window reproduces the single-window backtest
exactly, that the harness is deterministic, that the leakage guard fires for
every window and not just the anchor, and that the holdout windows do not share
matches, since the independence of the spread depends on it.
"""

from __future__ import annotations

import pandas as pd

from architect_wc import calibrate

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

# Draw-free, clear strength gradient, same shape as the ablation gate league, so
# the Dixon-Coles fit stays stable on this tiny synthetic set.
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


def _matches(repeats: int = 16) -> pd.DataFrame:
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


# Small windows and a few of them, so several non-overlapping windows fit on the
# synthetic data while every window still has training behind it.
CONFIG = {
    "as_of_date": "2025-06-01",
    "random_seed": 42,
    "dixon_coles": {"xi": 0.0018},
    "calibrate": {"backtest_size": 12, "walk_windows": 3},
}


def test_recent_window_reproduces_single_window_backtest() -> None:
    matches = _matches()
    single = calibrate.run_backtest(matches, CONFIG)
    walk = calibrate.run_walk_forward(matches, CONFIG)
    # The most recent walk-forward window must score exactly as the single-window
    # backtest does, the anchor that proves the instrument measures the same way.
    assert abs(walk["windows"][0]["model_mean_rps"] - single["model_mean_rps"]) <= 1e-12


def test_walk_forward_is_deterministic() -> None:
    matches = _matches()
    first = calibrate.run_walk_forward(matches, CONFIG)
    second = calibrate.run_walk_forward(matches, CONFIG)
    assert first == second


def test_leakage_guard_fires_for_every_window() -> None:
    matches = _matches()
    walk = calibrate.run_walk_forward(matches, CONFIG)
    # Tested on more than one window, not just the anchor.
    assert len(walk["windows"]) >= 2
    for w in walk["windows"]:
        train_max = pd.Timestamp(w["train_max_date"])
        holdout_start = pd.Timestamp(w["holdout_start"])
        assert train_max < holdout_start


def test_holdout_windows_do_not_share_matches() -> None:
    matches = _matches()
    walk = calibrate.run_walk_forward(matches, CONFIG)
    identifier_sets = [
        {(m["date"], m["home_team"], m["away_team"]) for m in w["holdout_matches"]}
        for w in walk["windows"]
    ]
    for i in range(len(identifier_sets)):
        for j in range(i + 1, len(identifier_sets)):
            assert identifier_sets[i].isdisjoint(identifier_sets[j])
