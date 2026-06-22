"""Layer 2, Elo ratings.

Owns team strength ratings computed from match history with a transparent,
hand-written Elo implementation. The pure scoring logic is kept separate from
any I/O so it is fully unit-testable. Elo is simple enough to own outright;
penaltyblog is reserved for the Dixon-Coles goal model in the next phase, which
is the part worth delegating.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

BASE_RATING = 1500.0
K_FACTOR = 32.0
HOME_ADVANTAGE = 65.0


def expected_score(
    rating_for: float, rating_against: float, home_advantage: float
) -> float:
    """Standard Elo logistic expectation for the rating_for side.

    home_advantage is added to the rating_for side's rating. The result is the
    expected score in the open interval (0, 1). Pure function with no I/O.
    """
    exponent = (rating_against - (rating_for + home_advantage)) / 400.0
    return 1.0 / (1.0 + 10.0**exponent)


def _is_neutral(value: Any) -> bool:
    """Interpret the neutral column robustly across bool and text encodings."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def compute_elo(
    matches: pd.DataFrame, config: dict[str, Any]
) -> list[tuple[str, float]]:
    """Compute Elo ratings from matches, processed chronologically by date.

    Unseen teams start at base_rating. For each match the actual score is 1 for
    a win, 0.5 for a draw, and 0 for a loss, and both ratings move by k_factor
    times actual minus expected. The home advantage is applied to the home team
    only when the match is not at a neutral venue. The update is zero-sum: the
    winner gains exactly what the loser loses. Returns the final ratings as a
    list of (team, rating) tuples sorted from highest to lowest.

    Goal-difference weighting and tournament-importance weighting are left out
    of this phase on purpose. They are candidate enhancements and good ablation
    factors for later.
    """
    elo_config = config.get("elo", {}) or {}
    base_rating = float(elo_config.get("base_rating", BASE_RATING))
    k_factor = float(elo_config.get("k_factor", K_FACTOR))
    home_advantage = float(elo_config.get("home_advantage", HOME_ADVANTAGE))

    ratings: dict[str, float] = {}
    ordered = matches.sort_values("date", kind="stable")

    for row in ordered.itertuples(index=False):
        home_score = row.home_score
        away_score = row.away_score
        if pd.isna(home_score) or pd.isna(away_score):
            continue

        home = row.home_team
        away = row.away_team
        rating_home = ratings.setdefault(home, base_rating)
        rating_away = ratings.setdefault(away, base_rating)

        advantage = 0.0 if _is_neutral(row.neutral) else home_advantage
        e_home = expected_score(rating_home, rating_away, advantage)

        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        # The away side faces the home advantage, so its expected score is
        # 1 - e_home and its update is the exact negative of the home update.
        # Applying delta and -delta keeps the match zero-sum.
        delta = k_factor * (actual_home - e_home)
        ratings[home] = rating_home + delta
        ratings[away] = rating_away - delta

    return sorted(ratings.items(), key=lambda item: item[1], reverse=True)
