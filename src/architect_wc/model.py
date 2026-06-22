"""Layer 3, Dixon-Coles.

Owns the goal model: a Dixon-Coles adjusted Poisson fit on the leakage-guarded
matches, producing scoreline probabilities. This is the part worth delegating,
so it uses penaltyblog (DixonColesGoalModel) rather than a hand-rolled model.
Recent matches carry more weight through Dixon-Coles time decay.

Targets the installed penaltyblog 1.11.0 API.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.models import DixonColesGoalModel, dixon_coles_weights

DEFAULT_XI = 0.0018


def _neutral_flags(neutral: pd.Series) -> pd.Series:
    """Coerce the neutral column to 0/1 flags across bool and text encodings."""
    text = neutral.astype(str).str.strip().str.lower()
    return text.isin(["true", "1"]).astype(int)


def fit_model(matches: pd.DataFrame, config: dict[str, Any]) -> DixonColesGoalModel:
    """Fit penaltyblog's Dixon-Coles on the guarded matches.

    Matches are weighted by Dixon-Coles time decay derived from each match
    date, so recent games carry more weight. The decay rate xi is read from the
    dixon_coles config block. Returns the fitted model.
    """
    dixon_coles_config = config.get("dixon_coles", {}) or {}
    xi = float(dixon_coles_config.get("xi", DEFAULT_XI))

    played = matches.dropna(subset=["home_score", "away_score"])

    as_of_date = config.get("as_of_date")
    base_date = pd.Timestamp(as_of_date) if as_of_date else None
    weights = dixon_coles_weights(played["date"], xi=xi, base_date=base_date)

    # penaltyblog's Cython loss needs writable, contiguous buffers, so pass
    # fresh copies rather than the possibly read-only views the filtered frame
    # returns.
    model = DixonColesGoalModel(
        goals_home=np.array(played["home_score"], dtype=np.int64),
        goals_away=np.array(played["away_score"], dtype=np.int64),
        teams_home=played["home_team"].to_numpy(),
        teams_away=played["away_team"].to_numpy(),
        weights=np.array(weights, dtype=np.float64),
        neutral_venue=np.array(_neutral_flags(played["neutral"]), dtype=np.int64),
    )
    model.fit()
    return model


def match_probabilities(
    model: DixonColesGoalModel, home_team: str, away_team: str
) -> dict[str, float]:
    """Return outcome probabilities for a fixture from the scoreline grid.

    The dict has p_home_win, p_draw, and p_away_win, each a valid probability
    derived from the model's scoreline probability grid.
    """
    grid = model.predict(home_team, away_team)
    return {
        "p_home_win": float(grid.home_win),
        "p_draw": float(grid.draw),
        "p_away_win": float(grid.away_win),
    }
