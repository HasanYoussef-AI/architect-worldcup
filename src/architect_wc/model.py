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
DEFAULT_FRIENDLY_WEIGHT = 0.5


def _neutral_flags(neutral: pd.Series) -> pd.Series:
    """Coerce the neutral column to 0/1 flags across bool and text encodings."""
    text = neutral.astype(str).str.strip().str.lower()
    return text.isin(["true", "1"]).astype(int)


def importance_weights(tournaments: pd.Series, config: dict[str, Any]) -> np.ndarray:
    """Per-match importance multiplier for the Dixon-Coles weights.

    Two tiers by a clear rule. Friendlies are the one unambiguous low-stakes,
    experimental-lineup category, about 37 percent of the data, so they are
    downweighted to friendly_weight; every other tournament label carries
    qualification or trophy stakes and stays at full weight 1.0. This keeps the
    scheme to a clean binary rather than hand-tiering 198 tournament labels.

    Returns all ones when the factors.friendly_downweight toggle is off, so the
    unweighted fit is reproduced exactly.
    """
    factors = config.get("factors", {}) or {}
    if not factors.get("friendly_downweight", False):
        return np.ones(len(tournaments), dtype=np.float64)
    dixon_coles_config = config.get("dixon_coles", {}) or {}
    friendly_weight = float(
        dixon_coles_config.get("friendly_weight", DEFAULT_FRIENDLY_WEIGHT)
    )
    is_friendly = tournaments.astype(str).str.strip().str.lower() == "friendly"
    return np.where(is_friendly.to_numpy(), friendly_weight, 1.0).astype(np.float64)


def fit_model(matches: pd.DataFrame, config: dict[str, Any]) -> DixonColesGoalModel:
    """Fit penaltyblog's Dixon-Coles on the guarded matches.

    Matches are weighted by Dixon-Coles time decay derived from each match date,
    so recent games carry more weight, and that weight is multiplied by a
    match-importance weight that downweights friendlies. The decay rate xi and the
    friendly weight are read from the dixon_coles config block, the latter applied
    only when the factors.friendly_downweight toggle is on. Returns the fitted
    model.
    """
    dixon_coles_config = config.get("dixon_coles", {}) or {}
    xi = float(dixon_coles_config.get("xi", DEFAULT_XI))

    played = matches.dropna(subset=["home_score", "away_score"])

    as_of_date = config.get("as_of_date")
    base_date = pd.Timestamp(as_of_date) if as_of_date else None
    time_weights = np.asarray(
        dixon_coles_weights(played["date"], xi=xi, base_date=base_date),
        dtype=np.float64,
    )

    # Time-decay weighting is preserved: time_weights above is the unchanged xi
    # decay. The match-importance multiplier stacks on top by elementwise product,
    # it does not replace the decay. Verified: with friendly_downweight off,
    # importance is all ones, so combined equals time_weights bit for bit and the
    # unweighted fit is reproduced exactly.
    importance = importance_weights(played["tournament"], config)
    combined = time_weights * importance

    # penaltyblog's Cython loss needs writable, contiguous buffers, so pass
    # fresh copies rather than the possibly read-only views the filtered frame
    # returns.
    model = DixonColesGoalModel(
        goals_home=np.array(played["home_score"], dtype=np.int64),
        goals_away=np.array(played["away_score"], dtype=np.int64),
        teams_home=played["home_team"].to_numpy(),
        teams_away=played["away_team"].to_numpy(),
        weights=np.array(combined, dtype=np.float64),
        neutral_venue=np.array(_neutral_flags(played["neutral"]), dtype=np.int64),
    )
    model.fit()
    return model


def match_probabilities(
    model: DixonColesGoalModel,
    home_team: str,
    away_team: str,
    neutral: bool = False,
) -> dict[str, float]:
    """Return outcome probabilities for a fixture from the scoreline grid.

    The dict has p_home_win, p_draw, and p_away_win, each a valid probability
    derived from the model's scoreline probability grid. When neutral is True the
    home advantage is excluded, so the home_team and away_team labels only set
    which side of the dict each probability lands on. This is the hook the
    simulator uses to keep World Cup venues neutral except for host nations.
    """
    grid = model.predict(home_team, away_team, neutral_venue=neutral)
    return {
        "p_home_win": float(grid.home_win),
        "p_draw": float(grid.draw),
        "p_away_win": float(grid.away_win),
    }
