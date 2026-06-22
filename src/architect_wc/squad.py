"""Layer 4, squad value.

Owns the squad-value adjustment derived from a dated squad snapshot. The Elo
ratings from Layer 2 are the backbone; this layer nudges each team by a bounded
amount based on how its current squad market value compares to the field. A team
whose present squad is stronger than its history suggests gets lifted, and one
that is weaker gets tempered down. The adjustment is capped so Elo stays the
backbone and is never overridden.

Reading the dated CSV snapshot is kept separate from the pure adjustment logic,
so the core rule is fully unit-testable without any I/O. This layer does not
touch ingest, the leakage guard, or the Dixon-Coles fitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_WEIGHT = 40.0
DEFAULT_MAX_ADJUSTMENT = 75.0

TEAM_COLUMN = "team"
VALUE_COLUMN = "market_value_eur_m"


def load_squad_values(path: Any) -> dict[str, float]:
    """Read a squad-value snapshot and return a team to market value dict.

    The CSV must have a team column and a market_value_eur_m column, the market
    value being in millions of euros. Raise a clear error if the file or either
    column is missing. Rows with a blank team or a missing value are skipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Squad value snapshot not found: {path}. Expected a committed, "
            f"dated CSV with columns {TEAM_COLUMN} and {VALUE_COLUMN}."
        )

    df = pd.read_csv(path)
    missing = [
        column for column in (TEAM_COLUMN, VALUE_COLUMN) if column not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Squad value snapshot {path} is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}."
        )

    values: dict[str, float] = {}
    for team, value in zip(df[TEAM_COLUMN], df[VALUE_COLUMN], strict=True):
        if pd.isna(team) or pd.isna(value):
            continue
        values[str(team).strip()] = float(value)
    return values


def adjust_ratings(
    ratings: list[tuple[str, float]],
    squad_values: dict[str, float],
    config: dict[str, Any],
) -> list[tuple[str, float]]:
    """Nudge Elo ratings by a bounded amount from squad market value.

    ratings is the list of (team, rating) tuples from Layer 2. Each team's
    market value is turned into a z-score across the field, the field being the
    rated teams that have a value in the snapshot. The rating is then shifted by
    weight times that z-score, clamped to plus or minus max_adjustment rating
    points. A team at the field mean gets no change. A team with no value in the
    snapshot is returned unchanged. The cap keeps the nudge a temper on Elo, not
    an override.

    Pure function with no I/O. Returns a new list of (team, adjusted_rating)
    tuples sorted from highest to lowest. The input list is not mutated.
    """
    squad_config = config.get("squad", {}) or {}
    weight = float(squad_config.get("weight", DEFAULT_WEIGHT))
    max_adjustment = abs(
        float(squad_config.get("max_adjustment", DEFAULT_MAX_ADJUSTMENT))
    )

    field = [squad_values[team] for team, _ in ratings if team in squad_values]
    count = len(field)
    if count:
        mean = sum(field) / count
        variance = sum((value - mean) ** 2 for value in field) / count
        std = variance**0.5
    else:
        mean = 0.0
        std = 0.0

    adjusted: list[tuple[str, float]] = []
    for team, rating in ratings:
        if team in squad_values and std > 0.0:
            z_score = (squad_values[team] - mean) / std
            nudge = max(-max_adjustment, min(max_adjustment, weight * z_score))
            adjusted.append((team, rating + nudge))
        else:
            adjusted.append((team, rating))

    return sorted(adjusted, key=lambda item: item[1], reverse=True)
