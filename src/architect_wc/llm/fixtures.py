"""Real knockout fixtures for a round, the source of truth for what we predict.

Once a round's fixtures are known (the group stage settles the Round of 32, each
later round is set by the prior round's real results), the official published
pairings are committed as a dated CSV, the same controlled-input pattern as the
group draw and the Annex C table. Predictions A and B both read these real
fixtures. The simulator-derived bracket is a cross-check that must equal the
committed field or fail loud; that check needs played results and is wired with
the live path.

Nominal home is the first-listed team in the official fixture, recorded in the
CSV. Since every knockout match is at a neutral venue, this is a labelling
convention only, applied identically to A, B, C, and scoring, so the three-way is
always read for the same nominal home side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

MATCH_COLUMN = "match"
ROUND_COLUMN = "round"
HOME_COLUMN = "home_team"
AWAY_COLUMN = "away_team"
REQUIRED_COLUMNS = (MATCH_COLUMN, ROUND_COLUMN, HOME_COLUMN, AWAY_COLUMN)


def load_fixtures(path: Any, round_code: str) -> list[dict[str, Any]]:
    """Read the committed knockout fixtures CSV and return one round's pairings.

    Returns a list of {match, home_team, away_team} for the given round, ordered by
    match number. home_team is the nominal home, the first-listed team. Raises a
    clear error on a missing file or column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Knockout fixtures not found: {path}. Expected a committed, dated CSV "
            f"with columns {list(REQUIRED_COLUMNS)}."
        )
    frame = pd.read_csv(path, comment="#")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Fixtures {path} is missing expected columns: {missing}. "
            f"Found {list(frame.columns)}."
        )

    rows = frame[frame[ROUND_COLUMN].astype(str).str.strip() == round_code]
    rows = rows.sort_values(MATCH_COLUMN, kind="stable")
    return [
        {
            "match": int(row.match),
            "home_team": str(getattr(row, HOME_COLUMN)).strip(),
            "away_team": str(getattr(row, AWAY_COLUMN)).strip(),
        }
        for row in rows.itertuples(index=False)
    ]
