"""Verification gate: the leakage guard.

Hermetic test, no network. Builds a small in-memory dataframe whose dates
straddle a chosen as_of_date, then asserts the leakage guard keeps only the
rows dated on or before the cutoff. This is the core data integrity rule for
Layer 1.
"""

from __future__ import annotations

import pandas as pd

from architect_wc.ingest import apply_leakage_guard

AS_OF_DATE = "2026-06-22"


def _straddling_df() -> pd.DataFrame:
    """A frame with rows before, exactly on, and after the cutoff."""
    frame = pd.DataFrame(
        {
            "date": [
                "2026-01-01",  # before
                "2026-03-15",  # before
                "2026-06-22",  # exactly on the cutoff
                "2026-06-23",  # after
                "2026-12-31",  # after
            ],
            "home_team": ["A", "C", "E", "G", "I"],
            "away_team": ["B", "D", "F", "H", "J"],
            "home_score": [1, 2, 0, 3, 1],
            "away_score": [0, 2, 0, 1, 1],
            "tournament": ["Friendly"] * 5,
            "city": ["Citytown"] * 5,
            "country": ["Countryland"] * 5,
            "neutral": [False] * 5,
        }
    )
    # Parse to a real date type, as load_raw does.
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame


def test_leakage_guard_keeps_only_on_or_before_cutoff() -> None:
    cutoff = pd.Timestamp(AS_OF_DATE)
    result = apply_leakage_guard(_straddling_df(), AS_OF_DATE)
    result_dates = pd.to_datetime(result["date"])

    # The maximum date in the result does not exceed the cutoff.
    assert result_dates.max() <= cutoff

    # No row dated after the cutoff survives.
    assert not (result_dates > cutoff).any()

    # The row dated exactly on the cutoff is kept.
    assert cutoff in set(result_dates)

    # The earlier rows are kept.
    assert pd.Timestamp("2026-01-01") in set(result_dates)
    assert pd.Timestamp("2026-03-15") in set(result_dates)

    # Exactly the three on-or-before rows remain.
    assert len(result) == 3
