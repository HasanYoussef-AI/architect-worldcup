"""Probability-distribution validity for the LLM predictions.

The simplex check used by the per-match prediction builders: an emitted three-way
must be a valid distribution. The per-match builders (prediction_b, reconcile_c)
carry their own citation and tolerance checks, so this module holds only the shared
distribution test.
"""

from __future__ import annotations

SUM_TOLERANCE = 1e-9


def three_way_sums_to_one(
    p_home_win_90: float, p_draw_90: float, p_away_win_90: float
) -> bool:
    """True if the three-way is a valid distribution within tolerance."""
    for value in (p_home_win_90, p_draw_90, p_away_win_90):
        if value < 0.0 or value > 1.0:
            return False
    return abs((p_home_win_90 + p_draw_90 + p_away_win_90) - 1.0) <= SUM_TOLERANCE
