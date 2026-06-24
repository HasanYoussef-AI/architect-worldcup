"""P4 validity and citation checks for the LLM predictions.

These enforce the prediction-phase contract beyond the JSON shape: probabilities
form a valid distribution, the bounded values stay bounded, and every nonzero
factor cites at least one dossier fact. An uncited nonzero factor is the
anti-hallucination gate: no evidence means the factor must score zero.
"""

from __future__ import annotations

from typing import Any

SUM_TOLERANCE = 1e-9


def three_way_sums_to_one(
    p_home_win_90: float, p_draw_90: float, p_away_win_90: float
) -> bool:
    """True if the three-way is a valid distribution within tolerance."""
    for value in (p_home_win_90, p_draw_90, p_away_win_90):
        if value < 0.0 or value > 1.0:
            return False
    return abs((p_home_win_90 + p_draw_90 + p_away_win_90) - 1.0) <= SUM_TOLERANCE


def nonzero_factors_are_cited(factor_scores: list[dict[str, Any]]) -> bool:
    """True if every nonzero factor cites at least one dossier fact.

    A factor scored zero may be uncited (it is neutral or flagged insufficient
    evidence). A nonzero factor with no evidence is a hallucination and fails.
    """
    for entry in factor_scores:
        if int(entry.get("score", 0)) != 0 and not entry.get("evidence"):
            return False
    return True


def assert_prediction_valid(tie: dict[str, Any], keys: dict[str, str]) -> None:
    """Raise if a tie's probabilities are not valid. keys is the A/B or C key map."""
    p_home = float(tie[keys["p_home_win_90"]])
    p_draw = float(tie[keys["p_draw_90"]])
    p_away = float(tie[keys["p_away_win_90"]])
    if not three_way_sums_to_one(p_home, p_draw, p_away):
        raise ValueError(
            f"Tie {tie.get('match')} three-way is not a valid distribution: "
            f"{(p_home, p_draw, p_away)}."
        )
    advance = float(tie[keys["p_advance_home"]])
    if advance < 0.0 or advance > 1.0:
        raise ValueError(f"Tie {tie.get('match')} advance probability out of bounds.")
