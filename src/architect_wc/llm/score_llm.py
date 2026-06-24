"""Scoring for the math versus LLM comparison.

Reuses the project's single RPS implementation (calibrate.rps); it does not write
a second scorer. The headline metric is the 90-minute three-way RPS, identical to
the target the model already scores. The secondary metric is a binary score on
advance, routed through the same RPS with a two-element vector: RPS on two ordered
outcomes equals the squared error (p - indicator)^2, the Brier-equivalent, so it
is the existing scorer, not a new one. Scores aggregate per round and cumulatively
across rounds.
"""

from __future__ import annotations

from typing import Any

from architect_wc import calibrate

# Which keys hold the three-way and advance in each prediction. A and B use the
# plain keys; C uses the reconciled_ prefix.
PLAIN_KEYS = {
    "p_home_win_90": "p_home_win_90",
    "p_draw_90": "p_draw_90",
    "p_away_win_90": "p_away_win_90",
    "p_advance_home": "p_advance_home",
}
RECONCILED_KEYS = {
    "p_home_win_90": "reconciled_p_home_win_90",
    "p_draw_90": "reconciled_p_draw_90",
    "p_away_win_90": "reconciled_p_away_win_90",
    "p_advance_home": "reconciled_p_advance_home",
}


def keys_for(prediction_kind: str) -> dict[str, str]:
    """Return the tie-record key map for prediction A, B, or C."""
    return RECONCILED_KEYS if prediction_kind == "C" else PLAIN_KEYS


def score_three_way(three_way: tuple[float, float, float], outcome_index: int) -> float:
    """RPS of a 90-minute three-way against the actual ordered outcome. Reuses RPS."""
    return calibrate.rps(three_way, outcome_index)


def score_advance(p_advance_home: float, home_advanced: bool) -> float:
    """Binary advance score via the existing RPS with a two-element vector.

    Equals (p_advance_home - indicator)^2, the Brier-equivalent. The nominal home
    side advancing is outcome 0, not advancing is outcome 1.
    """
    outcome = 0 if home_advanced else 1
    return calibrate.rps((p_advance_home, 1.0 - p_advance_home), outcome)


def score_tie(
    tie: dict[str, Any],
    keys: dict[str, str],
    outcome_index: int,
    home_advanced: bool,
) -> dict[str, float]:
    """Score one tie record: three-way RPS and advance RPS."""
    three_way = (
        float(tie[keys["p_home_win_90"]]),
        float(tie[keys["p_draw_90"]]),
        float(tie[keys["p_away_win_90"]]),
    )
    return {
        "rps_three_way": score_three_way(three_way, outcome_index),
        "rps_advance": score_advance(float(tie[keys["p_advance_home"]]), home_advanced),
    }


def mean(values: list[float]) -> float:
    """Plain mean, raising on an empty list rather than dividing by zero."""
    if not values:
        raise ValueError("Cannot take the mean of zero scores.")
    return sum(values) / len(values)
