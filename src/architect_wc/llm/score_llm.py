"""Scoring for the math versus LLM comparison.

Reuses the project's single RPS implementation (calibrate.rps); it does not write a
second scorer. The headline metric is the 90-minute three-way RPS, identical to the
target the model already scores. The secondary metric is a binary score on advance,
routed through the same RPS with a two-element vector: RPS on two ordered outcomes
equals the squared error (p - indicator)^2, the Brier-equivalent, so it is the
existing scorer, not a new one. Every prediction line, A, B, C, and the A-B pool
baseline, scores through the same score_line, so combination and judgment can be
separated later without a second scorer.
"""

from __future__ import annotations

from architect_wc import calibrate


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


def score_line(
    three_way: dict[str, float],
    advance_probability: float,
    outcome_index: int,
    home_advanced: bool,
) -> dict[str, float]:
    """Score one prediction line: three-way RPS and advance RPS, the one path.

    three_way is a {p_home, p_draw, p_away} dict; the same call scores Prediction A,
    B, C, and the A-B pool baseline, through the same RPS scorer with no second
    scorer.
    """
    ordered = (
        float(three_way["p_home"]),
        float(three_way["p_draw"]),
        float(three_way["p_away"]),
    )
    return {
        "rps_three_way": score_three_way(ordered, outcome_index),
        "rps_advance": score_advance(float(advance_probability), home_advanced),
    }


def mean(values: list[float]) -> float:
    """Plain mean, raising on an empty list rather than dividing by zero."""
    if not values:
        raise ValueError("Cannot take the mean of zero scores.")
    return sum(values) / len(values)
