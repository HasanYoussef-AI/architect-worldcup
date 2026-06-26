"""Frozen anchor-to-probability mapping for Prediction B, the prediction machine.

Pure, deterministic, no I/O. The anchor signal s is the frozen weighted sum of the
seven per-factor scores from the nominal home team's perspective, where nominal home
is the first-listed team and positive s favours it. The 90-minute three-way is a
symmetric ordered logit with no intercept, and the Prediction B shootout lean is a
logistic of s.

The frozen parameters delta, beta, gamma, and the tolerance box are pre-registered
priors from general football knowledge: set before the tournament, never fitted on
2026 outcomes, and never derived from Prediction A. Their canonical record lives in
config under llm.anchor; the module constants here mirror it, and a verification gate
asserts the two never drift, so the pure mapping functions can stay single-argument
and deterministic while config remains the documented frozen block. The weights are
not duplicated here: they are loaded from config via weights.load_weights and reused.

Two distinct shootout leans exist and must not collide. Prediction A uses the
strength-weighted Elo coin flip simulate.shootout_lean_from_elo; Prediction B uses
shootout_lean_from_anchor below.
"""

from __future__ import annotations

import math
from typing import Any

from architect_wc.llm.weights import FACTORS, aggregate

# Frozen, pre-registered priors. These mirror config.llm.anchor; the drift gate in
# the verification suite asserts the two are identical so neither can move silently.
DELTA = 0.62
BETA = 0.67
GAMMA = 0.21

# The pre-registered tolerance box: how far the emitted distribution may sit from the
# anchor on each axis before a written justification is required.
DEPARTURE_DIR_TOLERANCE = 0.15
DEPARTURE_DRAW_TOLERANCE = 0.10
LEAN_TOLERANCE = 0.10

# The hard cap is twice each tolerance. Beyond it the prediction is rejected for
# regeneration regardless of any justification.
HARD_CAP_MULTIPLE = 2.0

# Tolerance classification outcomes.
STATUS_WITHIN_BOX = "within_box"
STATUS_DEPARTED_JUSTIFIED = "departed_justified"
STATUS_REJECTED = "rejected"

Triple = tuple[float, float, float]


def _sigmoid(x: float) -> float:
    """Logistic function, the open unit interval."""
    return 1.0 / (1.0 + math.exp(-x))


def anchor_signal(factor_scores: dict[str, float], weights: dict[str, float]) -> float:
    """Return s, the frozen weighted sum of the per-factor scores.

    Each score is in minus three to plus three, from the nominal home perspective.
    Reuses the single weights aggregation so the weight vector is never redefined.
    """
    return aggregate(factor_scores, weights)


def reference_three_way(s: float) -> Triple:
    """Reference 90-minute three-way at anchor signal s. Returns (p_home, p_draw,
    p_away).

    Symmetric ordered logit, no intercept: eta = beta * s; p_home = sigmoid(eta -
    delta); p_away = sigmoid(-delta - eta); p_draw = the remainder. Symmetry gives
    p_home(s) equals p_away(-s).
    """
    eta = BETA * s
    p_home = _sigmoid(eta - DELTA)
    p_away = _sigmoid(-DELTA - eta)
    p_draw = 1.0 - p_home - p_away
    return p_home, p_draw, p_away


def shootout_lean_from_anchor(s: float) -> float:
    """Prediction B shootout lean off the anchor, sigmoid(gamma * s).

    Distinct from the Elo shootout lean simulate.shootout_lean_from_elo that
    Prediction A uses. The two never collide.
    """
    return _sigmoid(GAMMA * s)


def advance_probability(p_home: float, p_draw: float, lean: float) -> float:
    """Probability the nominal home side advances: p_home + p_draw * lean. Pure."""
    return p_home + p_draw * lean


def anchor_departure(q: Triple, r: Triple) -> tuple[float, float]:
    """Two independent departure axes of the emitted distribution q from the
    reference r.

    Each is a (p_home, p_draw, p_away) triple. departure_dir is the difference in the
    home-minus-away signed margin; departure_draw is the difference in the draw mass.
    """
    q_home, q_draw, q_away = q
    r_home, r_draw, r_away = r
    departure_dir = (q_home - q_away) - (r_home - r_away)
    departure_draw = q_draw - r_draw
    return departure_dir, departure_draw


def classify_tolerance(
    departure_dir: float,
    departure_draw: float,
    lean_departure: float,
    has_justification: bool,
) -> str:
    """Classify a prediction against the pre-registered box and the hard caps.

    within_box: all three departures within their tolerance. departed_justified:
    outside the box but within twice each tolerance, with a written justification.
    rejected: beyond any hard cap (regardless of justification), or outside the box
    with no justification.
    """
    within_box = (
        abs(departure_dir) <= DEPARTURE_DIR_TOLERANCE
        and abs(departure_draw) <= DEPARTURE_DRAW_TOLERANCE
        and abs(lean_departure) <= LEAN_TOLERANCE
    )
    beyond_cap = (
        abs(departure_dir) > HARD_CAP_MULTIPLE * DEPARTURE_DIR_TOLERANCE
        or abs(departure_draw) > HARD_CAP_MULTIPLE * DEPARTURE_DRAW_TOLERANCE
        or abs(lean_departure) > HARD_CAP_MULTIPLE * LEAN_TOLERANCE
    )
    if within_box:
        return STATUS_WITHIN_BOX
    if beyond_cap:
        return STATUS_REJECTED
    return STATUS_DEPARTED_JUSTIFIED if has_justification else STATUS_REJECTED


def frozen_params(config: dict[str, Any]) -> dict[str, float]:
    """The config record of the frozen anchor parameters, for the drift gate."""
    return dict((config.get("llm", {}) or {}).get("anchor", {}) or {})


def module_params() -> dict[str, float]:
    """The module constants, in the same shape as the config record."""
    return {
        "delta": DELTA,
        "beta": BETA,
        "gamma": GAMMA,
        "departure_dir_tolerance": DEPARTURE_DIR_TOLERANCE,
        "departure_draw_tolerance": DEPARTURE_DRAW_TOLERANCE,
        "lean_tolerance": LEAN_TOLERANCE,
    }


# Re-exported so consumers see the factor order without importing weights directly.
__all__ = [
    "FACTORS",
    "anchor_signal",
    "reference_three_way",
    "shootout_lean_from_anchor",
    "advance_probability",
    "anchor_departure",
    "classify_tolerance",
    "frozen_params",
    "module_params",
]
