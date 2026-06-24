"""The frozen, pre-registered factor taxonomy and weight vector.

These weights are priors, never fitted on outcomes. One review pass is allowed
before the Round of 32, then they become an ablation surface. They live in the
config so they are visible and ablatable, not buried in code. This module loads
them, checks the contract (all seven factors present, weights sum to one), and
aggregates per-factor scores into the single anchor signal. The model does not set
its own weights: P2 aggregation is this fixed dot product.
"""

from __future__ import annotations

from typing import Any

# The fixed factor taxonomy, in descending prior weight. The order is the
# canonical order used everywhere the factors are listed.
FACTORS = [
    "squad_availability",
    "recent_form",
    "tactical_matchup",
    "coaching_staff",
    "strategic_incentives",
    "psychological_momentum",
    "historical_head_to_head",
]

WEIGHT_SUM_TOLERANCE = 1e-9


def load_weights(config: dict[str, Any]) -> dict[str, float]:
    """Load and validate the frozen weight vector from config.

    Raises if a factor is missing, an unknown factor is present, or the weights do
    not sum to one within tolerance, because a silently malformed weight vector
    would corrupt every anchor signal.
    """
    llm_config = config.get("llm", {}) or {}
    raw = llm_config.get("weights", {}) or {}

    missing = [factor for factor in FACTORS if factor not in raw]
    if missing:
        raise ValueError(f"LLM weights are missing factors: {missing}.")
    unknown = [factor for factor in raw if factor not in FACTORS]
    if unknown:
        raise ValueError(f"LLM weights contain unknown factors: {unknown}.")

    weights = {factor: float(raw[factor]) for factor in FACTORS}
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"LLM weights sum to {total}, expected 1.0.")
    return weights


def aggregate(factor_scores: dict[str, float], weights: dict[str, float]) -> float:
    """Combine per-factor scores into the anchor signal, the fixed weighted sum.

    factor_scores maps each factor to its minus three to plus three score. The
    anchor signal is the weighted sum, bounded to the same minus three to plus
    three range since the weights sum to one. Raises if a factor is absent, so a
    missing factor is a loud error rather than a silent zero.
    """
    missing = [factor for factor in FACTORS if factor not in factor_scores]
    if missing:
        raise ValueError(f"Cannot aggregate, factor scores missing: {missing}.")
    return sum(weights[factor] * float(factor_scores[factor]) for factor in FACTORS)
