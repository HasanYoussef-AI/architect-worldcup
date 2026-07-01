"""Verification gate: the B-coherence prompt fix and the tolerance contract it must
make the model honor.

Offline and hermetic, no network, no dependency on anything under outputs/. The first
test pins the prompt so the coherence clauses cannot silently regress. The second and
third exercise the real, untouched anchor mapping and tolerance gate to prove two
things at once: the gate still rejects the exact match-74 incoherence (a strong-
favorite emission on near-even factor scores), and a coherent emission, whether
near-even or legitimately lopsided through the factor scores, passes. The prompt is
what pushes the model toward the coherent emission; the gate is unchanged.
"""

from __future__ import annotations

from architect_wc import pipeline
from architect_wc.llm import anchor, model_call
from architect_wc.llm.weights import load_weights

TOL = 1e-9


def _weights() -> dict[str, float]:
    return load_weights(pipeline.load_config())


def _reference(scores: dict[str, int]) -> tuple[float, float, float]:
    """Reference three-way through the real mapping the orchestrator uses."""
    s = anchor.anchor_signal(scores, _weights())
    return anchor.reference_three_way(s)


# --- test 1: the prompt cannot regress ------------------------------------------


def test_b_prompt_carries_the_coherence_clauses() -> None:
    system = model_call.PREDICTION_B_SYSTEM
    for sentinel in (
        "must follow from your seven factor scores",
        "Prediction A's domain and is already counted there",
        "excludes raw results, goal difference, and standings",
    ):
        assert sentinel in system, f"B prompt lost the coherence clause: {sentinel!r}"


# --- test 2: match-74 replay, the contract the prompt must make the model honor --

# The seven factor scores Prediction B emitted on the match-74 (Germany vs Paraguay)
# rehearsal, from the rejected artifact
# outputs/logs/llm_call_b_rejected_R32_match74_20260630T032435920107Z.json: they net
# essentially even (signal -0.01), yet B emitted a Germany-favored {0.52, 0.27, 0.21}.
MATCH_74_SCORES = {
    "squad_availability": -1,
    "recent_form": 2,
    "tactical_matchup": -1,
    "coaching_staff": 0,
    "strategic_incentives": 0,
    "psychological_momentum": -1,
    "historical_head_to_head": 1,
}
# The historical incoherent emission that was correctly rejected.
MATCH_74_EMITTED = (0.52, 0.27, 0.21)


def test_match74_incoherent_rejected_but_coherent_passes() -> None:
    signal = anchor.anchor_signal(MATCH_74_SCORES, _weights())
    assert abs(signal - (-0.01)) < TOL  # the scores net near even.

    r_home, r_draw, r_away = anchor.reference_three_way(signal)

    # The historical emission: a strong-favorite line on near-even scores. Its
    # direction departure exceeds the 0.30 hard cap, so the gate rejects it even with
    # a justification present. The gate is unchanged; this pins that.
    dep_dir, dep_draw = anchor.anchor_departure(
        MATCH_74_EMITTED, (r_home, r_draw, r_away)
    )
    assert dep_dir > anchor.HARD_CAP_MULTIPLE * anchor.DEPARTURE_DIR_TOLERANCE
    assert abs(dep_dir - 0.313) < 0.01  # matches the artifact's 0.3130.
    status_incoherent = anchor.classify_tolerance(
        dep_dir, dep_draw, 0.0, has_justification=True
    )
    assert status_incoherent == anchor.STATUS_REJECTED

    # A coherent emission on the exact same scores, equal to the reference, is not
    # rejected. This is the emission the coherence prompt steers the model toward.
    coh_dir, coh_draw = anchor.anchor_departure(
        (r_home, r_draw, r_away), (r_home, r_draw, r_away)
    )
    status_coherent = anchor.classify_tolerance(
        coh_dir, coh_draw, 0.0, has_justification=False
    )
    assert status_coherent != anchor.STATUS_REJECTED
    assert status_coherent == anchor.STATUS_WITHIN_BOX


# --- test 3: legitimate favoritism flows through the factors ---------------------

LOPSIDED_HOME_SCORES = {
    "squad_availability": 2,
    "recent_form": 2,
    "tactical_matchup": 2,
    "coaching_staff": 1,
    "strategic_incentives": 1,
    "psychological_momentum": 2,
    "historical_head_to_head": 1,
}


def test_lopsided_through_factors_leans_home_and_passes() -> None:
    r_home, r_draw, r_away = _reference(LOPSIDED_HOME_SCORES)
    # Clearly home-favoring factor scores produce a home-leaning reference.
    assert r_home > r_away

    # A home-favored emission within the direction tolerance of that reference (shift
    # 0.03 of mass from away to home) is not rejected: legitimate favoritism earned
    # through the factor scores flows through the gate rather than around it.
    emitted = (r_home + 0.03, r_draw, r_away - 0.03)
    dep_dir, dep_draw = anchor.anchor_departure(emitted, (r_home, r_draw, r_away))
    assert abs(dep_dir) <= anchor.DEPARTURE_DIR_TOLERANCE
    status = anchor.classify_tolerance(dep_dir, dep_draw, 0.0, has_justification=False)
    assert status != anchor.STATUS_REJECTED
    assert emitted[0] > emitted[2]  # the emission itself leans home.
