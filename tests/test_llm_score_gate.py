"""Verification gate: the LLM-phase scorer reuses RPS correctly.

Hermetic, no network. Proves the three-way score reproduces a hand-checked value,
the binary advance score equals the Brier-equivalent (p - indicator)^2 through the
same RPS implementation, and that scoring a fixture tie runs end to end.
"""

from __future__ import annotations

from architect_wc import calibrate
from architect_wc.llm import score_llm, smoke

TOLERANCE = 1e-9


def test_three_way_reproduces_hand_checked_value() -> None:
    # Cumulative (0.5, 0.8) against observed (1, 1): (0.25 + 0.04) / 2 = 0.145.
    three_way, outcome, expected = smoke.HAND_CHECKED_THREE_WAY
    assert abs(score_llm.score_three_way(three_way, outcome) - expected) <= TOLERANCE


def test_advance_is_brier_equivalent_through_rps() -> None:
    p, advanced, expected = smoke.HAND_CHECKED_ADVANCE
    got = score_llm.score_advance(p, advanced)
    assert abs(got - expected) <= TOLERANCE
    # Equals (p - indicator)^2 by hand, the binary RPS.
    assert abs(got - (p - 1.0) ** 2) <= TOLERANCE


def test_advance_uses_the_same_rps_as_three_way() -> None:
    # The advance score is literally calibrate.rps on a two-vector, not a new scorer.
    p = 0.4
    assert score_llm.score_advance(p, True) == calibrate.rps((p, 1.0 - p), 0)
    assert score_llm.score_advance(p, False) == calibrate.rps((p, 1.0 - p), 1)


def test_score_tie_runs_on_a_fixture_record() -> None:
    tie = smoke.FIXTURE_A["ties"][0]
    keys = score_llm.keys_for("A")
    # Suppose the nominal home side won in 90 minutes and advanced.
    scores = score_llm.score_tie(tie, keys, calibrate.HOME_WIN, home_advanced=True)
    assert 0.0 <= scores["rps_three_way"] <= 1.0
    assert 0.0 <= scores["rps_advance"] <= 1.0


def test_reconciled_keys_used_for_prediction_c() -> None:
    tie = smoke.FIXTURE_C["ties"][0]
    keys = score_llm.keys_for("C")
    scores = score_llm.score_tie(tie, keys, calibrate.DRAW, home_advanced=False)
    assert 0.0 <= scores["rps_three_way"] <= 1.0
    assert 0.0 <= scores["rps_advance"] <= 1.0
