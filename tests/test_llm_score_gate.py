"""Verification gate: the LLM-phase scorer reuses RPS, and the pool scores through it.

Hermetic, no network. Proves the three-way score reproduces a hand-checked value,
the binary advance score equals the Brier-equivalent (p - indicator)^2 through the
same RPS implementation, and that the A-B pool baseline routes through the same
score_line as A, B, and C with no second scorer.
"""

from __future__ import annotations

from architect_wc import calibrate
from architect_wc.llm import reconcile_c, score_llm, smoke

TOLERANCE = 1e-9


def test_three_way_reproduces_hand_checked_value() -> None:
    three_way, outcome, expected = smoke.HAND_CHECKED_THREE_WAY
    assert abs(score_llm.score_three_way(three_way, outcome) - expected) <= TOLERANCE


def test_advance_is_brier_equivalent_through_rps() -> None:
    p, advanced, expected = smoke.HAND_CHECKED_ADVANCE
    got = score_llm.score_advance(p, advanced)
    assert abs(got - expected) <= TOLERANCE
    assert abs(got - (p - 1.0) ** 2) <= TOLERANCE


def test_advance_uses_the_same_rps_as_three_way() -> None:
    p = 0.4
    assert score_llm.score_advance(p, True) == calibrate.rps((p, 1.0 - p), 0)
    assert score_llm.score_advance(p, False) == calibrate.rps((p, 1.0 - p), 1)


def test_pool_baseline_scores_through_the_same_rps_path() -> None:
    # The A-B pool is a scoreable line; it routes through score_line, the one scorer
    # A, B, and C use, with no second scorer.
    a_three_way = {"p_home": 0.40, "p_draw": 0.30, "p_away": 0.30}
    b_three_way = {"p_home": 0.30, "p_draw": 0.30, "p_away": 0.40}
    pool = reconcile_c.pool_ab(a_three_way, b_three_way, 0.55, 0.45)

    scores = score_llm.score_line(
        pool["three_way"],
        pool["advance_probability"],
        calibrate.HOME_WIN,
        home_advanced=True,
    )
    assert 0.0 <= scores["rps_three_way"] <= 1.0
    assert 0.0 <= scores["rps_advance"] <= 1.0
    # It is literally the project RPS, not a new scorer.
    pool_tw = pool["three_way"]
    expected = calibrate.rps(
        (pool_tw["p_home"], pool_tw["p_draw"], pool_tw["p_away"]), calibrate.HOME_WIN
    )
    assert scores["rps_three_way"] == expected
    assert scores["rps_advance"] == score_llm.score_advance(
        pool["advance_probability"], True
    )
