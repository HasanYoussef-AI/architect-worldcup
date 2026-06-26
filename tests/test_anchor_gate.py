"""Verification gate: the frozen anchor-to-probability mapping (Phase 2).

Hermetic, no network. Reproduces the pre-registered reference anchor points to
absolute tolerance 0.003, including the negative-s mirror and the two shootout-lean
points; proves the three-way is a simplex and monotone across a sweep of s; checks
the tolerance box and the hard caps; confirms anchor_signal reuses the single weight
aggregation; and gates that the config record and the module constants never drift.
"""

from __future__ import annotations

import pytest

from architect_wc import pipeline
from architect_wc.llm import anchor
from architect_wc.llm import weights as weights_mod

TOL = 0.003

# Pre-registered reference points: s -> (p_home, p_draw, p_away).
REFERENCE_POINTS = {
    0.0: (0.3497, 0.3006, 0.3497),
    1.5: (0.5950, 0.2405, 0.1645),
    3.0: (0.8006, 0.1321, 0.0673),
}


def test_reference_anchor_points() -> None:
    for s, (p_home, p_draw, p_away) in REFERENCE_POINTS.items():
        h, d, a = anchor.reference_three_way(s)
        assert abs(h - p_home) <= TOL
        assert abs(d - p_draw) <= TOL
        assert abs(a - p_away) <= TOL


def test_negative_s_mirrors_home_and_away() -> None:
    for s in (0.5, 1.5, 2.7):
        h_pos, _, _ = anchor.reference_three_way(s)
        _, _, a_neg = anchor.reference_three_way(-s)
        assert abs(h_pos - a_neg) <= 1e-12


def test_shootout_lean_reference_points() -> None:
    assert abs(anchor.shootout_lean_from_anchor(0.0) - 0.5000) <= TOL
    assert abs(anchor.shootout_lean_from_anchor(3.0) - 0.6524) <= TOL


def test_three_way_is_a_simplex_over_a_sweep() -> None:
    for i in range(-30, 31):
        s = i / 10.0
        h, d, a = anchor.reference_three_way(s)
        assert h >= 0.0 and d >= 0.0 and a >= 0.0
        assert abs((h + d + a) - 1.0) <= 1e-12


def test_three_way_is_monotone() -> None:
    xs = [i / 10.0 for i in range(-30, 31)]
    homes = [anchor.reference_three_way(s)[0] for s in xs]
    draws = [anchor.reference_three_way(s)[1] for s in xs]
    aways = [anchor.reference_three_way(s)[2] for s in xs]
    # p_home strictly increasing in s, p_away strictly decreasing.
    assert all(homes[i] < homes[i + 1] for i in range(len(homes) - 1))
    assert all(aways[i] > aways[i + 1] for i in range(len(aways) - 1))
    # p_draw single-peaked at s = 0.
    peak = xs.index(0.0)
    assert draws[peak] == max(draws)
    assert all(draws[i] < draws[i + 1] for i in range(peak))
    assert all(draws[i] > draws[i + 1] for i in range(peak, len(draws) - 1))


def test_advance_probability_formula() -> None:
    assert anchor.advance_probability(0.5, 0.3, 0.6) == 0.5 + 0.3 * 0.6


def test_anchor_departure_axes() -> None:
    q = (0.5, 0.3, 0.2)
    r = (0.4, 0.35, 0.25)
    departure_dir, departure_draw = anchor.anchor_departure(q, r)
    assert abs(departure_dir - ((0.5 - 0.2) - (0.4 - 0.25))) <= 1e-12
    assert abs(departure_draw - (0.3 - 0.35)) <= 1e-12


def test_anchor_signal_reuses_weight_aggregation() -> None:
    config = pipeline.load_config()
    weights = weights_mod.load_weights(config)
    scores = dict.fromkeys(weights_mod.FACTORS, 0.0)
    scores["recent_form"] = 2.0
    s = anchor.anchor_signal(scores, weights)
    assert s == weights_mod.aggregate(scores, weights)
    assert abs(s - 2.0 * weights["recent_form"]) <= 1e-12


def test_classify_tolerance_box_and_hard_caps() -> None:
    a = anchor
    # All three within their tolerance.
    assert a.classify_tolerance(0.10, 0.05, 0.05, False) == a.STATUS_WITHIN_BOX
    # The box edges are inclusive.
    assert a.classify_tolerance(0.15, 0.10, 0.10, False) == a.STATUS_WITHIN_BOX
    # Outside the box, within the cap, with a justification.
    assert a.classify_tolerance(0.20, 0.0, 0.0, True) == a.STATUS_DEPARTED_JUSTIFIED
    # Outside the box, within the cap, no justification.
    assert a.classify_tolerance(0.20, 0.0, 0.0, False) == a.STATUS_REJECTED
    # Beyond the hard cap on each axis, rejected even with a justification.
    assert a.classify_tolerance(0.40, 0.0, 0.0, True) == a.STATUS_REJECTED
    assert a.classify_tolerance(0.0, 0.25, 0.0, True) == a.STATUS_REJECTED
    assert a.classify_tolerance(0.0, 0.0, 0.25, True) == a.STATUS_REJECTED


def test_config_anchor_matches_module_constants() -> None:
    # The pre-registered config record and the module constants must never drift.
    config = pipeline.load_config()
    assert anchor.frozen_params(config) == anchor.module_params()


def test_module_constants_are_the_frozen_values() -> None:
    assert (anchor.DELTA, anchor.BETA, anchor.GAMMA) == (0.62, 0.67, 0.21)
    assert anchor.DEPARTURE_DIR_TOLERANCE == 0.15
    assert anchor.DEPARTURE_DRAW_TOLERANCE == 0.10
    assert anchor.LEAN_TOLERANCE == 0.10


@pytest.mark.parametrize("s", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_reference_three_way_sums_exactly_to_one(s) -> None:
    h, d, a = anchor.reference_three_way(s)
    assert h + d + a == pytest.approx(1.0, abs=1e-12)
