"""Verification gate: Monte Carlo bracket simulation.

Hermetic test, no network, on a tiny synthetic field with a low simulation
count. Match outcomes come from a synthetic prob_fn driven by team strengths,
so the gate never touches penaltyblog or fits a model. Checks that stage
probabilities are valid and stack correctly, that win probabilities sum to one,
that a clearly stronger team wins more often than a clearly weaker one, that
knockout matches never return a draw, and, the key gate, that the same seed and
inputs reproduce identical results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from architect_wc import simulate

# Home edge in the synthetic strengths, only applied to non-neutral matches.
HOME_EDGE = 60.0
SEED = 7
N_SIMS = 300

# Stage fields produced for a 16-team bracket, in the order a team passes them.
STAGE_FIELDS = [
    "p_round_of_16",
    "p_quarter_final",
    "p_semi_final",
    "p_final",
    "p_win",
]


def _field() -> tuple[dict[str, list[str]], dict[str, float]]:
    """Build 6 groups of 4 with a clear strength gradient, T01 best, T24 worst.

    Six groups give a 16-team bracket filled by the top two of each group plus
    four best thirds, so the thirds path is exercised. The snake of strengths
    across groups keeps the strongest teams apart.
    """
    strength = {f"T{n:02d}": 2100.0 - 50.0 * (n - 1) for n in range(1, 25)}
    groups = {
        f"G{g}": [f"T{g + 1:02d}", f"T{g + 7:02d}", f"T{g + 13:02d}", f"T{g + 19:02d}"]
        for g in range(6)
    }
    return groups, strength


def _prob_fn(strength: dict[str, float]) -> simulate.ProbFn:
    """Synthetic outcome model: a logistic on strength, with a carved draw share."""

    def prob_fn(home: str, away: str, neutral: bool) -> tuple[float, float, float]:
        edge = 0.0 if neutral else HOME_EDGE
        diff = (strength[home] - strength[away]) + edge
        p_home_raw = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        p_draw = 0.25 * (1.0 - abs(2.0 * p_home_raw - 1.0))
        return (
            p_home_raw * (1.0 - p_draw),
            p_draw,
            (1.0 - p_home_raw) * (1.0 - p_draw),
        )

    return prob_fn


def _run() -> list[dict[str, object]]:
    groups, strength = _field()
    return simulate.run_simulations(
        groups, _prob_fn(strength), strength, n_sims=N_SIMS, seed=SEED
    )


def test_stage_probabilities_are_valid_and_stack() -> None:
    for record in _run():
        for field in STAGE_FIELDS:
            assert 0.0 <= record[field] <= 1.0
        # Reaching a later stage can never be more likely than an earlier one.
        for earlier, later in zip(STAGE_FIELDS, STAGE_FIELDS[1:], strict=False):
            assert record[earlier] >= record[later] - 1e-12


def test_win_probabilities_sum_to_one() -> None:
    total = sum(record["p_win"] for record in _run())
    assert abs(total - 1.0) <= 1e-9


def test_stronger_team_wins_more_often() -> None:
    results = {record["team"]: record for record in _run()}
    assert results["T01"]["p_win"] > results["T24"]["p_win"]


def test_knockout_never_returns_a_draw() -> None:
    _groups, strength = _field()
    prob_fn = _prob_fn(strength)
    rng = np.random.default_rng(0)
    pair = ("T11", "T12")  # near-even, so the group stage would draw often
    for _ in range(300):
        result = simulate.simulate_match(
            prob_fn, pair[0], pair[1], rng, neutral=True, knockout=True
        )
        assert result in pair


def test_simulation_is_reproducible() -> None:
    groups, strength = _field()
    prob_fn = _prob_fn(strength)
    first = simulate.run_simulations(
        groups, prob_fn, strength, n_sims=N_SIMS, seed=SEED
    )
    second = simulate.run_simulations(
        groups, prob_fn, strength, n_sims=N_SIMS, seed=SEED
    )
    assert first == second


def test_load_structure_reads_groups(tmp_path) -> None:
    path = tmp_path / "groups.csv"
    pd.DataFrame({"group": ["A", "A", "B", "B"], "team": ["W", "X", "Y", "Z"]}).to_csv(
        path, index=False
    )
    assert simulate.load_structure(path) == {"A": ["W", "X"], "B": ["Y", "Z"]}


def test_load_structure_skips_comment_lines(tmp_path) -> None:
    path = tmp_path / "groups.csv"
    path.write_text("# a dated note\ngroup,team\nA,W\nA,X\n", encoding="utf-8")
    assert simulate.load_structure(path) == {"A": ["W", "X"]}


def test_load_structure_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        simulate.load_structure(tmp_path / "absent.csv")


def test_load_structure_missing_column_raises(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"group": ["A"], "side": ["W"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        simulate.load_structure(path)
