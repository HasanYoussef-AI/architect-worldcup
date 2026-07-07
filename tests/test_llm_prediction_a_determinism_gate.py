"""Fixed-as-of determinism gate for Prediction A.

Prediction A is analytic, the neutral-venue Dixon-Coles three-way plus the Elo
shootout lean, with no model call. At a single fixed as-of date it must therefore
be bit-identical on a repeat build. That is the property A guarantees and the one
this gate pins.

This is deliberately not a cross-as-of invariance test. Prediction A is not
bit-invariant across as-of dates, because the Dixon-Coles time-decay reference is
anchored to the as-of date, so the fit shifts slightly when the as-of moves.
Forcing invariance would remove the intended recency weighting. See the 2026-07-02
session-log finding for the root cause.

Snapshot-gated: build_prediction_a fits the pipeline on the local results snapshot,
which is gitignored and absent in CI, so this self-skips where the snapshot is not
present, mirroring the group-resolution gate in test_simulate_gate.py.
"""

from pathlib import Path

import pytest

from architect_wc import pipeline
from architect_wc.llm import live, prediction_a

# RAW_DIR copied exactly from tests/test_simulate_gate.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"


def _without_provenance(document: dict) -> dict:
    """The document without its provenance block.

    build_prediction_a stamps provenance.code_commit from the current git sha. That
    is identical within one run, but it is git state, not the forecast math.
    Determinism here is a claim about the math, so provenance is excluded before the
    equality assertion.
    """
    return {key: value for key, value in document.items() if key != "provenance"}


def test_build_prediction_a_is_deterministic_at_fixed_as_of() -> None:
    snapshots = sorted(RAW_DIR.glob("results_snapshot_*.csv"))
    if not snapshots:
        pytest.skip("no local results snapshot; build_prediction_a needs the fit data")

    config = pipeline.load_config()
    as_of_date = "2026-07-02"  # a fixed date the local snapshot covers
    round_code = "R32"

    # A genuine fixture from the committed knockout_fixtures_2026.csv via the live
    # tie-selection helper: R32 match 83, Portugal vs Croatia.
    tie = live.select_tie(config, round_code, 83)
    fixture = {
        "match": 83,
        "home_team": tie["home_team"],
        "away_team": tie["away_team"],
    }

    first = prediction_a.build_prediction_a(config, fixture, as_of_date, round_code)
    second = prediction_a.build_prediction_a(config, fixture, as_of_date, round_code)

    # Determinism is a claim about the math, so compare with the git-sha provenance
    # excluded.
    assert _without_provenance(first) == _without_provenance(second)
