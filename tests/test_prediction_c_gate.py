"""Verification gate: the offline Prediction C reconciliation and the A-B pool.

Hermetic, no network, model boundary stubbed. Proves the pool is the exact
componentwise average of A and B and a valid simplex; that C reuses anchor's
tolerance constants and classify_tolerance unforked, by import not redefinition; and
runs C end to end on the committed dossier with stubbed A and B in two cases, one
in-box move that is accepted and one beyond the pool hard cap that P4 rejects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from architect_wc import pipeline
from architect_wc.llm import anchor, reconcile_c, schemas

ROOT = Path(__file__).resolve().parents[1]
DOSSIER_PATH = ROOT / "outputs/llm/dossier_GROUP_match64_asof2026-06-25.json"
REF = {"path": "stub", "content_sha256": "stub"}


def _dossier() -> dict:
    return json.loads(DOSSIER_PATH.read_text(encoding="utf-8"))["dossier"]


def _dossier_ref() -> dict[str, str]:
    return {
        "path": str(DOSSIER_PATH),
        "content_sha256": hashlib.sha256(DOSSIER_PATH.read_bytes()).hexdigest(),
    }


def _stub_a() -> dict:
    return {
        "three_way": {"p_home": 0.40, "p_draw": 0.30, "p_away": 0.30},
        "shootout_lean": 0.55,
    }


def _stub_b() -> dict:
    return {
        "emitted_three_way": {"p_home": 0.30, "p_draw": 0.30, "p_away": 0.40},
        "shootout": {"emitted_lean": 0.45},
    }


def test_pool_is_the_exact_componentwise_average_and_a_simplex() -> None:
    pool = reconcile_c.pool_ab(
        _stub_a()["three_way"], _stub_b()["emitted_three_way"], 0.55, 0.45
    )
    tw = pool["three_way"]
    assert abs(tw["p_home"] - 0.35) <= 1e-12
    assert abs(tw["p_draw"] - 0.30) <= 1e-12
    assert abs(tw["p_away"] - 0.35) <= 1e-12
    assert abs(pool["lean"] - 0.50) <= 1e-12
    assert abs((tw["p_home"] + tw["p_draw"] + tw["p_away"]) - 1.0) <= 1e-12


def test_c_reuses_anchor_tolerance_unforked() -> None:
    # Reuse by import, never redefinition. reconcile_c calls anchor.classify_tolerance
    # and anchor's constants; it defines none of its own.
    assert reconcile_c.anchor.classify_tolerance is anchor.classify_tolerance
    assert "classify_tolerance" not in vars(reconcile_c)
    assert "DEPARTURE_DIR_TOLERANCE" not in vars(reconcile_c)
    assert anchor.DEPARTURE_DIR_TOLERANCE == 0.15


def _accept_c_call(dossier, prediction_a, prediction_b, config) -> dict:
    pool = reconcile_c.pool_from_predictions(prediction_a, prediction_b)
    p = pool["three_way"]
    # In-box move: shift 0.03 home to away, departure_dir 0.06.
    return {
        "emitted_three_way": {
            "p_home": p["p_home"] + 0.03,
            "p_draw": p["p_draw"],
            "p_away": p["p_away"] - 0.03,
        },
        "emitted_lean": pool["lean"] + 0.02,
        "justification": None,
        "citations": [{"source": "a_reasoning", "ref": "A's elo_diff favours home"}],
    }


def _reject_c_call(dossier, prediction_a, prediction_b, config) -> dict:
    pool = reconcile_c.pool_from_predictions(prediction_a, prediction_b)
    p = pool["three_way"]
    # Beyond the cap: shift 0.20, departure_dir 0.40 > 0.30.
    return {
        "emitted_three_way": {
            "p_home": p["p_home"] + 0.20,
            "p_draw": p["p_draw"],
            "p_away": p["p_away"] - 0.20,
        },
        "emitted_lean": pool["lean"],
        "justification": "deliberately beyond the hard cap, must still be rejected",
        "citations": [{"source": "b_reasoning", "ref": "B leaned the other way"}],
    }


def test_c_accept_case(tmp_path) -> None:
    config = pipeline.load_config()
    document = reconcile_c.predict_c(
        _dossier(),
        _stub_a(),
        _stub_b(),
        config,
        _accept_c_call,
        dossier_ref=_dossier_ref(),
        a_ref=REF,
        b_ref=REF,
        code_commit="test",
        timestamp="2026-06-25T00:00:00Z",
    )
    schemas.validate_document(document, "prediction_c")
    assert document["validation"]["accepted"] is True
    assert document["validation"]["rejection_reason"] is None
    assert document["tolerance"]["status"] == "within_box"
    assert abs(document["departure"]["dir"] - 0.06) <= 1e-9
    # The advance reads off C's emitted three-way and emitted lean.
    em = document["emitted_three_way"]
    sh = document["shootout"]
    expected = em["p_home"] + em["p_draw"] * sh["emitted_lean"]
    assert abs(sh["advance_probability"] - expected) <= 1e-12
    # The pool is recorded as a first-class line.
    assert abs(document["pool"]["three_way"]["p_home"] - 0.35) <= 1e-12

    out = tmp_path / "prediction_c_accept.json"
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    assert out.exists()


def test_c_reject_case(tmp_path) -> None:
    config = pipeline.load_config()
    document = reconcile_c.predict_c(
        _dossier(),
        _stub_a(),
        _stub_b(),
        config,
        _reject_c_call,
        dossier_ref=_dossier_ref(),
        a_ref=REF,
        b_ref=REF,
        code_commit="test",
        timestamp="2026-06-25T00:00:00Z",
    )
    schemas.validate_document(document, "prediction_c")
    assert document["tolerance"]["status"] == "rejected"
    assert document["validation"]["tolerance_ok"] is False
    assert document["validation"]["accepted"] is False
    assert document["validation"]["rejection_reason"]
    assert document["validation"]["simplex_ok"] is True
    assert abs(document["departure"]["dir"] - 0.40) <= 1e-9

    out = tmp_path / "prediction_c_reject.json"
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    assert out.exists()


def test_c_rejects_a_material_move_with_no_citation() -> None:
    # C's analog of the nonzero-score-must-cite guarantee, on the live path: a
    # material move off the pool with no citation is not accepted, even when it is
    # justified and inside the hard cap. C introduces no uncited moves.
    config = pipeline.load_config()

    def uncited_move_call(dossier, prediction_a, prediction_b, config):
        pool = reconcile_c.pool_from_predictions(prediction_a, prediction_b)
        p = pool["three_way"]
        # dir 0.20: outside the soft box, within the hard cap, justified, but uncited.
        return {
            "emitted_three_way": {
                "p_home": p["p_home"] + 0.10,
                "p_draw": p["p_draw"],
                "p_away": p["p_away"] - 0.10,
            },
            "emitted_lean": pool["lean"],
            "justification": "moved off the pool",
            "citations": [],
        }

    document = reconcile_c.predict_c(
        _dossier(),
        _stub_a(),
        _stub_b(),
        config,
        uncited_move_call,
        dossier_ref=_dossier_ref(),
        a_ref=REF,
        b_ref=REF,
        code_commit="test",
        timestamp="2026-06-25T00:00:00Z",
    )
    assert document["tolerance"]["status"] == "departed_justified"
    assert document["validation"]["citations_ok"] is False
    assert document["validation"]["accepted"] is False
    assert (
        "material move off the pool is uncited"
        in document["validation"]["rejection_reason"]
    )
