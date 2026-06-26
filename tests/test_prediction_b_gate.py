"""Verification gate: the offline Prediction B prediction machine (Phase 2).

End to end on the committed Phase 1 dossier, with only the model call boundary
stubbed by a local canned response. Everything else runs for real: load the dossier,
score the seven factors through the anchor signal, compute the reference three-way,
the departures against the stub's emitted distribution, the tolerance status, the
shootout block, build and validate the full JSON against the schema, and write it to
a pytest tmp path so the test stays hermetic. Two stub cases: a small in-box
departure that is accepted, and a beyond-cap departure that P4 rejects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from architect_wc import pipeline
from architect_wc.llm import anchor, prediction_b, schemas
from architect_wc.llm.weights import EXPERT_LENSES, FACTORS, load_weights

ROOT = Path(__file__).resolve().parents[1]
DOSSIER_PATH = ROOT / "outputs/llm/dossier_GROUP_match64_asof2026-06-25.json"


def _load_committed_dossier() -> dict:
    return json.loads(DOSSIER_PATH.read_text(encoding="utf-8"))["dossier"]


def _dossier_ref() -> dict[str, str]:
    content = DOSSIER_PATH.read_bytes()
    return {
        "path": str(DOSSIER_PATH),
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }


def _factor(name: str, score: int, cite: bool) -> dict:
    return {
        "name": name,
        "expert_lens": EXPERT_LENSES[name],
        "score": score,
        "citations": [f"{name}#0"] if cite else [],
        "insufficient_evidence": score == 0 and not cite,
    }


def _accept_model_call(dossier: dict, config: dict) -> dict:
    """A response with a small nonzero departure that sits inside the tolerance box."""
    weights = load_weights(config)
    scores = dict.fromkeys(FACTORS, 0)
    scores["squad_availability"] = -1
    scores["recent_form"] = -1
    scores["coaching_staff"] = -1
    s = anchor.anchor_signal(scores, weights)
    r_home, r_draw, r_away = anchor.reference_three_way(s)
    # Shift 0.03 of mass home to away: departure_dir 0.06, departure_draw 0, in box.
    q_home, q_away, q_draw = r_home + 0.03, r_away - 0.03, r_draw
    return {
        "factors": [_factor(f, scores[f], scores[f] != 0) for f in FACTORS],
        "emitted_three_way": {"p_home": q_home, "p_draw": q_draw, "p_away": q_away},
        "emitted_lean": anchor.shootout_lean_from_anchor(s) + 0.02,
        "justification": None,
    }


def _reject_model_call(dossier: dict, config: dict) -> dict:
    """A response whose direction departure is beyond the hard cap."""
    weights = load_weights(config)
    scores = dict.fromkeys(FACTORS, 0)  # s = 0, balanced reference
    s = anchor.anchor_signal(scores, weights)
    r_home, r_draw, r_away = anchor.reference_three_way(s)
    # Shift 0.20 of mass home to away: departure_dir 0.40, beyond the 0.30 cap.
    q_home, q_away, q_draw = r_home + 0.20, r_away - 0.20, r_draw
    return {
        "factors": [_factor(f, 0, False) for f in FACTORS],
        "emitted_three_way": {"p_home": q_home, "p_draw": q_draw, "p_away": q_away},
        "emitted_lean": anchor.shootout_lean_from_anchor(s),
        "justification": "deliberately beyond the hard cap, must still be rejected",
    }


def test_accept_case_validates_and_is_accepted(tmp_path) -> None:
    config = pipeline.load_config()
    dossier = _load_committed_dossier()
    document = prediction_b.predict_b(
        dossier,
        config,
        _accept_model_call,
        dossier_ref=_dossier_ref(),
        code_commit="test",
        timestamp="2026-06-25T00:00:00Z",
    )
    schemas.validate_document(document, "prediction_b")
    assert document["validation"]["simplex_ok"] is True
    assert document["validation"]["citations_ok"] is True
    assert document["validation"]["tolerance_ok"] is True
    assert document["validation"]["accepted"] is True
    assert document["validation"]["rejection_reason"] is None
    assert document["tolerance"]["status"] == "within_box"
    assert abs(document["departure"]["dir"] - 0.06) <= 1e-9
    # The scored advance reads off the emitted three-way and the emitted lean.
    emitted = document["emitted_three_way"]
    shootout = document["shootout"]
    expected_advance = emitted["p_home"] + emitted["p_draw"] * shootout["emitted_lean"]
    assert abs(shootout["advance_probability"] - expected_advance) <= 1e-12
    # And not the anchor lean, which differs here.
    assert shootout["emitted_lean"] != shootout["reference_lean"]
    # The dossier identity carried through.
    assert document["match"] == 64
    assert document["nominal_home"] == document["home_team"] == "Uruguay"

    out = tmp_path / "prediction_b_accept.json"
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    assert out.exists()


def test_reject_case_validates_and_is_rejected(tmp_path) -> None:
    config = pipeline.load_config()
    dossier = _load_committed_dossier()
    document = prediction_b.predict_b(
        dossier,
        config,
        _reject_model_call,
        dossier_ref=_dossier_ref(),
        code_commit="test",
        timestamp="2026-06-25T00:00:00Z",
    )
    schemas.validate_document(document, "prediction_b")
    # Beyond the cap: rejected regardless of the justification.
    assert document["tolerance"]["status"] == "rejected"
    assert document["validation"]["tolerance_ok"] is False
    assert document["validation"]["accepted"] is False
    assert document["validation"]["rejection_reason"]
    assert "rejected" in document["validation"]["rejection_reason"]
    # The emitted distribution is still a valid simplex; only tolerance failed.
    assert document["validation"]["simplex_ok"] is True
    assert abs(document["departure"]["dir"] - 0.40) <= 1e-9
    # The scored advance reads off the emitted three-way and the emitted lean.
    emitted = document["emitted_three_way"]
    shootout = document["shootout"]
    expected_advance = emitted["p_home"] + emitted["p_draw"] * shootout["emitted_lean"]
    assert abs(shootout["advance_probability"] - expected_advance) <= 1e-12

    out = tmp_path / "prediction_b_reject.json"
    out.write_text(json.dumps(document, indent=2), encoding="utf-8")
    assert out.exists()
