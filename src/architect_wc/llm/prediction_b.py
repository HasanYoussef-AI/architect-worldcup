"""Prediction B, the LLM analyst per-match prediction machine (Phase 2).

Offline path. The model supplies only the factor scores, the citations, the
insufficient-evidence flags, the emitted three-way, the emitted shootout lean, and
any justification. Everything else is computed by code from the frozen anchor
mapping: the anchor signal, the reference three-way, the two departures, the
tolerance status, the shootout block, and the validation verdict. The model call is
the only stubbed boundary; build_prediction_b takes the model's response rather than
calling the API, and predict_b takes an injected model_call, so there is no SDK, no
network, and no key in this module.

This package consumes the math spine, never the reverse. Prediction A's shootout
lean lives in the spine (simulate.shootout_lean_from_elo); Prediction B's lives in
anchor (shootout_lean_from_anchor). The two are distinct and never collide.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from architect_wc.llm import anchor, validity
from architect_wc.llm.weights import EXPERT_LENSES, FACTORS, load_weights

SCHEMA_VERSION = "1.0.0"


def factor_scores(factors: list[dict[str, Any]]) -> dict[str, int]:
    """Map the model's factor list to a name to score dict."""
    return {factor["name"]: int(factor["score"]) for factor in factors}


def citations_ok(factors: list[dict[str, Any]]) -> bool:
    """True if every nonzero-scored factor cites at least one dossier finding id.

    A factor scored zero may be uncited (neutral or flagged insufficient evidence);
    a nonzero factor with no citation is unsupported and fails the check.
    """
    for factor in factors:
        if int(factor.get("score", 0)) != 0 and not factor.get("citations"):
            return False
    return True


def _ordered(factors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the factors in the canonical taxonomy order, raising on a wrong set."""
    by_name = {factor["name"]: factor for factor in factors}
    if set(by_name) != set(FACTORS) or len(factors) != len(FACTORS):
        raise ValueError(
            f"Prediction B must score exactly the seven factors once each; got "
            f"{[factor.get('name') for factor in factors]}."
        )
    return [by_name[name] for name in FACTORS]


def build_prediction_b(
    dossier: dict[str, Any],
    model_response: dict[str, Any],
    config: dict[str, Any],
    *,
    dossier_ref: dict[str, str],
    code_commit: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the full Prediction B document from a dossier and the model's response.

    Code computes the anchor signal, the reference three-way, the departures, the
    tolerance status, the shootout block, and the validation verdict; the model
    supplied only the scores, citations, emitted three-way, emitted lean, and any
    justification. No model call here; model_response is injected.
    """
    from architect_wc import artifact

    weights = load_weights(config)
    factors = _ordered(model_response["factors"])
    scores = factor_scores(factors)
    s = anchor.anchor_signal(scores, weights)

    r_home, r_draw, r_away = anchor.reference_three_way(s)
    emitted = model_response["emitted_three_way"]
    q_home = float(emitted["p_home"])
    q_draw = float(emitted["p_draw"])
    q_away = float(emitted["p_away"])
    departure_dir, departure_draw = anchor.anchor_departure(
        (q_home, q_draw, q_away), (r_home, r_draw, r_away)
    )

    reference_lean = anchor.shootout_lean_from_anchor(s)
    emitted_lean = float(model_response["emitted_lean"])
    lean_departure = emitted_lean - reference_lean
    justification = model_response.get("justification")
    status = anchor.classify_tolerance(
        departure_dir, departure_draw, lean_departure, bool(justification)
    )
    # The advance is B's scored prediction, so it reads off the emitted three-way and
    # the emitted lean, not the anchor lean. The anchor (reference) lean stays the
    # reference that lean_departure is measured against, mirroring how
    # reference_three_way anchors the three-way departures; but the scored advance
    # must reflect what B actually predicted, including any within-tolerance,
    # shootout-specific judgment in its emitted lean.
    advance = anchor.advance_probability(q_home, q_draw, emitted_lean)

    simplex_ok = validity.three_way_sums_to_one(q_home, q_draw, q_away)
    cited_ok = citations_ok(factors)
    tolerance_ok = status != anchor.STATUS_REJECTED
    accepted = simplex_ok and cited_ok and tolerance_ok
    reasons = []
    if not simplex_ok:
        reasons.append("emitted three-way is not a valid distribution")
    if not cited_ok:
        reasons.append("a nonzero factor cites no dossier finding")
    if not tolerance_ok:
        reasons.append(f"tolerance status {status}")
    rejection_reason = "; ".join(reasons) if reasons else None

    llm = config.get("llm", {}) or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "round": dossier["round"],
        "match": int(dossier["match"]),
        "home_team": dossier["home_team"],
        "away_team": dossier["away_team"],
        "nominal_home": dossier["home_team"],
        "cutoff": dossier["as_of_date"],
        "dossier_ref": {
            "path": dossier_ref["path"],
            "content_sha256": dossier_ref["content_sha256"],
        },
        "factors": [
            {
                "name": factor["name"],
                "expert_lens": factor.get(
                    "expert_lens", EXPERT_LENSES.get(factor["name"], "")
                ),
                "score": int(factor["score"]),
                # The model's per-factor reasoning, carried into the committed
                # artifact so it reaches Prediction C, not only the thinking trace.
                "reasoning": str(factor.get("reasoning", "")),
                "citations": list(factor.get("citations", [])),
                "insufficient_evidence": bool(
                    factor.get("insufficient_evidence", False)
                ),
            }
            for factor in factors
        ],
        "anchor_signal": s,
        "reference_three_way": {"p_home": r_home, "p_draw": r_draw, "p_away": r_away},
        "emitted_three_way": {"p_home": q_home, "p_draw": q_draw, "p_away": q_away},
        "departure": {"dir": departure_dir, "draw": departure_draw},
        "tolerance": {"status": status, "justification": justification},
        "shootout": {
            "reference_lean": reference_lean,
            "emitted_lean": emitted_lean,
            "lean_departure": lean_departure,
            "advance_probability": advance,
        },
        "provenance": {
            "model": llm.get("model", "claude-opus-4-8"),
            "effort": llm.get("effort_prediction", "xhigh"),
            "thinking": "adaptive",
            "code_commit": code_commit or artifact.get_git_sha(),
            "timestamp": timestamp,
        },
        "validation": {
            "simplex_ok": simplex_ok,
            "citations_ok": cited_ok,
            "tolerance_ok": tolerance_ok,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
        },
    }


def predict_b(
    dossier: dict[str, Any],
    config: dict[str, Any],
    model_call: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    *,
    dossier_ref: dict[str, str],
    code_commit: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Run the prediction: call the injected model boundary, then build the document.

    model_call(dossier, config) returns the model's emitted response. Offline it is a
    local stub; in production it is the Anthropic structured-output call (wired in a
    later step, not here). This keeps the only network boundary in one place.
    """
    model_response = model_call(dossier, config)
    return build_prediction_b(
        dossier,
        model_response,
        config,
        dossier_ref=dossier_ref,
        code_commit=code_commit,
        timestamp=timestamp,
    )
