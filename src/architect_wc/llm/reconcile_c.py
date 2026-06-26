"""Prediction C, the reconciliation layer, and the A-B pool (Phase 3).

Offline path. C reconciles A and B at the distribution level. It does not select A
or B and it does not re-score the seven factors: it emits its own 90-minute
three-way and its own shootout lean, anchored to an equal-weight linear pool of A
and B. The pool is pre-registered: reference three-way is the componentwise average
of A's and B's three-ways, reference lean is the average of the two leans, equal
weights, never fitted, a linear pool not a log pool.

C is governed by the same tolerance box B uses, reusing anchor.classify_tolerance and
its constants unforked; the only difference is the reference. C's reference is the
A-B pool, B's reference is its own anchor mapping. Departures are expected for C,
since resolving disagreement is its job, so the soft box is a cite-this-move trigger
and the hard cap is the real gate. C's advance reads off C's emitted values, the same
formula B uses.

C's citation rule is the analog of B's nonzero-score-must-cite: every material move
off the pool must cite one of exactly three sources, an element of A's reasoning, an
element of B's reasoning, or a dossier finding id. C introduces no new world-facts.

The same pool object is exposed as a first-class scoreable line, pool_ab, so it can
be scored through the one RPS scorer alongside A, B, and C.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from architect_wc.llm import anchor, validity

SCHEMA_VERSION = "1.0.0"

# The exactly-three citation sources C may cite. C introduces no new world-facts.
CITATION_SOURCES = ("a_reasoning", "b_reasoning", "dossier_finding_id")


def pool_ab(
    a_three_way: dict[str, float],
    b_three_way: dict[str, float],
    a_lean: float,
    b_lean: float,
) -> dict[str, Any]:
    """The equal-weight linear pool of A and B, a first-class scoreable line.

    Componentwise average of the two three-ways and the two leans, pre-registered
    equal weights, never fitted, never tuned. Returns the three-way, the lean, and
    the advance probability, the last from the pool's own values so the line is
    scoreable on its own through the same RPS path as A, B, and C.
    """
    three_way = {
        "p_home": 0.5 * float(a_three_way["p_home"])
        + 0.5 * float(b_three_way["p_home"]),
        "p_draw": 0.5 * float(a_three_way["p_draw"])
        + 0.5 * float(b_three_way["p_draw"]),
        "p_away": 0.5 * float(a_three_way["p_away"])
        + 0.5 * float(b_three_way["p_away"]),
    }
    lean = 0.5 * float(a_lean) + 0.5 * float(b_lean)
    advance = anchor.advance_probability(three_way["p_home"], three_way["p_draw"], lean)
    return {"three_way": three_way, "lean": lean, "advance_probability": advance}


def pool_from_predictions(
    prediction_a: dict[str, Any], prediction_b: dict[str, Any]
) -> dict[str, Any]:
    """The A-B pool from full Prediction A and Prediction B documents."""
    return pool_ab(
        prediction_a["three_way"],
        prediction_b["emitted_three_way"],
        prediction_a["shootout_lean"],
        prediction_b["shootout"]["emitted_lean"],
    )


def citations_ok(status: str, citations: list[dict[str, Any]]) -> bool:
    """C's citation rule.

    Every citation must tag one of exactly three allowed sources, and a material move
    off the pool, any departure outside the soft box, must cite at least one. A move
    that stays within the box is noise-level and needs no citation.
    """
    for citation in citations:
        if citation.get("source") not in CITATION_SOURCES:
            return False
    if status != anchor.STATUS_WITHIN_BOX and not citations:
        return False
    return True


def build_prediction_c(
    dossier: dict[str, Any],
    prediction_a: dict[str, Any],
    prediction_b: dict[str, Any],
    model_response: dict[str, Any],
    config: dict[str, Any],
    *,
    dossier_ref: dict[str, str],
    a_ref: dict[str, str],
    b_ref: dict[str, str],
    code_commit: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the full Prediction C document from the dossier, A, B, and C's response.

    Code computes the A-B pool, the departures of C's emitted distribution from the
    pool, the tolerance status (reusing anchor.classify_tolerance unforked), the
    shootout block, and the validation verdict. The model supplied only C's emitted
    three-way, emitted lean, citations, and any justification. No model call here.
    """
    from architect_wc import artifact

    pool = pool_from_predictions(prediction_a, prediction_b)
    pool_three_way = pool["three_way"]
    pool_lean = pool["lean"]

    emitted = model_response["emitted_three_way"]
    q_home = float(emitted["p_home"])
    q_draw = float(emitted["p_draw"])
    q_away = float(emitted["p_away"])
    departure_dir, departure_draw = anchor.anchor_departure(
        (q_home, q_draw, q_away),
        (pool_three_way["p_home"], pool_three_way["p_draw"], pool_three_way["p_away"]),
    )

    emitted_lean = float(model_response["emitted_lean"])
    lean_departure = emitted_lean - pool_lean
    justification = model_response.get("justification")
    status = anchor.classify_tolerance(
        departure_dir, departure_draw, lean_departure, bool(justification)
    )
    advance = anchor.advance_probability(q_home, q_draw, emitted_lean)

    citations = list(model_response.get("citations", []))
    simplex_ok = validity.three_way_sums_to_one(q_home, q_draw, q_away)
    cited_ok = citations_ok(status, citations)
    tolerance_ok = status != anchor.STATUS_REJECTED
    accepted = simplex_ok and cited_ok and tolerance_ok
    reasons = []
    if not simplex_ok:
        reasons.append("emitted three-way is not a valid distribution")
    if not cited_ok:
        reasons.append("a material move off the pool is uncited or cites a bad source")
    if not tolerance_ok:
        reasons.append(f"tolerance status {status}")
    rejection_reason = "; ".join(reasons) if reasons else None

    llm = config.get("llm", {}) or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "prediction": "C",
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
        "a_ref": {"path": a_ref["path"], "content_sha256": a_ref["content_sha256"]},
        "b_ref": {"path": b_ref["path"], "content_sha256": b_ref["content_sha256"]},
        "pool": pool,
        "emitted_three_way": {"p_home": q_home, "p_draw": q_draw, "p_away": q_away},
        "departure": {"dir": departure_dir, "draw": departure_draw},
        "tolerance": {"status": status, "justification": justification},
        "shootout": {
            "pool_lean": pool_lean,
            "emitted_lean": emitted_lean,
            "lean_departure": lean_departure,
            "advance_probability": advance,
        },
        "citations": [
            {"source": citation["source"], "ref": citation["ref"]}
            for citation in citations
        ],
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


def predict_c(
    dossier: dict[str, Any],
    prediction_a: dict[str, Any],
    prediction_b: dict[str, Any],
    config: dict[str, Any],
    model_call: Callable[..., dict[str, Any]],
    *,
    dossier_ref: dict[str, str],
    a_ref: dict[str, str],
    b_ref: dict[str, str],
    code_commit: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Run C: call the injected model boundary (which sees the dossier plus the full
    structured output of both A and B), then build the document.

    model_call(dossier, prediction_a, prediction_b, config) returns C's emitted
    response. Offline it is a local stub; in production it is the Anthropic call. C's
    loader is allowed to read both A and B; B's loader is not allowed to read A.
    """
    model_response = model_call(dossier, prediction_a, prediction_b, config)
    return build_prediction_c(
        dossier,
        prediction_a,
        prediction_b,
        model_response,
        config,
        dossier_ref=dossier_ref,
        a_ref=a_ref,
        b_ref=b_ref,
        code_commit=code_commit,
        timestamp=timestamp,
    )
