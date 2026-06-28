"""Verification gate: the model-call boundary, its prompts, and its failure policy.

Offline and hermetic. The boundary invoke_structured is never called with a real
client here; the policy is exercised with a stub invoke_one, and the prompt builders
are exercised on the committed dossier and canned A and B documents. This proves the
transient retry and the one clean re-roll, the mechanical-failure classification, and
the two prompt-exclusion guarantees, all without the network.
"""

from __future__ import annotations

import json

import pytest

from architect_wc.llm import anchor, model_call
from architect_wc.llm.weights import FACTORS

# Tokens that must never appear in a filled prompt. The anchor mapping parameters and
# the tolerance box and hard cap are the secrets the model must not see, so it cannot
# reverse-engineer its scores or clamp to the cap.
ANCHOR_PARAM_TOKENS = ["0.62", "0.67", "0.21"]
TOLERANCE_BOX_TOKENS = [
    "0.15",
    "departure_dir_tolerance",
    "DEPARTURE_DIR_TOLERANCE",
    "lean_tolerance",
    "hard cap",
    "HARD_CAP",
    "tolerance box",
]


def _valid_b_output() -> dict:
    """A schema-valid Prediction B model output: seven factors, a simplex, a lean."""
    p_home, p_draw, p_away = anchor.reference_three_way(0.0)
    return {
        "factors": [
            {
                "name": name,
                "score": 0,
                "reasoning": "no admissible edge for this factor",
                "citations": [],
                "insufficient_evidence": True,
            }
            for name in FACTORS
        ],
        "emitted_three_way": {"p_home": p_home, "p_draw": p_draw, "p_away": p_away},
        "emitted_lean": anchor.shootout_lean_from_anchor(0.0),
        "justification": None,
    }


def _raw(text: str, stop: str = "end_turn") -> model_call.RawResponse:
    return model_call.RawResponse(
        text=text,
        stop_reason=stop,
        usage={"input_tokens": 10, "output_tokens": 10},
        message_id="msg_test",
        request_id="req_test",
        model="claude-opus-4-8",
        effort="xhigh",
    )


def _hooks() -> model_call.PolicyHooks:
    return model_call.PolicyHooks(
        authorize=lambda: None,
        record=lambda raw: None,
        sleep=lambda seconds: None,
    )


# --- the failure policy ---------------------------------------------------------


def test_transient_failures_retry_then_succeed() -> None:
    attempts = {"n": 0}

    def invoke_one() -> model_call.RawResponse:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise model_call.TransientCallError("network blip")
        return _raw(json.dumps(_valid_b_output()))

    parsed = model_call.call_with_policy(
        invoke_one,
        output_kind="prediction_b_output",
        require_seven_factors=True,
        hooks=_hooks(),
    )
    assert attempts["n"] == 3
    assert len(parsed["factors"]) == 7


def test_transient_failures_halt_after_three_attempts() -> None:
    def invoke_one() -> model_call.RawResponse:
        raise model_call.TransientCallError("down")

    with pytest.raises(model_call.PolicyHalt) as caught:
        model_call.call_with_policy(
            invoke_one,
            output_kind="prediction_b_output",
            require_seven_factors=True,
            hooks=_hooks(),
        )
    assert caught.value.kind == "transient"


def test_mechanical_failure_gets_one_clean_reroll() -> None:
    attempts = {"n": 0}

    def invoke_one() -> model_call.RawResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _raw("this is not json at all")
        return _raw(json.dumps(_valid_b_output()))

    parsed = model_call.call_with_policy(
        invoke_one,
        output_kind="prediction_b_output",
        require_seven_factors=True,
        hooks=_hooks(),
    )
    assert attempts["n"] == 2  # one broken call, then one clean re-roll.
    assert len(parsed["factors"]) == 7


def test_mechanical_failure_halts_when_reroll_also_broken() -> None:
    attempts = {"n": 0}

    def invoke_one() -> model_call.RawResponse:
        attempts["n"] += 1
        return _raw("still not json")

    with pytest.raises(model_call.PolicyHalt) as caught:
        model_call.call_with_policy(
            invoke_one,
            output_kind="prediction_b_output",
            require_seven_factors=True,
            hooks=_hooks(),
        )
    assert caught.value.kind == "mechanical"
    assert attempts["n"] == 2  # the call and exactly one re-roll, then halt.


def test_billed_calls_are_recorded_only_on_a_returned_response() -> None:
    # A transient exception produced no billable response; only the returned one is
    # recorded, so the per-tie cap and the dollar ceiling count real billed calls.
    recorded: list[model_call.RawResponse] = []
    attempts = {"n": 0}

    def invoke_one() -> model_call.RawResponse:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise model_call.TransientCallError("blip")
        return _raw(json.dumps(_valid_b_output()))

    hooks = model_call.PolicyHooks(
        authorize=lambda: None,
        record=recorded.append,
        sleep=lambda seconds: None,
    )
    model_call.call_with_policy(
        invoke_one,
        output_kind="prediction_b_output",
        require_seven_factors=True,
        hooks=hooks,
    )
    assert len(recorded) == 1  # the transient failure was not billed.


# --- mechanical-failure classification ------------------------------------------


def test_refusal_is_mechanical() -> None:
    raw = _raw(json.dumps(_valid_b_output()), stop="refusal")
    with pytest.raises(model_call.MechanicalError):
        model_call.validate_output(
            raw, "prediction_b_output", require_seven_factors=True
        )


def test_truncation_is_mechanical() -> None:
    raw = _raw(json.dumps(_valid_b_output()), stop="max_tokens")
    with pytest.raises(model_call.MechanicalError):
        model_call.validate_output(
            raw, "prediction_b_output", require_seven_factors=True
        )


def test_off_schema_output_is_mechanical() -> None:
    bad = _valid_b_output()
    del bad["emitted_lean"]  # required by the output schema.
    raw = _raw(json.dumps(bad))
    with pytest.raises(model_call.MechanicalError):
        model_call.validate_output(
            raw, "prediction_b_output", require_seven_factors=True
        )


def test_wrong_factor_count_is_mechanical() -> None:
    bad = _valid_b_output()
    bad["factors"] = bad["factors"][:6]  # six factors, not seven.
    raw = _raw(json.dumps(bad))
    with pytest.raises(model_call.MechanicalError):
        model_call.validate_output(
            raw, "prediction_b_output", require_seven_factors=True
        )


def test_missing_per_factor_reasoning_is_mechanical() -> None:
    # Per-factor reasoning is required in the forced output, so a B output missing it
    # is off-schema and rejected, the same as any other mechanical failure.
    bad = _valid_b_output()
    for factor in bad["factors"]:
        del factor["reasoning"]
    raw = _raw(json.dumps(bad))
    with pytest.raises(model_call.MechanicalError):
        model_call.validate_output(
            raw, "prediction_b_output", require_seven_factors=True
        )


def test_unparseable_output_is_mechanical() -> None:
    with pytest.raises(model_call.MechanicalError):
        model_call.parse_output("no json object here")


# --- prompt building and the exclusion guarantees -------------------------------

DOSSIER = {
    "round": "R32",
    "match": 73,
    "home_team": "Uruguay",
    "away_team": "Spain",
    "as_of_date": "2026-06-25",
    "factors": {
        "recent_form": [{"claim": "won two", "source": "espn.com", "team": "Uruguay"}]
    },
}

A_DOC = {
    "three_way": {"p_home": 0.40, "p_draw": 0.30, "p_away": 0.30},
    "shootout_lean": 0.55,
    "advance_probability": 0.565,
    "drivers": {"elo_diff": 20.0, "ability_diff": 0.1, "squad_value_adjustment": 5.0},
}

B_DOC = {
    "factors": [
        {
            "name": name,
            "expert_lens": "x",
            "score": 1 if name == "recent_form" else 0,
            "reasoning": f"reason-for-{name}",
            "citations": ["recent_form#0"] if name == "recent_form" else [],
            "insufficient_evidence": False,
        }
        for name in FACTORS
    ],
    "anchor_signal": 0.2,
    "reference_three_way": {"p_home": 0.31, "p_draw": 0.40, "p_away": 0.29},
    "emitted_three_way": {"p_home": 0.33, "p_draw": 0.40, "p_away": 0.27},
    "departure": {"dir": 0.04, "draw": 0.0},
    "tolerance": {"status": "within_box", "justification": "settled"},
    "shootout": {
        "reference_lean": 0.51,
        "emitted_lean": 0.52,
        "lean_departure": 0.01,
        "advance_probability": 0.54,
    },
    "validation": {
        "simplex_ok": True,
        "citations_ok": True,
        "tolerance_ok": True,
        "accepted": True,
        "rejection_reason": None,
    },
}


def test_b_prompt_fills_tokens_and_excludes_anchor_and_tolerance() -> None:
    system, user = model_call.build_b_prompt(DOSSIER)
    combined = system + user
    assert "{HOME_TEAM}" not in combined and "{AWAY_TEAM}" not in combined
    assert "Uruguay" in system and "Spain" in system
    assert "recent_form#0" in user  # citable finding id present.
    for token in ANCHOR_PARAM_TOKENS + TOLERANCE_BOX_TOKENS:
        assert token not in combined, f"B prompt leaked {token!r}"
    # B never sees Prediction A, by construction.
    assert "drivers" not in combined and "elo_diff" not in combined


def test_c_prompt_excludes_tolerance_box_and_hard_cap_and_anchor() -> None:
    pool = {
        "three_way": {"p_home": 0.36, "p_draw": 0.35, "p_away": 0.29},
        "lean": 0.53,
        "advance_probability": 0.55,
    }
    system, user = model_call.build_c_prompt(DOSSIER, A_DOC, B_DOC, pool)
    combined = system + user
    assert "{HOME_TEAM}" not in combined and "{AWAY_TEAM}" not in combined
    for token in ANCHOR_PARAM_TOKENS + TOLERANCE_BOX_TOKENS:
        assert token not in combined, f"C prompt leaked {token!r}"
    # C sees A's drivers and B's scores and the pool, the inputs it reconciles.
    assert "drivers" in user and "elo_diff" in user
    assert "THE POOL" in user
    # C is shown B's per-factor reasoning, so it reconciles against B's reasons.
    assert "reason-for-recent_form" in user


def test_b_view_for_c_carries_reasoning_and_excludes_anchor_and_tolerance() -> None:
    view = model_call.b_view_for_c(B_DOC)
    # The forecast and the per-factor reasoning are present.
    assert "factors" in view and "emitted_three_way" in view and "emitted_lean" in view
    assert all(factor["reasoning"] for factor in view["factors"])
    assert view["factors"][0]["reasoning"] == "reason-for-squad_availability"
    # The harness's audit of B is not: no anchor signal, reference, departures,
    # tolerance status, or validation reach C through B.
    for hidden in (
        "anchor_signal",
        "reference_three_way",
        "departure",
        "tolerance",
        "validation",
    ):
        assert hidden not in view, f"b_view_for_c leaked {hidden!r}"


def test_max_call_cost_is_a_positive_offline_estimate() -> None:
    cost = model_call.max_call_cost("system text", "user text", 16000)
    assert cost > 0.0
