"""Verification gate: the live two-call orchestrator, offline and hermetic.

The only thing stubbed is the model boundary, returning canned structured responses;
no real Anthropic client is ever constructed. Everything else runs for real: the
prompt building, the prediction documents and their schema validation, the caps, the
resume logic, and the commit ordering, with commits captured by a fake committer and
artifacts written to a pytest tmp path.

This proves the protocol contract: the dossier commits before the predictions, the
prompt templates fill and B's filled prompt carries no anchor parameters, the
provenance is complete, the dollar ceiling and the per-tie call cap each halt, a
substantive rule violation halts without regenerating, the re-research guard refuses
when a committed dossier exists, resume continues from B when only the dossier is
committed, and a missing key halts cleanly.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from architect_wc import artifact, pipeline
from architect_wc.llm import anchor, live, model_call, reconcile_c, schemas
from architect_wc.llm.weights import FACTORS

ROOT = Path(__file__).resolve().parents[1]
DOSSIER_PATH = ROOT / "outputs/llm/dossier_GROUP_match64_asof2026-06-25.json"

NOW = datetime(2026, 6, 26, tzinfo=UTC)
KICKOFF = datetime(2027, 1, 1, tzinfo=UTC)
DUMMY_KEY = "offline-stub-not-a-real-key"

# A canned analytic Prediction A. Its three-way and lean drive the A-B pool that
# Prediction C is anchored to; the test computes the matching C output from them.
A_THREE_WAY = {"p_home": 0.40, "p_draw": 0.30, "p_away": 0.30}
A_LEAN = 0.55


def _committed_dossier() -> dict:
    return json.loads(DOSSIER_PATH.read_text(encoding="utf-8"))["dossier"]


def _tie(dossier: dict) -> dict:
    return {
        "round": dossier["round"],
        "match": int(dossier["match"]),
        "home_team": dossier["home_team"],
        "away_team": dossier["away_team"],
        "kickoff": KICKOFF,
    }


def _b_factors() -> list[dict]:
    """Seven neutral factors, each with a distinctive per-factor reasoning string."""
    return [
        {
            "name": name,
            "score": 0,
            "reasoning": f"REASONING-{name}",
            "citations": [],
            "insufficient_evidence": True,
        }
        for name in FACTORS
    ]


def _accepted_b_output() -> dict:
    """An all-neutral B output: emitted equals the reference, so it is in box."""
    p_home, p_draw, p_away = anchor.reference_three_way(0.0)
    return {
        "factors": _b_factors(),
        "emitted_three_way": {"p_home": p_home, "p_draw": p_draw, "p_away": p_away},
        "emitted_lean": anchor.shootout_lean_from_anchor(0.0),
        "justification": None,
    }


def _rejected_b_output() -> dict:
    """A coherent B output that is beyond the hard cap: a substantive violation."""
    p_home, p_draw, p_away = anchor.reference_three_way(0.0)
    return {
        "factors": _b_factors(),
        # Shift 0.20 home to away: departure_dir 0.40, beyond the 0.30 hard cap.
        "emitted_three_way": {
            "p_home": p_home + 0.20,
            "p_draw": p_draw,
            "p_away": p_away - 0.20,
        },
        "emitted_lean": anchor.shootout_lean_from_anchor(0.0),
        "justification": "deliberately beyond the cap; must still be rejected",
    }


def _accepted_c_output() -> dict:
    """A C output sitting exactly on the A-B pool, so it is in box."""
    b = _accepted_b_output()
    pool = reconcile_c.pool_ab(
        A_THREE_WAY, b["emitted_three_way"], A_LEAN, b["emitted_lean"]
    )
    return {
        "emitted_three_way": pool["three_way"],
        "emitted_lean": pool["lean"],
        "citations": [],
        "justification": None,
    }


def _raw(output: dict, model: str, effort: str) -> model_call.RawResponse:
    return model_call.RawResponse(
        text=json.dumps(output),
        stop_reason="end_turn",
        usage={
            "input_tokens": 200,
            "output_tokens": 80,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
        message_id="msg_offline",
        request_id="req_offline",
        model=model,
        effort=effort,
    )


def _make_invoke(records: list, b_output: dict, c_output: dict):
    """A stub model boundary that dispatches on the system prompt and records calls."""

    def invoke(
        client,
        *,
        model,
        effort,
        system,
        user_content,
        schema,
        max_tokens,
        use_prompt_cache,
    ):
        records.append({"system": system, "user": user_content, "schema": schema})
        if "panel of seven" in system:
            output = b_output
        elif "reconciliation analyst" in system:
            output = c_output
        else:  # pragma: no cover - guards an unexpected prompt
            raise AssertionError("unexpected system prompt")
        return _raw(output, model, effort)

    return invoke


def _make_research(dossier: dict, output_dir: Path, cost: float, calls: list):
    """A stub research seam that writes the dossier and returns its meta, no network."""

    def research_fn(
        config,
        fixture,
        as_of_date,
        round_code,
        *,
        client=None,
        rehearsal=False,
        dossiers_dir=None,
        **_kwargs,
    ):
        calls.append(fixture)
        path = artifact.write_dossier(
            dossier,
            meta={"cost": {"total_usd": cost}},
            rehearsal=rehearsal,
            dossiers_dir=dossiers_dir or output_dir,
        )
        return {
            "dossier": dossier,
            "meta": {"dossier_path": str(path), "cost": {"total_usd": cost}},
        }

    return research_fn


def _build_a(config, fixture, as_of_date, round_code) -> dict:
    """A stub analytic A, valid against the Prediction A schema."""
    return {
        "schema_version": "1.0.0",
        "prediction": "A",
        "round": round_code,
        "match": int(fixture["match"]),
        "home_team": fixture["home_team"],
        "away_team": fixture["away_team"],
        "nominal_home": fixture["home_team"],
        "cutoff": as_of_date,
        "committed_at": None,
        "three_way": dict(A_THREE_WAY),
        "shootout_lean": A_LEAN,
        "advance_probability": A_THREE_WAY["p_home"] + A_THREE_WAY["p_draw"] * A_LEAN,
        "drivers": {
            "elo_diff": 20.0,
            "ability_diff": 0.1,
            "squad_value_adjustment": 5.0,
        },
        "seed": 42,
        "provenance": {
            "source": "test-stub",
            "model_version": "0.0.0",
            "code_commit": "test",
            "timestamp": None,
        },
    }


class _FakeCommitter:
    """Records the ordered runtime commits instead of running git."""

    def __init__(self) -> None:
        self.commits: list[dict] = []

    def __call__(self, paths: list[Path], message: str) -> None:
        self.commits.append(
            {"paths": [str(path) for path in paths], "message": message}
        )


class _NoClient:
    """A stub client factory whose use would mean a real client leaked into a test."""


def _no_real_client(monkeypatch) -> dict:
    """Fail loudly if a real Anthropic client is constructed, proving it never is."""
    import anthropic

    state = {"constructed": False}

    def boom(*_args, **_kwargs):
        state["constructed"] = True
        raise AssertionError("a real Anthropic client was constructed in a test")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    return state


def _run(
    config,
    tie,
    *,
    invoke,
    research_fn,
    calls_research,
    committer,
    committed,
    tmp_path,
    monkeypatch,
    rehearsal=False,
    force_research=False,
    set_key=True,
):
    if set_key:
        monkeypatch.setenv(live.API_KEY_ENV, DUMMY_KEY)
    else:
        monkeypatch.delenv(live.API_KEY_ENV, raising=False)
    return live.run_tie(
        config,
        tie,
        str(tie_as_of(tie)),
        now=NOW,
        invoke=invoke,
        research_fn=research_fn,
        build_a=_build_a,
        client_factory=_NoClient,
        committer=committer,
        committed=committed,
        output_dir=tmp_path,
        logs_dir=tmp_path / "logs",
        session_log_path=tmp_path / "session-log.md",
        repo_root=ROOT,
        rehearsal=rehearsal,
        force_research=force_research,
    )


def tie_as_of(_tie: dict) -> str:
    return "2026-06-25"


# --- the full forward-only run --------------------------------------------------


def test_full_run_commits_in_order_and_validates(tmp_path, monkeypatch) -> None:
    state = _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)
    config = copy.deepcopy(pipeline.load_config())
    records: list = []
    research_calls: list = []
    invoke = _make_invoke(records, _accepted_b_output(), _accepted_c_output())
    research_fn = _make_research(dossier, tmp_path, cost=0.5, calls=research_calls)
    committer = _FakeCommitter()

    result = _run(
        config,
        tie,
        invoke=invoke,
        research_fn=research_fn,
        calls_research=research_calls,
        committer=committer,
        committed=lambda path: False,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert result.status == "completed"
    assert state["constructed"] is False  # no real client was ever constructed.
    assert len(research_calls) == 1
    assert result.accepted == {"b": True, "c": True}

    # Three runtime commits, in exact order: dossier, then predictions, then the log.
    assert len(committer.commits) == 3
    assert "dossier" in committer.commits[0]["message"].lower()
    assert "predictions a, b, c" in committer.commits[1]["message"].lower()
    assert "log the live run" in committer.commits[2]["message"].lower()
    # The dossier file is committed in commit one, before any prediction file.
    assert any("dossier_GROUP_match64" in p for p in committer.commits[0]["paths"])
    assert any("prediction_b_" in p for p in committer.commits[1]["paths"])
    assert any("prediction_a_" in p for p in committer.commits[1]["paths"])
    assert any("prediction_c_" in p for p in committer.commits[1]["paths"])
    assert any("session-log" in p for p in committer.commits[2]["paths"])

    # Both predictions validate against their published schemas.
    b_env = json.loads(Path(result.paths["b"]).read_text(encoding="utf-8"))
    c_env = json.loads(Path(result.paths["c"]).read_text(encoding="utf-8"))
    schemas.validate_document(b_env["document"], "prediction_b")
    schemas.validate_document(c_env["document"], "prediction_c")

    # The committed B artifact carries the seven per-factor reasoning strings.
    b_factors = b_env["document"]["factors"]
    assert len(b_factors) == 7
    assert all(factor["reasoning"] for factor in b_factors)
    assert {factor["reasoning"] for factor in b_factors} == {
        f"REASONING-{name}" for name in FACTORS
    }

    # Provenance is complete and the committed envelope references the raw log.
    prov = b_env["document"]["provenance"]
    assert (
        prov["model"] and prov["effort"] == "xhigh" and prov["thinking"] == "adaptive"
    )
    assert prov["code_commit"] and prov["timestamp"]
    assert b_env["meta"]["raw_log"]
    assert Path(b_env["meta"]["raw_log"]).exists()

    # The boundary was the stub, called for B then C.
    assert len(records) == 2
    assert "panel of seven" in records[0]["system"]
    assert "reconciliation analyst" in records[1]["system"]


def test_b_prompt_excludes_anchor_and_c_prompt_excludes_tolerance(
    tmp_path, monkeypatch
) -> None:
    _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)
    config = copy.deepcopy(pipeline.load_config())
    records: list = []
    research_calls: list = []
    invoke = _make_invoke(records, _accepted_b_output(), _accepted_c_output())
    research_fn = _make_research(dossier, tmp_path, cost=0.5, calls=research_calls)

    _run(
        config,
        tie,
        invoke=invoke,
        research_fn=research_fn,
        calls_research=research_calls,
        committer=_FakeCommitter(),
        committed=lambda path: False,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    b_prompt = records[0]["system"] + records[0]["user"]
    c_prompt = records[1]["system"] + records[1]["user"]
    forbidden = ["0.62", "0.67", "0.21", "tolerance", "within_box", "hard cap", "0.15"]
    for token in forbidden:
        assert token not in b_prompt, f"B prompt leaked {token!r}"
        assert token not in c_prompt, f"C prompt leaked {token!r}"
    # B never sees Prediction A.
    assert "drivers" not in b_prompt and "elo_diff" not in b_prompt
    # C is shown A's drivers, B's scores and per-factor reasoning, and the pool.
    assert "drivers" in c_prompt and "THE POOL" in c_prompt
    assert "REASONING-recent_form" in c_prompt


# --- the two circuit breakers ---------------------------------------------------


def test_dollar_ceiling_halts_before_the_b_call(tmp_path, monkeypatch) -> None:
    _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)
    config = copy.deepcopy(pipeline.load_config())
    # Research passes (its estimate is zeroed and its actual cost is zero), then B's
    # maximum possible cost crosses a tiny ceiling.
    config["llm"]["research_max_cost_estimate"] = 0.0
    config["llm"]["session_dollar_ceiling"] = 0.0001
    research_calls: list = []
    invoke = _make_invoke([], _accepted_b_output(), _accepted_c_output())
    research_fn = _make_research(dossier, tmp_path, cost=0.0, calls=research_calls)
    committer = _FakeCommitter()

    with pytest.raises(live.CostCeilingError):
        _run(
            config,
            tie,
            invoke=invoke,
            research_fn=research_fn,
            calls_research=research_calls,
            committer=committer,
            committed=lambda path: False,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    # Only the dossier was committed before the halt; no predictions.
    assert len(committer.commits) == 1
    assert "dossier" in committer.commits[0]["message"].lower()


def test_per_tie_call_cap_halts_a_runaway(tmp_path, monkeypatch) -> None:
    _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)
    config = copy.deepcopy(pipeline.load_config())
    config["llm"]["max_billed_calls_per_tie"] = 1  # research alone exhausts the cap.
    research_calls: list = []
    invoke = _make_invoke([], _accepted_b_output(), _accepted_c_output())
    research_fn = _make_research(dossier, tmp_path, cost=0.1, calls=research_calls)

    with pytest.raises(live.CallCapError):
        _run(
            config,
            tie,
            invoke=invoke,
            research_fn=research_fn,
            calls_research=research_calls,
            committer=_FakeCommitter(),
            committed=lambda path: False,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )


# --- substantive violation halts without regenerating ---------------------------


def test_substantive_violation_halts_and_records_the_attempt(
    tmp_path, monkeypatch
) -> None:
    _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)
    config = copy.deepcopy(pipeline.load_config())
    records: list = []
    research_calls: list = []
    invoke = _make_invoke(records, _rejected_b_output(), _accepted_c_output())
    research_fn = _make_research(dossier, tmp_path, cost=0.2, calls=research_calls)
    committer = _FakeCommitter()

    with pytest.raises(live.SubstantiveHalt):
        _run(
            config,
            tie,
            invoke=invoke,
            research_fn=research_fn,
            calls_research=research_calls,
            committer=committer,
            committed=lambda path: False,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
        )
    # The B call happened once and was not regenerated; C was never reached.
    assert len(records) == 1
    # Only the dossier committed; the rejected prediction is not committed.
    assert len(committer.commits) == 1
    # The rejected attempt is recorded as real data in the gitignored logs.
    logs = list((tmp_path / "logs").glob("llm_call_b_rejected_*.json"))
    assert len(logs) == 1


# --- the re-research guard -------------------------------------------------------


def test_re_research_is_refused_when_a_committed_dossier_exists(
    tmp_path, monkeypatch
) -> None:
    _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)
    config = copy.deepcopy(pipeline.load_config())
    d_path = artifact.dossier_path(
        tie["round"], tie["match"], "2026-06-25", dossiers_dir=tmp_path
    )
    research_calls: list = []
    invoke = _make_invoke([], _accepted_b_output(), _accepted_c_output())
    research_fn = _make_research(dossier, tmp_path, cost=0.2, calls=research_calls)

    with pytest.raises(live.ReResearchError):
        _run(
            config,
            tie,
            invoke=invoke,
            research_fn=research_fn,
            calls_research=research_calls,
            committer=_FakeCommitter(),
            committed=lambda path: str(path) == str(d_path),
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            force_research=True,
        )
    assert research_calls == []  # research was never run.


# --- resume continues from B on a committed dossier -----------------------------


def test_resume_continues_from_b_when_only_dossier_is_committed(
    tmp_path, monkeypatch
) -> None:
    _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)
    config = copy.deepcopy(pipeline.load_config())
    # The dossier is already committed and on disk; the predictions are not.
    d_path = artifact.write_dossier(
        dossier, meta={"cost": {"total_usd": 0.0}}, dossiers_dir=tmp_path
    )
    records: list = []
    research_calls: list = []
    invoke = _make_invoke(records, _accepted_b_output(), _accepted_c_output())
    research_fn = _make_research(dossier, tmp_path, cost=0.5, calls=research_calls)
    committer = _FakeCommitter()

    result = _run(
        config,
        tie,
        invoke=invoke,
        research_fn=research_fn,
        calls_research=research_calls,
        committer=committer,
        committed=lambda path: str(path) == str(d_path),
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert result.status == "completed"
    assert research_calls == []  # never re-researched.
    assert len(records) == 2  # B and C still ran on the frozen dossier.
    # Two runtime commits on resume: predictions, then the log. No new dossier commit.
    assert len(committer.commits) == 2
    assert "predictions a, b, c" in committer.commits[0]["message"].lower()
    assert "log the live run" in committer.commits[1]["message"].lower()


# --- missing key halts cleanly --------------------------------------------------


def test_missing_key_halts_cleanly(tmp_path, monkeypatch) -> None:
    _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)
    config = copy.deepcopy(pipeline.load_config())
    research_calls: list = []
    invoke = _make_invoke([], _accepted_b_output(), _accepted_c_output())
    research_fn = _make_research(dossier, tmp_path, cost=0.5, calls=research_calls)
    committer = _FakeCommitter()

    with pytest.raises(live.KeyMissingError):
        _run(
            config,
            tie,
            invoke=invoke,
            research_fn=research_fn,
            calls_research=research_calls,
            committer=committer,
            committed=lambda path: False,
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            rehearsal=True,
            set_key=False,
        )
    assert committer.commits == []  # halted before any commit.
    assert research_calls == []
