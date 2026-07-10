"""Forward-only gate: the two refusals README sections 8 and 11 rest on.

A prediction that could be made after the result is not a prediction. Two branches of
live.run_tie enforce that, and until now neither was pinned by a test: the pre-kickoff
guard refuses a tie whose kickoff has passed, and the resume check skips a tie whose
three prediction artifacts are already committed. This gate drives the real branches
in run_tie, not a mock of them, and proves each refuses before any model call, because
re-predicting a settled or already-predicted tie is exactly the failure the refusals
exist to prevent.

Hermetic: no API, no key required for the refusals, no network, no real Anthropic
client, and no data snapshot. It reuses the committed-dossier stubs and the injected
model boundary from test_llm_live_gate rather than forking them.
"""

from __future__ import annotations

import copy
from datetime import timedelta

import pytest
from test_llm_live_gate import (
    KICKOFF,
    ROOT,
    _accepted_b_output,
    _accepted_c_output,
    _build_a,
    _committed_dossier,
    _FakeCommitter,
    _make_invoke,
    _make_research,
    _no_real_client,
    _NoClient,
    _run,
    _tie,
    tie_as_of,
)

from architect_wc import pipeline
from architect_wc.llm import live


def test_run_refuses_a_tie_whose_kickoff_has_passed(tmp_path, monkeypatch) -> None:
    _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)  # kickoff is KICKOFF, far in the future
    config = copy.deepcopy(pipeline.load_config())

    # Boundary, future side: now is before kickoff, so the run is not refused and
    # completes. This pins the boundary rather than the exception type alone.
    records: list = []
    research_calls: list = []
    future = _run(
        config,
        tie,
        invoke=_make_invoke(records, _accepted_b_output(), _accepted_c_output()),
        research_fn=_make_research(dossier, tmp_path, cost=0.5, calls=research_calls),
        calls_research=research_calls,
        committer=_FakeCommitter(),
        committed=lambda path: False,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    assert future.status == "completed"

    # Boundary, past side: drive the real guard inside run_tie with now past kickoff.
    past_records: list = []
    past_research: list = []
    past_invoke = _make_invoke(past_records, _accepted_b_output(), _accepted_c_output())
    past_research_fn = _make_research(dossier, tmp_path, cost=0.5, calls=past_research)
    with pytest.raises(live.KickoffPassedError):
        live.run_tie(
            config,
            tie,
            tie_as_of(tie),
            now=KICKOFF + timedelta(days=1),
            invoke=past_invoke,
            research_fn=past_research_fn,
            build_a=_build_a,
            client_factory=_NoClient,
            committer=_FakeCommitter(),
            committed=lambda path: False,
            output_dir=tmp_path,
            logs_dir=tmp_path / "logs",
            session_log_path=tmp_path / "session-log.md",
            repo_root=ROOT,
        )
    # The refusal came before any model work: no research, no B, no C call.
    assert past_research == []
    assert past_records == []


def test_run_skips_a_tie_whose_predictions_are_already_committed(
    tmp_path, monkeypatch
) -> None:
    _no_real_client(monkeypatch)
    dossier = _committed_dossier()
    tie = _tie(dossier)
    config = copy.deepcopy(pipeline.load_config())
    records: list = []
    research_calls: list = []

    # All three prediction artifacts already committed: run_tie takes the skip branch
    # before the kickoff guard and before any model call.
    result = _run(
        config,
        tie,
        invoke=_make_invoke(records, _accepted_b_output(), _accepted_c_output()),
        research_fn=_make_research(dossier, tmp_path, cost=0.5, calls=research_calls),
        calls_research=research_calls,
        committer=_FakeCommitter(),
        committed=lambda path: True,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )

    assert result.status == "skipped_done"
    # Re-predicting a committed tie is the failure this refusal prevents: no calls.
    assert research_calls == []
    assert records == []
