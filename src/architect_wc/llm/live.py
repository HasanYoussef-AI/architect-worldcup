"""The live two-call orchestrator: research, Prediction B, Prediction C, per tie.

This is the per-tie protocol that ties the frozen research dossier, the analytic
Prediction A, the analyst Prediction B, and the reconciliation Prediction C into one
forward-only run with the full discipline: a pre-kickoff guard, key and gitignore
discipline, a session dollar ceiling and a per-tie billed-call cap, the halt-and-
surface failure policy, the runtime three-commit cadence, coarse commit-based resume,
and a labelled rehearsal mode with a single B variance probe.

The model call is the only network boundary, and it is injected. In every test it is
a stub returning canned structured responses, and no real Anthropic client is ever
constructed. The pure protocol around it (tie selection, the guards, the caps, the
commit ordering, the resume logic, the prompt building) runs for real offline.

The runtime protocol, in exact order, every step before that tie's kickoff:

  1. Code selects the tie from the committed fixtures, deterministic.
  2. Pre-kickoff guard: refuse if now is at or past kickoff, except in rehearsal.
  3. Research call, build the dossier, gate it, commit the dossier alone (commit one).
  4. Prediction B reads only the frozen committed dossier, no tools.
  5. Prediction A computed, analytic, free.
  6. Prediction C reads the frozen dossier plus A plus B.
  7. Commit B, C, and A together (commit two).
  8. Append the runtime session-log entry, commit three.

All three runtime commits land before kickoff.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from architect_wc import artifact
from architect_wc.llm import model_call, prediction_a, reconcile_c, research, rounds

# --- halts: every failure is a clear, surfaced exception -------------------------


class LiveHalt(Exception):
    """Base for a halt-and-surface failure in the live orchestrator."""


class KeyMissingError(LiveHalt):
    """ANTHROPIC_API_KEY is absent when a paid call is about to run."""


class EnvNotIgnoredError(LiveHalt):
    """.gitignore does not ignore .env, so a key could be committed."""


class KickoffPassedError(LiveHalt):
    """The tie's scheduled kickoff is at or before now, in a forward-only run."""


class CostCeilingError(LiveHalt):
    """The session dollar ceiling would be crossed by the next paid call."""


class CallCapError(LiveHalt):
    """The per-tie billed-call cap would be exceeded by the next call."""


class ReResearchError(LiveHalt):
    """A re-research was requested while a committed dossier already exists."""


class SubstantiveHalt(LiveHalt):
    """A coherent prediction violated a rule; halt without regenerating."""


# --- key and gitignore discipline -----------------------------------------------

API_KEY_ENV = "ANTHROPIC_API_KEY"


def require_api_key(env: dict[str, str] | None = None) -> None:
    """Raise unless ANTHROPIC_API_KEY is present and non-empty in the environment.

    The key is read only from the environment, never from an argument, a config file,
    or anywhere logged. This only checks presence; the value is never read here, never
    returned, and never logged. If the key is absent when a paid call is about to run,
    the run halts with a clear message and no fallback.
    """
    environ = os.environ if env is None else env
    if not (environ.get(API_KEY_ENV) or "").strip():
        raise KeyMissingError(
            f"{API_KEY_ENV} is not set. The live path reads the key only from this "
            f"environment variable. Run inside a subshell that sources it for the one "
            f"command, for example: ( set -a; . ./.env; set +a; uv run wc-llm-live "
            f"--round R32 --match 73 ). Halting; no fallback."
        )


def assert_env_gitignored(repo_root: str | Path = ".") -> None:
    """Raise unless .gitignore ignores .env, so a key file can never be committed.

    Ignore before create: this is checked before any client is constructed, so the
    discipline is verified before any code path could write a key to disk.
    """
    gitignore = Path(repo_root) / ".gitignore"
    entries = set()
    if gitignore.exists():
        entries = {
            line.strip()
            for line in gitignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
    if ".env" not in entries:
        raise EnvNotIgnoredError(
            f"{gitignore} does not ignore .env. Add a .env entry before any run, so a "
            f"key written there can never be committed. Halting."
        )


# --- tie selection and the pre-kickoff guard ------------------------------------

FIXTURE_COLUMNS = ("match", "round", "home_team", "away_team", "kickoff")


def select_tie(config: dict[str, Any], round_code: str, match: int) -> dict[str, Any]:
    """Select one tie from the committed fixtures, deterministic, code not the model.

    Reads the committed dated fixtures CSV named in config.llm.fixtures, the same
    controlled-input pattern as the group draw and the national sources. The CSV
    carries match, round, home_team, away_team, and kickoff (an ISO datetime). Returns
    the single matching tie with kickoff parsed to a timestamp. Fails loud on a
    missing file, a missing column, or no matching row.
    """
    rounds.require_round(round_code)
    llm = config.get("llm", {}) or {}
    path = llm.get("fixtures")
    if not path:
        raise FileNotFoundError(
            "config.llm.fixtures is not set. Commit a dated fixtures CSV with columns "
            f"{list(FIXTURE_COLUMNS)} and point config.llm.fixtures at it before a "
            "live run."
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fixtures not found: {path}. Expected a committed, dated CSV with columns "
            f"{list(FIXTURE_COLUMNS)}."
        )
    frame = pd.read_csv(path, comment="#")
    missing = [column for column in FIXTURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Fixtures {path} is missing expected columns: {missing}. "
            f"Found {list(frame.columns)}."
        )
    rows = frame[
        (frame["round"].astype(str).str.strip() == round_code)
        & (frame["match"].astype(int) == int(match))
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one fixture for round {round_code} match {match} in "
            f"{path}; found {len(rows)}."
        )
    row = rows.iloc[0]
    kickoff = pd.Timestamp(row["kickoff"])
    if kickoff.tzinfo is None:
        kickoff = kickoff.tz_localize("UTC")
    return {
        "match": int(row["match"]),
        "round": str(row["round"]).strip(),
        "home_team": str(row["home_team"]).strip(),
        "away_team": str(row["away_team"]).strip(),
        "kickoff": kickoff.to_pydatetime(),
    }


def assert_before_kickoff(tie: dict[str, Any], now: datetime, rehearsal: bool) -> None:
    """Refuse to proceed if now is at or past the tie's kickoff. Skipped in rehearsal.

    For a forward-only run this guard is what makes the committed dossier a genuine
    pre-kickoff leakage proof. Rehearsal is the dry run on a decided match, so it skips
    only this guard; everything else runs the identical path.
    """
    if rehearsal:
        return
    kickoff = tie.get("kickoff")
    if kickoff is None:
        home, away = tie.get("home_team"), tie.get("away_team")
        raise KickoffPassedError(
            f"No kickoff time for {home} versus {away}; a forward-only run needs a "
            f"known kickoff. Use rehearsal for a decided match."
        )
    if now >= kickoff:
        raise KickoffPassedError(
            f"Kickoff for {tie['home_team']} versus {tie['away_team']} is {kickoff}. "
            f"Now is {now}, at or past kickoff. A forward-only run must finish "
            f"before kickoff. Halting."
        )


# --- the session budget: dollar ceiling and per-tie billed-call cap -------------


@dataclass
class Budget:
    """The two real circuit breakers, both enforced before every paid call.

    ceiling is the session dollar ceiling, computed from actual token usage. cap is
    the per-tie maximum count of billed model calls, a backstop against any logic
    loop. authorize is called before each paid call (and before each transient retry):
    if the next call would exceed the cap, or if cumulative actual spend plus this
    call's maximum possible cost would cross the ceiling, it halts. record is called
    after each response that came back, accumulating actual spend and counting the
    billed call.
    """

    ceiling: float
    cap: int
    spent: float = 0.0
    billed_calls: int = 0

    def authorize(self, max_cost: float) -> None:
        if self.billed_calls + 1 > self.cap:
            raise CallCapError(
                f"Per-tie billed-call cap of {self.cap} reached; refusing a further "
                f"call. This is the logic-loop backstop. Halting."
            )
        if self.spent + max_cost > self.ceiling:
            raise CostCeilingError(
                f"Session dollar ceiling {self.ceiling:.2f} would be crossed: spent "
                f"{self.spent:.4f} plus this call's max {max_cost:.4f}. Halting before "
                f"the call. No fallback."
            )

    def record(self, actual: float) -> None:
        self.spent += actual
        self.billed_calls += 1


# --- git, the only durable state ------------------------------------------------


def git_commit(paths: list[Path], message: str, repo_root: str | Path = ".") -> None:
    """Stage the given paths and commit them, the runtime durable-state step."""
    root = str(repo_root)
    rels = [os.path.relpath(str(path), root) for path in paths]
    subprocess.run(["git", "-C", root, "add", *rels], check=True)
    subprocess.run(["git", "-C", root, "commit", "-m", message], check=True)


def git_committed(path: str | Path, repo_root: str | Path = ".") -> bool:
    """True if the path is committed in HEAD, the coarse durable-state probe.

    Commits are the only durable state, so resume reads HEAD, not the working tree.
    """
    root = str(repo_root)
    rel = os.path.relpath(str(path), root)
    result = subprocess.run(
        ["git", "-C", root, "ls-tree", "HEAD", "--", rel],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def noop_committer(paths: list[Path], message: str) -> None:
    """A committer that performs no git operation, injected for rehearsal runs.

    A rehearsal writes its REHEARSAL-marked artifacts to disk for inspection but must
    produce zero git commits, so the runtime three-commit cadence never lands a
    rehearsal on the branch. The real forward-only path keeps the default git_commit
    and its full cadence unchanged.
    """
    return None


def file_ref(path: str | Path) -> dict[str, str]:
    """A {path, content_sha256} reference over a written artifact's exact bytes."""
    path = Path(path)
    return {
        "path": str(path),
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


# --- the per-call model boundary closure ----------------------------------------


@dataclass
class _CallTrace:
    """What one B or C model call captured, for the raw log and the document meta."""

    system: str = ""
    user: str = ""
    parsed: dict[str, Any] = field(default_factory=dict)
    raw_calls: list[model_call.RawResponse] = field(default_factory=list)


def _default_client_factory() -> Any:
    """Construct the real Anthropic client. The key is read from the environment."""
    import anthropic

    return anthropic.Anthropic()


@dataclass
class LiveResult:
    """The outcome of a tie run, for the CLI and the gate to assert on."""

    status: str
    tie: dict[str, Any]
    paths: dict[str, str] = field(default_factory=dict)
    spent: float = 0.0
    billed_calls: int = 0
    accepted: dict[str, bool] = field(default_factory=dict)
    variance: dict[str, float] | None = None


def run_tie(
    config: dict[str, Any],
    tie: dict[str, Any],
    as_of_date: str,
    *,
    now: datetime,
    invoke: Callable[..., model_call.RawResponse] = model_call.invoke_structured,
    research_fn: Callable[..., dict[str, Any]] = research.research_fixture,
    build_a: Callable[..., dict[str, Any]] = prediction_a.build_prediction_a,
    client_factory: Callable[[], Any] = _default_client_factory,
    committer: Callable[[list[Path], str], None] | None = None,
    committed: Callable[[str | Path], bool] | None = None,
    output_dir: str | Path = artifact.DOSSIERS_DIR,
    logs_dir: str | Path = artifact.LOGS_DIR,
    session_log_path: str | Path = "session-log.md",
    repo_root: str | Path = ".",
    rehearsal: bool = False,
    force_research: bool = False,
) -> LiveResult:
    """Run the full per-tie protocol, halting and surfacing on any failure.

    The model boundary, research, A, and the client are all injected so the protocol
    runs hermetically in tests with a stub boundary and no real client; in production
    they default to the real implementations. Returns a LiveResult; raises a LiveHalt
    (or a model_call.PolicyHalt) on any halt condition, leaving the durable state at
    the last successful commit.
    """
    round_code = rounds.require_round(str(tie["round"]))
    match = int(tie["match"])
    home = str(tie["home_team"])
    away = str(tie["away_team"])
    output_dir = Path(output_dir)
    logs_dir = Path(logs_dir)

    if committer is None:
        committer = functools.partial(git_commit, repo_root=repo_root)
    if committed is None:
        committed = functools.partial(git_committed, repo_root=repo_root)

    # Resume, coarse, on committed state only. If the predictions are committed the
    # tie is done; if only the dossier is committed, resume from B on the frozen
    # dossier and never re-research.
    d_path = artifact.dossier_path(
        round_code, match, as_of_date, rehearsal=rehearsal, dossiers_dir=output_dir
    )
    pred_paths = {
        kind: artifact.prediction_path(
            kind,
            round_code,
            match,
            as_of_date,
            rehearsal=rehearsal,
            predictions_dir=output_dir,
        )
        for kind in ("a", "b", "c")
    }
    dossier_committed = committed(d_path)
    if all(committed(path) for path in pred_paths.values()):
        return LiveResult(status="skipped_done", tie=tie)

    # 2. Pre-kickoff guard (skipped only in rehearsal).
    assert_before_kickoff(tie, now, rehearsal)

    # Key and gitignore discipline, before any client is constructed.
    assert_env_gitignored(repo_root)
    require_api_key()
    client = client_factory()

    llm = config.get("llm", {}) or {}
    model = llm.get("model", "claude-opus-4-8")
    effort = llm.get("effort_prediction", "xhigh")
    use_cache = bool(llm.get("prompt_cache", True))
    max_tokens = int(llm.get("max_tokens_prediction", 16000))
    ceiling = float(llm.get("session_dollar_ceiling", 6.0))
    cap = int(llm.get("max_billed_calls_per_tie", 6))
    research_max = float(llm.get("research_max_cost_estimate", 2.5))
    # Rehearsal adds the single B variance probe (and tolerates its one re-roll), so
    # its cap is the forward cap plus two; the forward cap is the real bound.
    budget = Budget(ceiling=ceiling, cap=cap + (2 if rehearsal else 0))
    git_sha = artifact.get_git_sha()
    fixture = {"match": match, "home_team": home, "away_team": away}

    commits: list[dict[str, Any]] = []

    # 3. Research, then commit the dossier alone (commit one). Resume reuses a frozen
    # committed dossier and never re-researches.
    if dossier_committed:
        if force_research:
            raise ReResearchError(
                f"A committed dossier already exists for {round_code} match {match} "
                f"at cutoff {as_of_date}; refusing to re-research. The committed "
                f"dossier is the frozen leakage anchor; resume from Prediction B."
            )
        envelope = json.loads(d_path.read_text(encoding="utf-8"))
        dossier = envelope["dossier"]
        dossier_path = d_path
    else:
        budget.authorize(research_max)
        produced = _research_with_one_reresearch(
            research_fn,
            config,
            fixture,
            as_of_date,
            round_code,
            client=client,
            rehearsal=rehearsal,
            dossiers_dir=output_dir,
        )
        budget.record(float(produced["meta"].get("cost", {}).get("total_usd", 0.0)))
        dossier = produced["dossier"]
        dossier_path = Path(produced["meta"]["dossier_path"])
        committer([dossier_path], _dossier_commit_message(tie, as_of_date, rehearsal))
        commits.append({"kind": "dossier", "paths": [str(dossier_path)]})

    dossier_ref = file_ref(dossier_path)
    dossier_sha = dossier_ref["content_sha256"]
    stamp = now.isoformat()

    # 4. Prediction B, reading only the frozen committed dossier, no tools.
    b_trace = _CallTrace()
    b_model_call = _make_b_model_call(
        invoke,
        client,
        budget,
        b_trace,
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        use_cache=use_cache,
    )
    from architect_wc.llm import prediction_b

    b_doc = prediction_b.predict_b(
        dossier,
        config,
        b_model_call,
        dossier_ref=dossier_ref,
        code_commit=git_sha,
        timestamp=stamp,
    )
    b_log_path = _write_call_log(
        "b",
        round_code,
        match,
        dossier_sha,
        git_sha,
        stamp,
        model,
        effort,
        b_trace,
        b_doc,
        logs_dir=logs_dir,
    )
    _halt_if_rejected(
        "B",
        b_doc,
        b_trace,
        round_code,
        match,
        dossier_sha,
        git_sha,
        stamp,
        model,
        effort,
        logs_dir,
    )
    b_path = artifact.write_prediction(
        b_doc,
        "b",
        meta=_pred_meta(b_trace, b_log_path),
        rehearsal=rehearsal,
        predictions_dir=output_dir,
    )

    # 5. Prediction A, analytic and free, no budget.
    a_doc = build_a(config, fixture, as_of_date, round_code)
    a_path = artifact.write_prediction(
        a_doc,
        "a",
        meta={"source": "analytic, no model call"},
        rehearsal=rehearsal,
        predictions_dir=output_dir,
    )

    # 6. Prediction C, reading the frozen dossier plus A plus B.
    pool = reconcile_c.pool_from_predictions(a_doc, b_doc)
    c_trace = _CallTrace()
    c_model_call = _make_c_model_call(
        invoke,
        client,
        budget,
        c_trace,
        pool,
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        use_cache=use_cache,
    )
    c_doc = reconcile_c.predict_c(
        dossier,
        a_doc,
        b_doc,
        config,
        c_model_call,
        dossier_ref=dossier_ref,
        a_ref=file_ref(a_path),
        b_ref=file_ref(b_path),
        code_commit=git_sha,
        timestamp=stamp,
    )
    c_log_path = _write_call_log(
        "c",
        round_code,
        match,
        dossier_sha,
        git_sha,
        stamp,
        model,
        effort,
        c_trace,
        c_doc,
        logs_dir=logs_dir,
    )
    _halt_if_rejected(
        "C",
        c_doc,
        c_trace,
        round_code,
        match,
        dossier_sha,
        git_sha,
        stamp,
        model,
        effort,
        logs_dir,
    )
    c_path = artifact.write_prediction(
        c_doc,
        "c",
        meta=_pred_meta(c_trace, c_log_path),
        rehearsal=rehearsal,
        predictions_dir=output_dir,
    )

    commit_two_paths = [a_path, b_path, c_path]
    variance: dict[str, float] | None = None
    if rehearsal:
        variance, probe_path = _variance_probe(
            invoke,
            client,
            budget,
            dossier,
            config,
            b_doc,
            dossier_ref,
            git_sha,
            stamp,
            model=model,
            effort=effort,
            max_tokens=max_tokens,
            use_cache=use_cache,
            round_code=round_code,
            match=match,
            output_dir=output_dir,
        )
        commit_two_paths.append(probe_path)

    # 7. Commit B, C, and A together (commit two).
    committer(commit_two_paths, _predictions_commit_message(tie, as_of_date, rehearsal))
    commits.append({"kind": "predictions", "paths": [str(p) for p in commit_two_paths]})

    # 8. Append the runtime session-log entry and commit it (commit three).
    log_path = Path(session_log_path)
    _append_session_log(
        log_path, tie, as_of_date, rehearsal, b_doc, c_doc, budget, variance, commits
    )
    committer([log_path], _sessionlog_commit_message(tie, as_of_date, rehearsal))
    commits.append({"kind": "session_log", "paths": [str(log_path)]})

    return LiveResult(
        status="completed",
        tie=tie,
        paths={
            "dossier": str(dossier_path),
            "a": str(a_path),
            "b": str(b_path),
            "c": str(c_path),
        },
        spent=budget.spent,
        billed_calls=budget.billed_calls,
        accepted={
            "b": bool(b_doc["validation"]["accepted"]),
            "c": bool(c_doc["validation"]["accepted"]),
        },
        variance=variance,
    )


# --- research with at most one re-research --------------------------------------


def _research_with_one_reresearch(
    research_fn: Callable[..., dict[str, Any]],
    config: dict[str, Any],
    fixture: dict[str, Any],
    as_of_date: str,
    round_code: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run research, allowing exactly one full re-research on a gate failure.

    A coverage or schema gate failure is a research semantic failure: research never
    loops, but it may be redone once, logged as a replacement dossier with a new hash
    and timestamp, still pre-kickoff. If the re-research also fails, halt. Transient
    transport failures inside the research call are retried by the research call
    itself; this re-research is only for a genuine gate failure on a produced dossier.
    """
    try:
        return research_fn(config, fixture, as_of_date, round_code, **kwargs)
    except Exception:
        # One clean re-research on the same frozen inputs; a second failure propagates.
        return research_fn(config, fixture, as_of_date, round_code, **kwargs)


# --- the B and C model-call closures the prediction builders call ----------------


def _make_b_model_call(
    invoke: Callable[..., model_call.RawResponse],
    client: Any,
    budget: Budget,
    trace: _CallTrace,
    *,
    model: str,
    effort: str,
    max_tokens: int,
    use_cache: bool,
) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    """Build the model_call(dossier, config) closure predict_b expects.

    It builds B's prompt from the frozen dossier, computes this call's maximum cost
    for the ceiling, then runs the boundary under the transient-retry and one-re-roll
    policy, capturing the prompts and raw responses for the log and the prompt
    exclusion gate.
    """

    def model_call_fn(
        dossier: dict[str, Any], _config: dict[str, Any]
    ) -> dict[str, Any]:
        system, user = model_call.build_b_prompt(dossier)
        trace.system, trace.user = system, user
        max_cost = model_call.max_call_cost(system, user, max_tokens)
        schema = _output_schema("prediction_b_output")
        hooks = model_call.PolicyHooks(
            authorize=lambda: budget.authorize(max_cost),
            record=lambda raw: budget.record(model_call.actual_cost(raw.usage)),
        )
        parsed = model_call.call_with_policy(
            lambda: invoke(
                client,
                model=model,
                effort=effort,
                system=system,
                user_content=user,
                schema=schema,
                max_tokens=max_tokens,
                use_prompt_cache=use_cache,
            ),
            output_kind="prediction_b_output",
            require_seven_factors=True,
            hooks=hooks,
        )
        trace.parsed = parsed
        trace.raw_calls = hooks.raw_calls
        return parsed

    return model_call_fn


def _make_c_model_call(
    invoke: Callable[..., model_call.RawResponse],
    client: Any,
    budget: Budget,
    trace: _CallTrace,
    pool: dict[str, Any],
    *,
    model: str,
    effort: str,
    max_tokens: int,
    use_cache: bool,
) -> Callable[..., dict[str, Any]]:
    """Build the model_call(dossier, a, b, config) closure predict_c expects."""

    def model_call_fn(
        dossier: dict[str, Any],
        prediction_a_doc: dict[str, Any],
        prediction_b_doc: dict[str, Any],
        _config: dict[str, Any],
    ) -> dict[str, Any]:
        system, user = model_call.build_c_prompt(
            dossier, prediction_a_doc, prediction_b_doc, pool
        )
        trace.system, trace.user = system, user
        max_cost = model_call.max_call_cost(system, user, max_tokens)
        schema = _output_schema("prediction_c_output")
        hooks = model_call.PolicyHooks(
            authorize=lambda: budget.authorize(max_cost),
            record=lambda raw: budget.record(model_call.actual_cost(raw.usage)),
        )
        parsed = model_call.call_with_policy(
            lambda: invoke(
                client,
                model=model,
                effort=effort,
                system=system,
                user_content=user,
                schema=schema,
                max_tokens=max_tokens,
                use_prompt_cache=use_cache,
            ),
            output_kind="prediction_c_output",
            require_seven_factors=False,
            hooks=hooks,
        )
        trace.parsed = parsed
        trace.raw_calls = hooks.raw_calls
        return parsed

    return model_call_fn


def _output_schema(kind: str) -> dict[str, Any]:
    from architect_wc.llm import schemas

    return schemas.load_schema(kind)


# --- the rehearsal variance probe -----------------------------------------------


def _variance_probe(
    invoke: Callable[..., model_call.RawResponse],
    client: Any,
    budget: Budget,
    dossier: dict[str, Any],
    config: dict[str, Any],
    b_doc: dict[str, Any],
    dossier_ref: dict[str, str],
    git_sha: str,
    stamp: str,
    *,
    model: str,
    effort: str,
    max_tokens: int,
    use_cache: bool,
    round_code: str,
    match: int,
    output_dir: Path,
) -> tuple[dict[str, float], Path]:
    """Rehearsal only: one extra Prediction B call, recording the spread of the two.

    This is the only place a best-of-style duplication exists, and it is a measurement,
    not aggregation: the first B stands as the prediction. The probe runs the identical
    B path on the same frozen dossier and records the spread between the two emitted
    three-ways and the two advance probabilities, written to a REHEARSAL-marked probe
    artifact.
    """
    from architect_wc.llm import prediction_b

    probe_trace = _CallTrace()
    probe_call = _make_b_model_call(
        invoke,
        client,
        budget,
        probe_trace,
        model=model,
        effort=effort,
        max_tokens=max_tokens,
        use_cache=use_cache,
    )
    b2 = prediction_b.predict_b(
        dossier,
        config,
        probe_call,
        dossier_ref=dossier_ref,
        code_commit=git_sha,
        timestamp=stamp,
    )
    first = b_doc["emitted_three_way"]
    second = b2["emitted_three_way"]
    spread = {
        "p_home": abs(first["p_home"] - second["p_home"]),
        "p_draw": abs(first["p_draw"] - second["p_draw"]),
        "p_away": abs(first["p_away"] - second["p_away"]),
        "advance_probability": abs(
            b_doc["shootout"]["advance_probability"]
            - b2["shootout"]["advance_probability"]
        ),
    }
    spread["max_component"] = max(spread.values())
    probe = {
        "rehearsal": True,
        "round": round_code,
        "match": match,
        "first_b": first,
        "probe_b": second,
        "spread": spread,
    }
    name = artifact._llm_artifact_name(
        "b_variance_probe", round_code, match, str(b_doc["cutoff"]), True
    )
    path = Path(output_dir) / name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(probe, indent=2, default=str) + "\n", encoding="utf-8")
    return spread, path


# --- logging, provenance, commit messages, and the session-log entry ------------


def _write_call_log(
    kind: str,
    round_code: str,
    match: int,
    dossier_sha: str,
    git_sha: str,
    stamp: str,
    model: str,
    effort: str,
    trace: _CallTrace,
    document: dict[str, Any],
    *,
    logs_dir: Path,
) -> Path:
    """Write the heavy raw model-call log for B or C, the gitignored provenance."""
    record = {
        "kind": kind,
        "round": round_code,
        "match": match,
        "dossier_sha256": dossier_sha,
        "model": model,
        "effort": effort,
        "thinking": "adaptive",
        "git_sha": git_sha,
        "timestamp": stamp,
        "prompt": {"system": trace.system, "user": trace.user},
        "raw_responses": [
            {
                "stop_reason": raw.stop_reason,
                "message_id": raw.message_id,
                "request_id": raw.request_id,
                "usage": raw.usage,
                "text": raw.text,
            }
            for raw in trace.raw_calls
        ],
        "structured_output": trace.parsed,
        "document": document,
    }
    return artifact.write_llm_call_log(record, logs_dir=logs_dir)


def _pred_meta(trace: _CallTrace, log_path: Path) -> dict[str, Any]:
    """The committed prediction envelope meta, referencing the gitignored raw log."""
    last = trace.raw_calls[-1] if trace.raw_calls else None
    return {
        "raw_log": str(log_path),
        "message_id": getattr(last, "message_id", None),
        "request_id": getattr(last, "request_id", None),
        "usage": getattr(last, "usage", None),
        "n_billed_calls": len(trace.raw_calls),
    }


def _halt_if_rejected(
    label: str,
    document: dict[str, Any],
    trace: _CallTrace,
    round_code: str,
    match: int,
    dossier_sha: str,
    git_sha: str,
    stamp: str,
    model: str,
    effort: str,
    logs_dir: Path,
) -> None:
    """Substantive failure: a coherent prediction that violates a rule. Do not
    regenerate; halt, surface, and record the rejected attempt as real data.
    """
    if document["validation"]["accepted"]:
        return
    record = {
        "kind": f"{label.lower()}_rejected",
        "round": round_code,
        "match": match,
        "dossier_sha256": dossier_sha,
        "model": model,
        "effort": effort,
        "thinking": "adaptive",
        "git_sha": git_sha,
        "timestamp": stamp,
        "rejected_document": document,
        "structured_output": trace.parsed,
    }
    rejected_path = artifact.write_llm_call_log(record, logs_dir=logs_dir)
    raise SubstantiveHalt(
        f"Prediction {label} is a coherent prediction that violates a rule: "
        f"{document['validation']['rejection_reason']}. Not auto-regenerated. The "
        f"rejected attempt is recorded at {rejected_path}. Halting for review."
    )


def _label(tie: dict[str, Any], rehearsal: bool) -> str:
    marker = " REHEARSAL" if rehearsal else ""
    return (
        f"{tie['round']} match {tie['match']} "
        f"{tie['home_team']} vs {tie['away_team']}{marker}"
    )


def _dossier_commit_message(
    tie: dict[str, Any], as_of_date: str, rehearsal: bool
) -> str:
    return f"Commit the frozen dossier for {_label(tie, rehearsal)} as of {as_of_date}"


def _predictions_commit_message(
    tie: dict[str, Any], as_of_date: str, rehearsal: bool
) -> str:
    return f"Commit Predictions A, B, C for {_label(tie, rehearsal)} as of {as_of_date}"


def _sessionlog_commit_message(
    tie: dict[str, Any], as_of_date: str, rehearsal: bool
) -> str:
    return f"Log the live run for {_label(tie, rehearsal)} as of {as_of_date}"


def _append_session_log(
    path: Path,
    tie: dict[str, Any],
    as_of_date: str,
    rehearsal: bool,
    b_doc: dict[str, Any],
    c_doc: dict[str, Any],
    budget: Budget,
    variance: dict[str, float] | None,
    commits: list[dict[str, Any]],
) -> None:
    """Append the runtime session-log entry for this live tie. Commit three."""
    date = str(as_of_date)
    marker = "  REHEARSAL" if rehearsal else ""
    variance_line = (
        f"\n- Variance probe: max B spread {variance['max_component']:.4f}"
        if variance
        else ""
    )
    entry = (
        f"\n## {date}  Live run{marker}  {tie['round']} match {tie['match']} "
        f"{tie['home_team']} vs {tie['away_team']}\n"
        f"- Did: research, dossier committed, then B, A, C built and committed; "
        f"three runtime commits before kickoff.\n"
        f"- Cost: spent {budget.spent:.4f} of ceiling {budget.ceiling:.2f} over "
        f"{budget.billed_calls} billed calls.\n"
        f"- Verdict: B accepted={b_doc['validation']['accepted']}, "
        f"C accepted={c_doc['validation']['accepted']}.{variance_line}\n"
        f"- Commits: {', '.join(commit['kind'] for commit in commits)}, this entry.\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
