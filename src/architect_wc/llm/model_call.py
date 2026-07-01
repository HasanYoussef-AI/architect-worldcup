"""The single model-call boundary for the live Prediction B and Prediction C calls.

Everything the network touches for B and C lives here, in one place, so the rest
of the live path is pure and offline-testable. The boundary is invoke_structured:
one claude-opus-4-8 message, no tools, forced structured output bound to a json
schema, adaptive thinking, effort xhigh. It is the only function any test stubs;
every test passes a stub and no real client is ever constructed in the suite.

The prompt templates are stored verbatim as the canonical templates. Code fills the
bracketed tokens at runtime from the frozen dossier and from A and B; the model
never fills a token. B's filled prompt carries only the dossier, so B never sees the
anchor mapping parameters or the tolerance box. C's filled prompt carries the
dossier, A's forecast and drivers, B's forecast and reasoning, and the computed
pool, but never the tolerance box or the hard cap. They reason, code measures.

The failure policy is halt and surface, never halt and loop:

  - Transient failure (network, timeout, rate limit, 5xx): the boundary raises
    TransientCallError; call_with_policy retries the identical call with exponential
    backoff, max three attempts, then halts.
  - Mechanical semantic failure (unparseable, truncated, off-schema, refusal):
    one automatic clean re-roll on the same frozen inputs, no corrective prompt
    fishing; if the re-roll is also broken, halt.
  - Substantive semantic failure (a coherent prediction that violates a rule) is not
    handled here: it is caught by the prediction builders downstream and the live
    orchestrator halts on it without regenerating.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jsonschema

from architect_wc.llm import schemas
from architect_wc.llm.weights import FACTORS

# --- the verbatim canonical prompt templates ------------------------------------
# Stored exactly as defined. Code fills {HOME_TEAM} and {AWAY_TEAM}; nothing else is
# a token. Do not paraphrase or edit these strings.

PREDICTION_B_SYSTEM = (
    "You are a panel of seven senior football analysts assembled to assess a single "
    "2026 World Cup knockout tie. Each of you holds one specialism and assesses only "
    "your factor. Your seven specialisms are: squad availability and starting "
    "lineup, recent form, meaning underlying performance quality not visible in "
    "results, tactical and stylistic matchup, coaching and staff, strategic "
    "incentives, psychological and momentum, and historical head to head.\n"
    "You assess one tie only: {HOME_TEAM} versus {AWAY_TEAM}. {HOME_TEAM} is the "
    "nominal home team. Every factor is scored from {HOME_TEAM}'s perspective on a "
    "scale from minus three to plus three, where plus three strongly favors "
    "{HOME_TEAM}, minus three strongly favors {AWAY_TEAM}, and zero is no edge to "
    "either side.\n"
    "You are given a dossier of researched findings for this tie, below. The dossier "
    "is your only admissible evidence. You may not use any knowledge of these teams "
    "from outside the dossier, including anything you recall about results, form, "
    "injuries, or events. If it is not in the dossier, you do not know it. This rule "
    "is absolute, because the value of this assessment depends on it resting only on "
    "evidence that was frozen before this match.\n"
    "For each of the seven factors you return: a score from minus three to plus "
    "three, a reasoning passage that justifies the score by referring to specific "
    "dossier findings, and the list of dossier finding ids you relied on. Every "
    "nonzero score must cite at least one dossier finding id. If the dossier holds no "
    "admissible evidence for your factor, you score it zero and set "
    "insufficient_evidence to true, and you do not invent a reason.\n"
    "When you score recent_form, assess only underlying-performance quality that "
    "scorelines hide: chance quality created and conceded, finishing and "
    "shot-quality trend, set-piece threat, and whether performances are trending up "
    "or down independent of results. This factor excludes raw results, goal "
    "difference, and standings, which are Prediction A's domain.\n"
    "After the seven factors, you return your own assessment of the 90 minute result "
    "as three probabilities, for a {HOME_TEAM} win, a draw, and an {AWAY_TEAM} win, "
    "that sum to one. A draw is a valid 90 minute result even though this is a "
    "knockout tie. This tie always resolves to one team advancing, so you also return "
    "a shootout lean, the probability that {HOME_TEAM} advances if the match is level "
    "after 90 minutes, as a number between zero and one. Base the shootout lean only "
    "on dossier evidence bearing on a shootout, such as penalty record, goalkeeper "
    "history, or the availability of designated takers, and if the dossier holds "
    "none, return 0.5.\n"
    "Your emitted three-way probability and your emitted lean must follow from your "
    "seven factor scores. Read your scores together: their balance sets the center "
    "of your emission. A set of scores that nets close to even must produce a "
    "near-even three-way. You may lean away from that center only for an intangible "
    "reason your own factors express, an interaction between factors or a single "
    "dominant intangible the simple sum underweights, and only modestly. Never emit "
    "a distribution your factor scores do not support.\n"
    "Results-quality, meaning group finish, points, goal difference, standings, and "
    "any results-based measure of strength or momentum, is Prediction A's domain and "
    "is already counted there. It must not drive your emission and must not justify "
    "a lean away from your factor-implied center. Your task is the intangible read "
    "the math model cannot see: squad availability, tactical matchup, coaching, "
    "strategic incentives, psychology, and history.\n"
    "You return only the structured object defined by the schema. You do not address "
    "the reader, you do not preface, you do not summarize outside the fields."
)

PREDICTION_C_SYSTEM = (
    "You are a senior reconciliation analyst issuing the final call on a single 2026 "
    "World Cup knockout tie, {HOME_TEAM} versus {AWAY_TEAM}, with {HOME_TEAM} as "
    "nominal home. Two independent forecasts already exist for this tie, and your "
    "task is to reconcile them into a final 90 minute result and shootout lean. You "
    "do not re-score the factors. You work at the level of the result.\n"
    "You are given three things. First, the frozen dossier for this tie, the same "
    "evidence both forecasts used. Second, Prediction A, an analytic model's "
    "forecast, with a drivers explainer stating the strength differentials that "
    "produced it: the Elo difference, the expected goals difference, and the squad "
    "value adjustment, each from {HOME_TEAM}'s perspective. Third, Prediction B, an "
    "analyst panel's forecast, with its seven factor scores and the reasoning behind "
    "each.\n"
    "Your starting reference is the equal weight average of A and B, the pool. You "
    "will be shown the pool. Your job is to decide where the final call should sit "
    "relative to that pool. Where A and B agree, the pool is already a strong answer "
    "and you should stay close to it. Where A and B disagree, your value is in "
    "resolving the disagreement, deciding which forecast the evidence supports on the "
    "points where they diverge, and moving toward it.\n"
    "Any material move away from the pool must be justified, and every justification "
    "must cite exactly one of three sources: a specific element of Prediction A's "
    "reasoning or drivers, a specific element of Prediction B's reasoning, or a "
    "specific dossier finding id. You introduce no new facts about these teams. You "
    "may only reason from what A said, what B said, and what the dossier holds. If "
    "you cannot ground a move in one of those three, you do not make it.\n"
    "You return your final 90 minute result as three probabilities, for a "
    "{HOME_TEAM} win, a draw, and an {AWAY_TEAM} win, summing to one, and a final "
    "shootout lean between zero and one for {HOME_TEAM} advancing from a level "
    "match. For any material move off the pool, you return the justification text and "
    "its cited source. You return only the structured object defined by the schema, "
    "with no preface and no address to the reader."
)


# --- token filling and serialization (pure) -------------------------------------


def fill_template(template: str, home_team: str, away_team: str) -> str:
    """Fill the only two tokens, {HOME_TEAM} and {AWAY_TEAM}. The model fills none."""
    return template.replace("{HOME_TEAM}", home_team).replace("{AWAY_TEAM}", away_team)


def dossier_with_finding_ids(dossier: dict[str, Any]) -> dict[str, Any]:
    """Return the dossier factor evidence with a stable id on every finding.

    The id is "{factor}#{index}", the same convention the prediction builders and
    their citation checks use, so the model cites findings by an id that exists. Pure:
    builds a new structure, does not mutate the input.
    """
    factors_with_ids: dict[str, list[dict[str, Any]]] = {}
    for factor, entries in (dossier.get("factors", {}) or {}).items():
        labelled = []
        for index, entry in enumerate(entries or []):
            labelled.append({"id": f"{factor}#{index}", **entry})
        factors_with_ids[factor] = labelled
    return factors_with_ids


def _dumps(value: Any) -> str:
    """Deterministic JSON for prompts, so the prompt-cache key is stable."""
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def build_b_prompt(dossier: dict[str, Any]) -> tuple[str, str]:
    """Build Prediction B's (system, user) messages from the frozen dossier.

    The system is the verbatim B template with the two tokens filled. The user
    carries only the dossier evidence with citable finding ids, nothing else: B never
    sees the anchor mapping parameters, the tolerance box, Prediction A, or any
    code-computed field.
    """
    home = str(dossier["home_team"])
    away = str(dossier["away_team"])
    system = fill_template(PREDICTION_B_SYSTEM, home, away)
    findings = dossier_with_finding_ids(dossier)
    user = (
        f"Tie: {home} versus {away}, nominal home {home}.\n"
        "Frozen dossier of admissible findings. Each finding has a stable id; cite "
        "findings by these ids.\n\n"
        f"{_dumps(findings)}\n\n"
        "Return the structured object the schema defines."
    )
    return system, user


def a_view_for_c(prediction_a: dict[str, Any]) -> dict[str, Any]:
    """Prediction A's forecast and drivers, the view C is shown.

    A's three-way, shootout lean, advance, and the drivers explainer. A carries no
    anchor or tolerance internals, so this is A's full substantive output; the
    administrative provenance and seed are dropped as not part of the forecast.
    """
    return {
        "three_way": prediction_a["three_way"],
        "shootout_lean": prediction_a["shootout_lean"],
        "advance_probability": prediction_a.get("advance_probability"),
        "drivers": prediction_a["drivers"],
    }


def b_view_for_c(prediction_b: dict[str, Any]) -> dict[str, Any]:
    """Prediction B's forecast and reasoning, the view C is shown.

    B's seven factor scores with their per-factor reasoning, citations, and
    insufficient-evidence flags, B's emitted three-way and emitted lean, and any
    justification B wrote, so C reconciles against B's reasons, not only B's numbers.
    The
    code-computed anchor and tolerance internals of B's document (the anchor signal,
    the reference three-way, the departures, the tolerance status and validation, the
    reference lean) are deliberately excluded, so the tolerance box and hard cap can
    never reach C through B's output. C sees B's judgment, not the harness's audit of
    it.
    """
    factors = [
        {
            "name": factor["name"],
            "score": factor["score"],
            # C reconciles against B's reasons, not only B's numbers, so the
            # per-factor reasoning travels in the view. This is the point of the field.
            "reasoning": str(factor.get("reasoning", "")),
            "citations": list(factor.get("citations", [])),
            "insufficient_evidence": bool(factor.get("insufficient_evidence", False)),
        }
        for factor in prediction_b["factors"]
    ]
    return {
        "factors": factors,
        "emitted_three_way": prediction_b["emitted_three_way"],
        "emitted_lean": prediction_b["shootout"]["emitted_lean"],
        "justification": prediction_b.get("tolerance", {}).get("justification"),
    }


def build_c_prompt(
    dossier: dict[str, Any],
    prediction_a: dict[str, Any],
    prediction_b: dict[str, Any],
    pool: dict[str, Any],
) -> tuple[str, str]:
    """Build Prediction C's (system, user) messages.

    The system is the verbatim C template with the two tokens filled. The user
    carries the frozen dossier with finding ids, A's forecast and drivers, B's
    forecast and reasoning, and the computed A-B pool, each clearly labelled. It never
    carries the tolerance box or the hard cap, by construction: those live only in
    config and anchor.py and are never serialized into A's or B's views.
    """
    home = str(dossier["home_team"])
    away = str(dossier["away_team"])
    system = fill_template(PREDICTION_C_SYSTEM, home, away)
    findings = dossier_with_finding_ids(dossier)
    user = (
        f"Tie: {home} versus {away}, nominal home {home}.\n\n"
        "FROZEN DOSSIER (admissible findings, cite by id):\n"
        f"{_dumps(findings)}\n\n"
        "PREDICTION A (analytic forecast with drivers):\n"
        f"{_dumps(a_view_for_c(prediction_a))}\n\n"
        "PREDICTION B (analyst panel forecast with reasoning):\n"
        f"{_dumps(b_view_for_c(prediction_b))}\n\n"
        "THE POOL (equal-weight average of A and B, your starting reference):\n"
        f"{_dumps(pool)}\n\n"
        "Return the structured object the schema defines."
    )
    return system, user


# --- the model-call boundary (I/O) ----------------------------------------------


class TransientCallError(Exception):
    """A retryable model-call failure: network, timeout, rate limit, or 5xx."""


class MechanicalError(Exception):
    """A mechanical semantic failure: unparseable, truncated, off-schema, refusal."""


@dataclass(frozen=True)
class RawResponse:
    """One raw structured-call response, the unit the policy reasons over."""

    text: str
    stop_reason: str
    usage: dict[str, int]
    message_id: str | None
    request_id: str | None
    model: str
    effort: str


# Anthropic per-million-token rates for claude-opus-4-8. B and C use no web search.
INPUT_PER_TOKEN = 5.0 / 1_000_000
OUTPUT_PER_TOKEN = 25.0 / 1_000_000
CACHE_WRITE_PER_TOKEN = 6.25 / 1_000_000
CACHE_READ_PER_TOKEN = 0.50 / 1_000_000


def response_usage(usage: Any) -> dict[str, int]:
    """Pull the billable token counts off a response usage object, defensively."""
    return {
        field_name: int(getattr(usage, field_name, 0) or 0)
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }


def actual_cost(usage: dict[str, int]) -> float:
    """USD cost of one B or C response from its actual token usage."""
    return (
        usage.get("input_tokens", 0) * INPUT_PER_TOKEN
        + usage.get("output_tokens", 0) * OUTPUT_PER_TOKEN
        + usage.get("cache_creation_input_tokens", 0) * CACHE_WRITE_PER_TOKEN
        + usage.get("cache_read_input_tokens", 0) * CACHE_READ_PER_TOKEN
    )


def max_call_cost(system: str, user_content: str, max_tokens: int) -> float:
    """A conservative upper bound on one B or C call's cost, computed offline.

    No API and no token-counting call. The prompt token count is over-estimated from
    its character length, and the output is assumed to fill max_tokens entirely, both
    at full (uncached, cache-write) rates. This is the circuit breaker's worst case,
    deliberately above any realistic actual cost.
    """
    est_input_tokens = (len(system) + len(user_content)) / 3.0 + 2000
    return est_input_tokens * CACHE_WRITE_PER_TOKEN + max_tokens * OUTPUT_PER_TOKEN


def invoke_structured(
    client: Any,
    *,
    model: str,
    effort: str,
    system: str,
    user_content: str,
    schema: dict[str, Any],
    max_tokens: int,
    use_prompt_cache: bool,
) -> RawResponse:
    """Make one forced structured-output call and return the raw response.

    THE model-call boundary for B and C: the only function in the live path that
    touches the Anthropic API. One claude-opus-4-8 message, no tools, adaptive
    thinking, effort xhigh, output_config.format bound to the given json schema. No
    temperature, top_p, or top_k: Opus 4.8 rejects them. Retryable transport failures
    are translated to TransientCallError for the policy to retry; a 4xx is re-raised
    as a fatal error to halt on.
    """
    import anthropic

    system_param: Any
    user_param: Any
    if use_prompt_cache:
        system_param = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        user_param = [{"type": "text", "text": user_content}]
    else:
        system_param = system
        user_param = user_content

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_param,
        "thinking": {"type": "adaptive"},
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": schema},
        },
        "messages": [{"role": "user", "content": user_param}],
    }
    if use_prompt_cache:
        kwargs["cache_control"] = {"type": "ephemeral"}

    try:
        response = client.messages.create(**kwargs)
    except anthropic.APIConnectionError as exc:  # includes APITimeoutError
        raise TransientCallError(f"connection error: {exc}") from exc
    except anthropic.RateLimitError as exc:
        raise TransientCallError(f"rate limited: {exc}") from exc
    except anthropic.APIStatusError as exc:
        if getattr(exc, "status_code", 0) >= 500:
            raise TransientCallError(f"server error {exc.status_code}: {exc}") from exc
        raise  # a 4xx is a fatal request error, not retryable: halt.

    text = "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    )
    return RawResponse(
        text=text,
        stop_reason=str(response.stop_reason),
        usage=response_usage(response.usage),
        message_id=getattr(response, "id", None),
        request_id=getattr(response, "_request_id", None),
        model=str(getattr(response, "model", model)),
        effort=effort,
    )


# --- parse, validate, and the retry/re-roll policy (pure) -----------------------


def parse_output(text: str) -> dict[str, Any]:
    """Parse the structured JSON from the model's final text. Raises MechanicalError.

    Forced structured output returns the object as the final text; this tolerates a
    stray code fence but does not prompt-fish. A parse failure is mechanical.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[len("json") :]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise MechanicalError(f"no JSON object in model output: {text[:200]!r}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise MechanicalError(f"unparseable model output: {exc}") from exc


def validate_output(
    raw: RawResponse,
    output_kind: str,
    *,
    require_seven_factors: bool,
) -> dict[str, Any]:
    """Turn a raw response into a validated model-output dict, or raise MechanicalError.

    A refusal or a truncated response is mechanical. A parse failure is mechanical. An
    off-schema object is mechanical. For B, a factor set that is not exactly the seven
    canonical factors once each is mechanical. All mechanical failures get one clean
    re-roll upstream; a substantive rule violation (a coherent but rule-breaking
    prediction) is not caught here, it is caught by the prediction builder downstream.
    """
    if raw.stop_reason == "refusal":
        raise MechanicalError("model refused the call")
    if raw.stop_reason == "max_tokens":
        raise MechanicalError("model output truncated at max_tokens")

    parsed = parse_output(raw.text)
    try:
        jsonschema.validate(instance=parsed, schema=schemas.load_schema(output_kind))
    except jsonschema.ValidationError as exc:
        raise MechanicalError(f"output off-schema: {exc.message}") from exc

    if require_seven_factors:
        names = [factor.get("name") for factor in parsed.get("factors", [])]
        if sorted(n for n in names if n is not None) != sorted(FACTORS) or len(
            names
        ) != len(FACTORS):
            raise MechanicalError(
                f"Prediction B must score exactly the seven factors once each; got "
                f"{names}."
            )
    return parsed


@dataclass
class PolicyHooks:
    """Per-call hooks the live orchestrator supplies to enforce its caps.

    authorize is called before every boundary attempt and may raise to halt the run
    (the per-tie billed-call cap or the session dollar ceiling). record is called
    after every response that actually came back, billed or not retryable, so the
    orchestrator can accumulate actual spend and count the billed call.
    """

    authorize: Callable[[], None]
    record: Callable[[RawResponse], None]
    sleep: Callable[[float], None] = time.sleep
    max_transient_attempts: int = 3
    backoff_base_seconds: float = 1.0
    raw_calls: list[RawResponse] = field(default_factory=list)


def call_with_policy(
    invoke_one: Callable[[], RawResponse],
    *,
    output_kind: str,
    require_seven_factors: bool,
    hooks: PolicyHooks,
) -> dict[str, Any]:
    """Produce one validated model-output dict under the full failure policy.

    invoke_one performs exactly one boundary call (a closure over the frozen request
    and the injected boundary). Transient failures retry the identical call with
    exponential backoff up to three attempts then halt; a returned response that is
    mechanically broken triggers exactly one clean re-roll of the whole attempt on the
    same frozen inputs, and a second mechanical failure halts. Every response that
    comes back is recorded through the hooks, so the billed-call cap and the dollar
    ceiling see every billed call.
    """

    def boundary_with_transient_retry() -> RawResponse:
        last: Exception | None = None
        for attempt in range(hooks.max_transient_attempts):
            hooks.authorize()  # cap and ceiling check, may halt.
            try:
                raw = invoke_one()
            except TransientCallError as exc:
                last = exc
                if attempt < hooks.max_transient_attempts - 1:
                    hooks.sleep(hooks.backoff_base_seconds * (2**attempt))
                    continue
                raise _halted(
                    "transient",
                    f"model call failed {hooks.max_transient_attempts} times: {exc}",
                ) from exc
            hooks.record(raw)
            hooks.raw_calls.append(raw)
            return raw
        raise _halted("transient", f"model call exhausted retries: {last}")

    def one_full_attempt() -> dict[str, Any]:
        raw = boundary_with_transient_retry()
        return validate_output(
            raw, output_kind, require_seven_factors=require_seven_factors
        )

    try:
        return one_full_attempt()
    except MechanicalError:
        try:
            return one_full_attempt()  # one clean re-roll, same frozen inputs.
        except MechanicalError as exc:
            raise _halted(
                "mechanical", f"model output broken on the call and its re-roll: {exc}"
            ) from exc


class PolicyHalt(Exception):
    """A halt-and-surface failure from the model-call policy, with its kind."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _halted(kind: str, message: str) -> PolicyHalt:
    return PolicyHalt(kind, message)
