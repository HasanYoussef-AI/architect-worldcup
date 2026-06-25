"""Phase 1 of the LLM analyst layer: the research call that builds a per-match
dossier from allow-listed web search, then gates it.

This is the only module that calls the Anthropic API in the LLM layer so far, and
it calls it for research only: a single claude-opus-4-8 message with the
web_search_20260209 server tool, its allowed_domains pinned to the per-fixture
allow-list and its max_uses to the per-match budget. The model researches the
fixed seven-factor taxonomy and returns a JSON dossier; this module does not use
forced structured output, it instructs the schema in the prompt and validates the
result with jsonschema, the same prompt-instructed discipline the skeleton uses.

After the call, three gates run on the produced dossier, fail-loud:

  1. jsonschema validation against the published dossier schema.
  2. the round-aware, date-bounded quarantine gate, iterating the dossier, which
     forbids the fixture's own result and any later-round result.
  3. assert_coverage_complete, the coverage manifest gate, and an allow-list check
     that every admissible source sits on an allow-listed domain.

The pure parts (allow-list expansion, prompt building, JSON extraction, the source
allow check) are separated from the single I/O function so they are offline
testable with no network and no key.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from architect_wc.llm import coverage, quarantine, rounds, schemas
from architect_wc.llm.weights import FACTORS

# Anthropic per-million-token rates for claude-opus-4-8, for the cost estimate.
# The web search surcharge is the published per-search rate; the Console is the
# authoritative figure and this estimate is checked against it.
INPUT_PER_TOKEN = 5.0 / 1_000_000
OUTPUT_PER_TOKEN = 25.0 / 1_000_000
CACHE_WRITE_PER_TOKEN = 6.25 / 1_000_000
CACHE_READ_PER_TOKEN = 0.50 / 1_000_000
WEB_SEARCH_PER_USE = 10.0 / 1_000  # $10 per 1,000 searches.


# --- allow-list expansion (pure) ------------------------------------------------


def load_national_sources(path: Any) -> dict[str, list[tuple[str, int, str]]]:
    """Read the committed national-sources snapshot into team to (domain, tier, kind).

    Raises a clear error on a missing file or column. The snapshot is the verified,
    per-team half of the allow-list, expanded per round.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"National sources snapshot not found: {path}. Expected a committed, "
            f"dated CSV with columns team, kind, domain, tier."
        )
    frame = pd.read_csv(path, comment="#")
    required = ("team", "kind", "domain", "tier")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"National sources {path} is missing expected columns: {missing}. "
            f"Found {list(frame.columns)}."
        )
    by_team: dict[str, list[tuple[str, int, str]]] = {}
    for row in frame.itertuples(index=False):
        team = str(row.team).strip()
        by_team.setdefault(team, []).append(
            (str(row.domain).strip().lower(), int(row.tier), str(row.kind).strip())
        )
    return by_team


def core_domain_tiers(config: dict[str, Any]) -> dict[str, int]:
    """The static-core domains and their tiers from config.llm.sources.core."""
    core = ((config.get("llm", {}) or {}).get("sources", {}) or {}).get(
        "core", {}
    ) or {}
    tiers: dict[str, int] = {}
    for tier_key, domains in core.items():
        tier = int(str(tier_key).removeprefix("tier"))
        for domain in domains or []:
            tiers[str(domain).strip().lower()] = tier
    return tiers


def build_allowed_domains(
    config: dict[str, Any], home_team: str, away_team: str
) -> tuple[list[str], dict[str, int]]:
    """Return the allowed_domains list and a domain to tier map for one fixture.

    The static core plus both teams' national federations and outlets, minus any
    domain on the crawler_blocked record, since the web-search tool rejects domains
    its crawler cannot access. Fails loud if a fixture team has no national rows in
    the snapshot, because researching a team on the core alone is exactly the thin
    sourcing that turns into a false insufficient-evidence neutral. The list is
    sorted and de-duplicated so the request, and therefore the prompt cache key, is
    deterministic.
    """
    llm = config.get("llm", {}) or {}
    sources = llm.get("sources", {}) or {}
    snapshot = sources.get("national_snapshot")
    national = load_national_sources(snapshot)

    tiers = core_domain_tiers(config)
    for team in (home_team, away_team):
        rows = national.get(team)
        if not rows:
            raise ValueError(
                f"No verified national sources for {team!r} in {snapshot}. Add and "
                f"verify the team's federation and national outlets before "
                f"researching it, so both sides of the tie are sourced equally."
            )
        for domain, tier, _kind in rows:
            tiers[domain] = tier

    # Drop domains the web-search crawler cannot reach, kept on record but never
    # passed to the tool, which would reject the whole request.
    for blocked in sources.get("crawler_blocked", []) or []:
        tiers.pop(str(blocked).strip().lower(), None)

    return sorted(tiers), tiers


def source_is_allowed(source: str, domain_tiers: dict[str, int]) -> bool:
    """True if the source names an allow-listed domain (substring match, lowercased)."""
    lowered = str(source).lower()
    return any(domain in lowered for domain in domain_tiers)


def assert_sources_allowed(
    dossier: dict[str, Any], domain_tiers: dict[str, int]
) -> None:
    """Raise if any admissible finding cites a source off the allow-list.

    The admissible findings must all sit on allow-listed domains; anything else
    should have been filtered into the raw-hit count, not kept. Fail loud rather
    than silently drop, so an off-list citation is a visible error.
    """
    offending: list[str] = []
    for factor, entries in (dossier.get("factors", {}) or {}).items():
        for index, entry in enumerate(entries or []):
            if not source_is_allowed(entry.get("source", ""), domain_tiers):
                offending.append(f"{factor}#{index}: {entry.get('source')!r}")
    if offending:
        raise ValueError(
            "Dossier cites sources off the allow-list:\n  " + "\n  ".join(offending)
        )


# --- prompt building (pure) -----------------------------------------------------

_SYSTEM = (
    "You are a forward-only football research analyst. You research one upcoming "
    "fixture and return a structured JSON dossier of evidence, nothing else. You "
    "never state, imply, or infer the result of the fixture being researched, and "
    "you never include any result, scoreline, advancement, or elimination from a "
    "later round than the one being researched. You use only the web search tool, "
    "restricted to the allow-listed domains. You never use betting, odds, tipster, "
    "prediction-market, open-edit (wiki), or forum sources."
)


def _factor_guide() -> str:
    """A one-line research brief per factor, in the canonical order."""
    return "\n".join(
        f"  - {factor}: {hint}"
        for factor, hint in {
            "squad_availability": "injuries, suspensions, returns, fitness",
            "recent_form": "results and performances in the matches before the cutoff",
            "tactical_matchup": "styles, formations, and how the two sides match up",
            "coaching_staff": "head coach and backroom stability",
            "strategic_incentives": "what each side needs, rotation, motivation",
            "psychological_momentum": "confidence, pressure, run of form narrative",
            "historical_head_to_head": "past meetings between the teams, dated",
        }.items()
    )


def build_research_prompt(
    home_team: str,
    away_team: str,
    as_of_date: str,
    round_code: str,
    allowed_domains: list[str],
    domain_tiers: dict[str, int],
) -> str:
    """Build the user prompt for the research call. Pure: no I/O, no key."""
    round_label = rounds.ROUND_LABELS[rounds.require_round(round_code)]
    tier_lines = "\n".join(
        f"  - tier {domain_tiers[domain]}: {domain}" for domain in allowed_domains
    )
    return f"""Research the upcoming {round_label} fixture {home_team} versus \
{away_team}, as of {as_of_date} (the day before kickoff). Use only the web search \
tool, and only these allow-listed domains, which carry the source tier shown:

{tier_lines}

Source both teams equally. Cover all seven factors, in this canonical order:
{_factor_guide()}

Forward-only discipline, strictly enforced downstream by an automated gate that \
will reject the dossier and fail the run if violated:
  - Do NOT include the result, scoreline, or outcome of {home_team} versus \
{away_team}. It has not been played.
  - Do NOT include any result, advancement, or elimination from a round later than \
the {round_label}.
  - Group-stage results from before {as_of_date} are legitimate form evidence.
  - For a historical head-to-head fact, give the meeting_date; only meetings \
strictly before {as_of_date} are admissible.
  - No betting, odds, tipster, prediction-market, wiki, or forum content.

Return ONLY a single JSON object, no prose and no code fence, with exactly these \
keys:
  "factors": an object with all seven factor names as keys, each mapping to an \
array of finding objects. A finding object has:
      "claim" (string, the fact in plain language),
      "source" (string, the allow-listed domain or a full URL on it),
      "source_tier" (integer 1, 2, or 3, matching the domain's tier above),
      "team" (string or null, the team it concerns),
      "meeting_date" (string YYYY-MM-DD or null, only for historical_head_to_head),
      "retrieved_at" (string or null).
    Put only admissible findings here: on an allow-listed domain, not a forbidden \
result, surviving the forward-only rules above. If a factor has none, use an empty \
array.
  "coverage": an array of exactly seven objects, one per factor, each with:
      "factor" (the factor name),
      "n_raw_hits" (integer, how many relevant search results you saw for this \
factor, including ones you discarded for being off-list or a forbidden result),
      "n_admissible_findings" (integer, equal to the number you placed in \
factors[factor]),
      "researchable" (boolean, false only if no allow-listed source covered this \
factor for this fixture at all),
      "status" (one of "has_findings", "queried_no_findings", \
"quarantined_or_filtered", "unresearchable"),
      "note" (string or null, e.g. why unresearchable or what was filtered out).
    Set status honestly: has_findings if you kept at least one; \
quarantined_or_filtered if raw hits came back but you discarded them all; \
queried_no_findings if you searched and nothing came back; unresearchable if no \
allow-listed source covered the factor.
"""


# --- JSON extraction (pure) -----------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def extract_dossier_json(text: str) -> dict[str, Any]:
    """Parse the dossier JSON from the model's final text, tolerating a code fence."""
    cleaned = _FENCE.sub("", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in research output: {text[:200]!r}")
    return json.loads(cleaned[start : end + 1])


def quarantine_filter(
    dossier: dict[str, Any],
    fixture: dict[str, Any],
    round_code: str,
    as_of_date: str,
) -> dict[str, int]:
    """Drop quarantine-flagged findings from the admissible set, filter and record.

    Refinement A's behaviour: a flagged finding is not kept and does not abort the
    run. It is removed from its factor array and, through reconcile_coverage, shows
    up as a raw hit that did not become an admissible finding, so the manifest
    records the drop rather than passing it off as a clean neutral. The leakage
    semantics are unchanged, the gate flags exactly what it flagged before,
    including current-tournament forward references like a stakes line that names
    the tie; only the response changes from aborting to filter-and-record. Returns
    the count dropped per factor so the manifest can record it.
    """
    violations = quarantine.find_violations(
        dossier, round_code, fixtures=[fixture], cutoff=as_of_date
    )
    drop_indices: dict[str, set[int]] = {}
    for violation in violations:
        drop_indices.setdefault(violation.factor, set()).add(violation.index)

    dropped: dict[str, int] = {}
    factors = dossier.get("factors", {}) or {}
    for factor, indices in drop_indices.items():
        entries = factors.get(factor, []) or []
        kept = [entry for i, entry in enumerate(entries) if i not in indices]
        dropped[factor] = len(entries) - len(kept)
        factors[factor] = kept
    return dropped


def reconcile_coverage(
    dossier: dict[str, Any], quarantined: dict[str, int] | None = None
) -> dict[str, Any]:
    """Rebuild a complete, internally-consistent coverage manifest, one cell per
    factor.

    The admissible count is definitional, the number of findings that survived in
    factors, so it is taken from the factor array rather than trusted from the
    writer; the status is derived from that count, the model's raw-hit count, and
    its researchable flag. The model only supplies what it alone knows: how many
    raw hits it saw, and whether the cell was researchable. Any finding the
    quarantine filter dropped is added back into the raw count (raw is at least
    admissible plus dropped) and noted, so a dropped leakage-flagged fact is
    recorded as a raw hit that did not survive, not hidden. The coverage gate then
    verifies the result, and proves on bad input that it catches inconsistencies;
    here it passes by construction.
    """
    quarantined = quarantined or {}
    factors = dossier.get("factors", {}) or {}
    provided = {cell.get("factor"): cell for cell in (dossier.get("coverage") or [])}
    cells = []
    for factor in FACTORS:
        n_admissible = len(factors.get(factor, []) or [])
        cell = provided.get(factor, {})
        researchable = bool(cell.get("researchable", True))
        dropped = int(quarantined.get(factor, 0))
        n_raw = int(cell.get("n_raw_hits", n_admissible) or 0)
        n_raw = max(n_raw, n_admissible + dropped)
        note = cell.get("note")
        if dropped:
            drop_note = (
                f"quarantine dropped {dropped} leakage-flagged finding(s), recorded "
                f"as raw hits not admissible"
            )
            note = f"{drop_note}. {note}" if note else drop_note
        cells.append(
            {
                "factor": factor,
                "n_raw_hits": n_raw,
                "n_admissible_findings": n_admissible,
                "researchable": researchable,
                "status": coverage.derive_status(n_raw, n_admissible, researchable),
                "note": note,
            }
        )
    dossier["coverage"] = cells
    return dossier


# --- cost (pure) ----------------------------------------------------------------


def compute_cost(usage: dict[str, int]) -> dict[str, float]:
    """Estimate the run cost from accumulated usage, claude-opus-4-8 rates.

    Returns the per-component breakdown and the total in USD. The web search line
    uses the published per-search rate; the Console figure is authoritative and
    this estimate is reported alongside it.
    """
    input_cost = usage.get("input_tokens", 0) * INPUT_PER_TOKEN
    output_cost = usage.get("output_tokens", 0) * OUTPUT_PER_TOKEN
    cache_write_cost = (
        usage.get("cache_creation_input_tokens", 0) * CACHE_WRITE_PER_TOKEN
    )
    cache_read_cost = usage.get("cache_read_input_tokens", 0) * CACHE_READ_PER_TOKEN
    search_cost = usage.get("web_search_requests", 0) * WEB_SEARCH_PER_USE
    total = input_cost + output_cost + cache_write_cost + cache_read_cost + search_cost
    return {
        "input_usd": round(input_cost, 6),
        "output_usd": round(output_cost, 6),
        "cache_write_usd": round(cache_write_cost, 6),
        "cache_read_usd": round(cache_read_cost, 6),
        "web_search_usd": round(search_cost, 6),
        "total_usd": round(total, 6),
    }


# --- the research call (I/O) ----------------------------------------------------


def _accumulate_usage(total: dict[str, int], usage: Any) -> None:
    """Add one response's usage onto the running totals, defensively."""
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        total[field] = total.get(field, 0) + (getattr(usage, field, 0) or 0)
    server = getattr(usage, "server_tool_use", None)
    if server is not None:
        total["web_search_requests"] = total.get("web_search_requests", 0) + (
            getattr(server, "web_search_requests", 0) or 0
        )


def gate_dossier(
    dossier: dict[str, Any],
    fixture: dict[str, Any],
    round_code: str,
    as_of_date: str,
    domain_tiers: dict[str, int],
) -> None:
    """Run the dossier gates, fail-loud, in order.

    Schema, then the round-aware date-bounded quarantine, then the coverage
    manifest, then the allow-list check. Kept separate from production so a caller
    can inspect the produced dossier and its cost even when a gate would reject it,
    which is what the cost measurement does. In normal use research_fixture runs
    this immediately after producing the dossier.
    """
    schemas.validate_document(dossier, "dossier")
    quarantine.assert_no_dossier_leakage(
        dossier, round_code, fixtures=[fixture], cutoff=as_of_date
    )
    coverage.assert_coverage_complete(dossier)
    assert_sources_allowed(dossier, domain_tiers)


def produce_dossier(
    config: dict[str, Any],
    fixture: dict[str, Any],
    as_of_date: str,
    round_code: str,
    *,
    client: Any = None,
    git_sha: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Run the research call and build the dossier and meta, WITHOUT gating.

    Builds the per-fixture allow-list, calls claude-opus-4-8 with the web search
    tool bounded to that allow-list and the per-match budget, parses the dossier,
    fills its envelope, provenance, and cost, and rebuilds the coverage manifest.
    Returns {"dossier", "meta", "domain_tiers"}; the gates are deliberately not
    run here. The client is injected so the pure path is testable; in normal use it
    is anthropic.Anthropic().
    """
    from datetime import UTC, datetime

    from architect_wc import artifact

    rounds.require_round(round_code)
    home = str(fixture["home_team"]).strip()
    away = str(fixture["away_team"]).strip()
    match = int(fixture["match"])

    llm = config.get("llm", {}) or {}
    model = llm.get("model", "claude-opus-4-8")
    effort = llm.get("effort_floor", "high")
    tool_type = llm.get("web_search_tool", "web_search_20260209")
    max_uses = int(llm.get("max_search_uses", 14))

    allowed_domains, domain_tiers = build_allowed_domains(config, home, away)
    user_prompt = build_research_prompt(
        home, away, as_of_date, round_code, allowed_domains, domain_tiers
    )

    if client is None:
        import anthropic

        client = anthropic.Anthropic()

    tools = [
        {
            "type": tool_type,
            "name": "web_search",
            "max_uses": max_uses,
            "allowed_domains": allowed_domains,
        }
    ]

    usage_total: dict[str, int] = {}
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    response = None
    for _ in range(12):  # bound the server-tool pause_turn loop.
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=_SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            tools=tools,
            messages=messages,
        )
        _accumulate_usage(usage_total, response.usage)
        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue
        break

    if response is None:
        raise RuntimeError("Research call produced no response.")
    if response.stop_reason == "refusal":
        raise RuntimeError("Research call was refused by the model safety system.")

    final_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    dossier = extract_dossier_json(final_text)

    # Fill the envelope and provenance around the model's factors and coverage.
    dossier["round"] = round_code
    dossier["match"] = match
    dossier["home_team"] = home
    dossier["away_team"] = away
    dossier["as_of_date"] = as_of_date
    dossier.setdefault("committed_at", None)
    dossier["git_sha"] = git_sha or artifact.get_git_sha()
    dossier["model_version"] = model_version or artifact.MODEL_VERSION
    cost = compute_cost(usage_total)
    dossier["provenance"] = {
        "model": model,
        "effort": effort,
        "web_search_tool": tool_type,
        "max_search_uses": max_uses,
        "allowed_domains": allowed_domains,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "usage": usage_total,
        "cost_usd_estimate": cost,
        "stop_reason": response.stop_reason,
        "request_id": getattr(response, "_request_id", None),
    }

    # Filter quarantine-flagged findings (Refinement A: drop and record, never
    # abort), then rebuild the coverage manifest from the surviving findings so a
    # dropped fact is recorded as a raw hit that did not survive.
    dropped = quarantine_filter(dossier, fixture, round_code, as_of_date)
    reconcile_coverage(dossier, quarantined=dropped)

    meta = {
        "usage": usage_total,
        "cost": cost,
        "n_web_searches": usage_total.get("web_search_requests", 0),
        "allowed_domains": allowed_domains,
        "quarantine_dropped": dropped,
    }
    # Persist the raw produced dossier and its cost before any gate runs, so billed
    # work is never discarded if a gate then rejects the dossier.
    meta["dossier_path"] = str(artifact.write_dossier(dossier, meta=meta))
    return {"dossier": dossier, "meta": meta, "domain_tiers": domain_tiers}


def research_fixture(
    config: dict[str, Any],
    fixture: dict[str, Any],
    as_of_date: str,
    round_code: str,
    *,
    client: Any = None,
    git_sha: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Produce the dossier for one fixture and gate it, fail-loud.

    The production path (research call, dossier assembly, coverage reconciliation)
    is in produce_dossier; the gates are in gate_dossier. This composes the two so
    the normal path stays a single call that raises on any gate failure. Returns
    {"dossier", "meta"}.
    """
    produced = produce_dossier(
        config,
        fixture,
        as_of_date,
        round_code,
        client=client,
        git_sha=git_sha,
        model_version=model_version,
    )
    gate_dossier(
        produced["dossier"], fixture, round_code, as_of_date, produced["domain_tiers"]
    )
    return {"dossier": produced["dossier"], "meta": produced["meta"]}
