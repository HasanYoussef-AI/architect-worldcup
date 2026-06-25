"""Verification gate: the pure parts of the research module (Phase 1).

Hermetic, no network, no API key. Proves the allow-list expansion (static core
plus both teams' verified national sources), the fail-loud on a team with no
national sources, the source allow-list check, the prompt's forward-only and
allow-list instructions, JSON extraction with and without a code fence, and the
cost arithmetic. The single I/O function research_fixture is exercised live in the
one-fixture cost measurement, not here.
"""

from __future__ import annotations

import pytest

from architect_wc import pipeline
from architect_wc.llm import research


def _config():
    return pipeline.load_config()


def test_national_snapshot_loads_both_fixture_teams() -> None:
    config = _config()
    snapshot = config["llm"]["sources"]["national_snapshot"]
    national = research.load_national_sources(snapshot)
    assert "Spain" in national and "Uruguay" in national
    spain_domains = {d for d, _t, _k in national["Spain"]}
    assert "rfef.es" in spain_domains and "marca.com" in spain_domains


def test_core_domain_tiers_from_config() -> None:
    tiers = research.core_domain_tiers(_config())
    # Tier 1 governing bodies and wires, tier 2 international desks.
    assert tiers["the-afc.com"] == 1
    assert tiers["uefa.com"] == 1
    assert tiers["bbc.com"] == 2


def test_build_allowed_domains_expands_core_plus_both_teams() -> None:
    config = _config()
    allowed, domain_tiers = research.build_allowed_domains(config, "Uruguay", "Spain")
    # Crawler-accessible static core present.
    accessible_core = {"uefa.com", "conmebol.com", "the-afc.com", "espn.com"}
    assert accessible_core <= set(allowed)
    # Both teams' crawler-accessible national sources present.
    assert "rfef.es" in allowed
    assert {"auf.org.uy", "elobservador.com.uy"} <= set(allowed)
    # Tiers carried through, federations tier 1, outlets tier 2.
    assert domain_tiers["auf.org.uy"] == 1
    assert domain_tiers["elobservador.com.uy"] == 2
    # Deterministic, sorted, de-duplicated.
    assert allowed == sorted(set(allowed))


def test_build_allowed_domains_drops_crawler_blocked() -> None:
    config = _config()
    allowed, domain_tiers = research.build_allowed_domains(config, "Uruguay", "Spain")
    blocked = set(config["llm"]["sources"]["crawler_blocked"])
    # Nothing the crawler cannot reach is ever passed to the tool.
    assert not (blocked & set(allowed))
    assert not (blocked & set(domain_tiers))
    # Spain's national dailies are crawler-blocked, but its federation remains, so
    # the team is still sourced.
    assert "marca.com" not in allowed and "rfef.es" in allowed


def test_build_allowed_domains_fails_loud_on_unsourced_team() -> None:
    # A team with no verified national rows must stop the run, not be researched on
    # the core alone, since thin sourcing turns into a false neutral.
    with pytest.raises(ValueError, match="No verified national sources"):
        research.build_allowed_domains(_config(), "Atlantis", "Spain")


def test_source_allow_check() -> None:
    config = _config()
    _allowed, domain_tiers = research.build_allowed_domains(config, "Uruguay", "Spain")
    assert research.source_is_allowed("https://www.espn.com/soccer", domain_tiers)
    assert not research.source_is_allowed("https://www.bbc-sport.com/x", domain_tiers)

    clean = {"factors": {"recent_form": [{"claim": "x", "source": "auf.org.uy"}]}}
    research.assert_sources_allowed(clean, domain_tiers)
    dirty = {
        "factors": {"recent_form": [{"claim": "x", "source": "reddit.com/r/soccer"}]}
    }
    with pytest.raises(ValueError, match="off the allow-list"):
        research.assert_sources_allowed(dirty, domain_tiers)


def test_research_prompt_states_the_forward_only_rules() -> None:
    config = _config()
    allowed, domain_tiers = research.build_allowed_domains(config, "Uruguay", "Spain")
    prompt = research.build_research_prompt(
        "Uruguay", "Spain", "2026-06-25", "GROUP", allowed, domain_tiers
    )
    assert "Uruguay" in prompt and "Spain" in prompt
    assert "group stage" in prompt
    assert "auf.org.uy" in prompt and "rfef.es" in prompt
    # The forward-only discipline and the manifest contract are stated.
    assert "Do NOT include the result" in prompt
    assert "coverage" in prompt
    assert "n_raw_hits" in prompt


def test_extract_dossier_json_tolerates_a_code_fence() -> None:
    plain = '{"factors": {}, "coverage": []}'
    assert research.extract_dossier_json(plain) == {"factors": {}, "coverage": []}
    fenced = "```json\n" + plain + "\n```"
    assert research.extract_dossier_json(fenced) == {"factors": {}, "coverage": []}
    prefixed = "Here is the dossier:\n" + plain
    assert research.extract_dossier_json(prefixed) == {"factors": {}, "coverage": []}
    with pytest.raises(ValueError, match="No JSON object"):
        research.extract_dossier_json("no json here")


def test_quarantine_filter_drops_and_records_a_flagged_finding() -> None:
    # The forward-stakes line that names the tie is flagged, then dropped to the
    # raw count rather than aborting the run; the dossier passes clean without it
    # and the manifest records raw greater than admissible for that cell.
    from architect_wc.llm import quarantine
    from architect_wc.llm.weights import FACTORS

    fixture = {"match": 64, "home_team": "Uruguay", "away_team": "Spain"}
    dossier = {"factors": {factor: [] for factor in FACTORS}, "coverage": []}
    dossier["factors"]["strategic_incentives"] = [
        {
            "claim": (
                "Uruguay head into the final group match knowing victory over "
                "Spain may be required to reach the knockout stage."
            ),
            "source": "fifa.com",
            "source_tier": 1,
            "team": "Uruguay",
        },
        {
            "claim": "Spain rotated their squad to manage fitness.",
            "source": "rfef.es",
            "source_tier": 1,
            "team": "Spain",
        },
    ]

    # The stakes line is flagged by the gate, the leakage semantics unchanged.
    assert quarantine.find_violations(dossier, "GROUP", [fixture], cutoff="2026-06-25")

    # Filtering drops it and records the count; the clean finding survives.
    dropped = research.quarantine_filter(dossier, fixture, "GROUP", "2026-06-25")
    assert dropped["strategic_incentives"] == 1
    assert len(dossier["factors"]["strategic_incentives"]) == 1

    # The dossier now passes the gate clean, no abort.
    assert (
        quarantine.find_violations(dossier, "GROUP", [fixture], cutoff="2026-06-25")
        == []
    )

    # The manifest records the drop: raw greater than admissible, with a note.
    research.reconcile_coverage(dossier, quarantined=dropped)
    cell = next(c for c in dossier["coverage"] if c["factor"] == "strategic_incentives")
    assert cell["n_admissible_findings"] == 1
    assert cell["n_raw_hits"] > cell["n_admissible_findings"]
    assert "quarantine dropped" in (cell["note"] or "")


def test_cost_arithmetic() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "web_search_requests": 10,
    }
    cost = research.compute_cost(usage)
    assert cost["input_usd"] == 5.0
    assert cost["output_usd"] == 25.0
    assert cost["web_search_usd"] == 0.1  # 10 searches at $10 per 1000.
    assert abs(cost["total_usd"] - 30.1) < 1e-9
