# LLM research sources

Controlled inputs for the Phase 8 LLM analyst research call. Like the squad-value
and tournament snapshots, these are committed and dated, not scraped at run time,
so a run is reproducible and the source policy is auditable.

## The allow-list

The research web search is constrained to an allow-list, defined in two places:

- `config.yaml` under `llm.sources.core` holds the static core: the governing
  bodies and confederations (FIFA, UEFA, CONMEBOL, CONCACAF, CAF, AFC, OFC) and a
  small set of top international wires and sports desks, tiered by reputability.
- `national_sources_<date>.csv` here holds the per-team national federations and
  outlets, added per round once the field is known.

For one fixture the research call allows the static core plus both teams' national
rows. The goal is equal-quality sourcing for every team in a tie: a non-European
side gets its own federation and national press, not just the international desks,
so thin sourcing never turns into a false insufficient-evidence neutral.

## Verification

Every domain was verified before being added: federation and outlet home pages
confirmed by web search and fetch. Two lookalike or redirect domains were found
and deliberately excluded, recorded in `config.yaml` under `llm.sources.excluded`:
`bbc-sport.com` (not the real BBC) and `ovacion.net` (a redirect brand domain; the
canonical Uruguayan outlet is `elpais.com.uy`). The Asian confederation's domain
is hyphenated, `the-afc.com`, not `afc.com`.

## Crawler accessibility

The web-search tool can only search domains its crawler can reach; it rejects the
whole request if any allow-listed domain is blocked. Several verified, reputable
sources block the crawler and are recorded under `llm.sources.crawler_blocked` in
`config.yaml`: the major wires and desks (Reuters, AP, BBC, The Guardian, The
Athletic via nytimes.com) and the Spanish dailies Marca and AS and Uruguay's El
Pais. They stay listed so the verification is preserved, and `build_allowed_domains`
filters them out of the live allow-list. The working core therefore leans on the
governing bodies and ESPN; replacing the blocked national dailies with verified,
crawler-accessible national outlets is a follow-up before the knockout rounds.

## Excluded by design

Wikipedia and other open-edit wikis, forums, and all betting, odds, and tipster
sites are never listed. Because the web search only ever queries listed domains,
the whole class of market and open-edit sources is excluded by construction, which
keeps the forecast independent of the betting market and of unvetted sources.

## Tiers

- Tier 1: official federations and confederations, and top international wires
  (Reuters, AP).
- Tier 2: established international sports desks (BBC, ESPN, The Guardian, The
  Athletic via nytimes.com) and reputable national sports press.

The snapshot is extended as later rounds are researched; each addition is verified
the same way and committed before the round it informs.
