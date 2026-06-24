"""R3 quarantine gate, the leakage guard for the LLM phase.

This is the assert_no_leakage analog for the research dossier. It is round aware,
keyed to the target round, not a blanket scoreline stripper. Pre-cutoff group
stage results are legitimate inputs to the form and tactical factors and must be
preserved. The gate forbids only results and advancement status for the target
round and later:

  - A group-stage scoreline from before the cutoff is allowed.
  - Any scoreline, win, elimination, or advancement statement for a team in the
    context of a target-round or later fixture is forbidden and fails the gate.
  - All betting and prediction-market prices, including Polymarket, are forbidden
    by design, so the forecast stays independent of the market.

The gate is deliberately conservative: when a fact carries a result or status
signal in a target-or-later knockout context it fails loud, because a false
positive (rejecting a borderline fact) is far safer than a false negative
(leaking a result). Reaching the target round itself is allowed, since the bracket
is known before the round kicks off; reaching a strictly later round is not, since
that would reveal an intervening knockout result.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any, NamedTuple

from architect_wc.llm import rounds

# Patterns that name each knockout round in free text. The final is matched after
# the quarter-final and semi-final compounds are removed, so "quarter-final" does
# not falsely register as "final".
_ROUND_PATTERNS = {
    "R32": re.compile(r"round of 32|round-of-32|\br32\b"),
    "R16": re.compile(r"round of 16|round-of-16|last 16|\br16\b"),
    "QF": re.compile(r"quarter[\s-]*finals?|last 8|\bqf\b"),
    "SF": re.compile(r"semi[\s-]*finals?|last 4|\bsf\b"),
}
_COMPOUND_FINAL = re.compile(r"quarter[\s-]*finals?|semi[\s-]*finals?")
_FINAL_PATTERN = re.compile(
    r"\bfinals?\b|\bchampions?\b|won the (?:world )?cup|"
    r"lift(?:ed)? the (?:world cup|trophy)"
)

# A 90-minute or knockout result: a scoreline, a win or loss verb, a shootout, or
# an elimination. Elimination is a result (it is the outcome of a knockout tie).
_SCORELINE = re.compile(r"\b\d+\s*[-–—:]\s*\d+\b")
_RESULT_VERBS = re.compile(
    r"\b(?:beat|beats|defeat|defeated|thrash|thrashed|edged|won|win|wins|lost|"
    r"loses|knocked out|eliminat\w*|on penalties|won on pen\w*|"
    r"out of the (?:tournament|cup|competition))\b"
)
# An advancement statement and the round reached. "advanced to the round of 16"
# reveals the result of the round before it; reaching the target round itself is
# group-stage or prior-round knowledge and is allowed. Bare "into" is excluded as
# too generic; advancement only ever triggers a violation alongside a strictly
# later round mention, so the verbs stay concrete.
_ADVANCE_VERBS = re.compile(
    r"\b(?:advanc\w*|progress\w*|through to|booked\b|reach\w*|qualif\w*\s+for)\b"
)

# The historical head-to-head factor legitimately holds past results between the
# two fixture teams, so the both-teams-named rule cannot discriminate there on text
# alone. The exemption is date-bounded: it applies only to a meeting dated strictly
# before the cutoff. A head-to-head entry with no date, or dated on or after the
# cutoff, is not a past meeting and is not exempt, so a current-tournament result
# framed as a head-to-head is still caught. The round-context rules apply to this
# factor too, so a head-to-head naming the target round or later with a result is
# caught regardless of date.
_FIXTURE_PAIR_EXEMPT_FACTORS = {"historical_head_to_head"}


def _is_pre_cutoff_meeting(meeting_date: Any, cutoff: Any) -> bool:
    """True only if meeting_date is a valid date strictly before the cutoff.

    A missing, null, unparseable, or on-or-after-cutoff date returns False, so the
    head-to-head exemption is denied and the entry falls under the fixture-pair
    rule.
    """
    if not meeting_date or not cutoff:
        return False
    try:
        return date.fromisoformat(str(meeting_date)) < date.fromisoformat(str(cutoff))
    except ValueError:
        return False


# Betting and prediction-market content, forbidden in every round by design. Kept
# focused on market and betting terms so legitimate form stats (possession, pass
# completion) are not swept up.
_MARKET_PATTERN = re.compile(
    r"\bpolymarket\b|\bbetfair\b|\bbet365\b|\bbookmaker\b|betting odds|"
    r"\bmoneyline\b|implied probabilit\w*|decimal odds|title odds|"
    r"odds of\b|to win outright|prediction market"
)


class Violation(NamedTuple):
    """One quarantine hit: where it was found and why it is forbidden."""

    factor: str
    index: int
    reason: str
    claim: str


class LeakageError(Exception):
    """Raised, fail loud, when the dossier carries forbidden content."""

    def __init__(self, violations: list[Violation]) -> None:
        self.violations = violations
        lines = [
            f"  [{v.factor}#{v.index}] {v.reason}: {v.claim!r}" for v in violations
        ]
        super().__init__(
            "Dossier quarantine failed, forbidden content found:\n" + "\n".join(lines)
        )


def _mentioned_rounds(text: str) -> set[str]:
    """Return the set of knockout rounds named in the text."""
    found = {code for code, pattern in _ROUND_PATTERNS.items() if pattern.search(text)}
    stripped = _COMPOUND_FINAL.sub(" ", text)
    if _FINAL_PATTERN.search(stripped):
        found.add("F")
    return found


def _fixture_pairs(fixtures: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    """Lowercased (home, away) team-name pairs for the round's fixtures."""
    if not fixtures:
        return []
    pairs = []
    for fixture in fixtures:
        home = str(fixture.get("home_team", "")).strip().lower()
        away = str(fixture.get("away_team", "")).strip().lower()
        if home and away:
            pairs.append((home, away))
    return pairs


def _entry_violation(
    text: str,
    factor: str,
    target_round: str,
    fixture_pairs: list[tuple[str, str]],
    meeting_date: Any,
    cutoff: Any,
) -> str | None:
    """Return a reason if the entry is forbidden for the target round, else None."""
    lowered = text.lower()

    if _MARKET_PATTERN.search(lowered):
        return "betting or prediction-market content"

    mentioned = _mentioned_rounds(lowered)
    has_result = bool(_SCORELINE.search(lowered) or _RESULT_VERBS.search(lowered))
    has_advance = bool(_ADVANCE_VERBS.search(lowered))

    # A result statement that names the target round or a later one.
    for code in mentioned:
        if rounds.is_target_or_later(code, target_round) and has_result:
            return f"result for {code}, the target round or later"
        # Advancing to a strictly later round reveals an intervening result.
        if rounds.is_strictly_later(code, target_round) and has_advance:
            return f"advancement to {code}, a later round than the target"

    # A result or advancement statement naming both teams of a target-or-later
    # fixture, even without a round word, is leakage of that fixture's outcome.
    if (has_result or has_advance) and fixture_pairs:
        names_fixture = any(
            home in lowered and away in lowered for home, away in fixture_pairs
        )
        if names_fixture:
            if factor in _FIXTURE_PAIR_EXEMPT_FACTORS:
                # Head-to-head: exempt only for a meeting strictly before the cutoff.
                if not _is_pre_cutoff_meeting(meeting_date, cutoff):
                    return (
                        "head-to-head between the fixture teams with no meeting "
                        "date strictly before the cutoff"
                    )
            else:
                return "result or status for a target-round fixture matchup"

    return None


def find_violations(
    dossier: dict[str, Any],
    target_round: str,
    fixtures: list[dict[str, Any]] | None = None,
    cutoff: Any = None,
) -> list[Violation]:
    """Scan the dossier and return every quarantine violation, empty if clean.

    fixtures, when given, are the target round's real pairings, so a result that
    names both teams of a tie is caught even with no round word in the text. cutoff
    is the round cutoff date that bounds the head-to-head exemption; it defaults to
    the dossier's own as_of_date, so a head-to-head meeting is preserved only if it
    is dated strictly before the cutoff.
    """
    rounds.require_round(target_round)
    fixture_pairs = _fixture_pairs(fixtures)
    if cutoff is None:
        cutoff = dossier.get("as_of_date")
    factors = dossier.get("factors", {}) or {}

    violations: list[Violation] = []
    for factor, entries in factors.items():
        for index, entry in enumerate(entries or []):
            text = " ".join(
                str(entry.get(field, ""))
                for field in ("claim", "source", "team")
                if entry.get(field)
            )
            reason = _entry_violation(
                text,
                factor,
                target_round,
                fixture_pairs,
                entry.get("meeting_date"),
                cutoff,
            )
            if reason is not None:
                violations.append(
                    Violation(factor, index, reason, str(entry.get("claim", "")))
                )
    return violations


def assert_no_dossier_leakage(
    dossier: dict[str, Any],
    target_round: str,
    fixtures: list[dict[str, Any]] | None = None,
    cutoff: Any = None,
) -> None:
    """Raise LeakageError if the dossier carries forbidden content. The R3 gate.

    Mirrors calibrate.assert_no_leakage: any hit is a failure, not a warning, so it
    raises rather than logs. A clean dossier returns None.
    """
    violations = find_violations(dossier, target_round, fixtures, cutoff)
    if violations:
        raise LeakageError(violations)
