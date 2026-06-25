"""Round codes and their ordering.

Shared by the fixtures loader, the quarantine gate, and the schemas so the meaning
of "the target round and later" is defined in exactly one place. The order is the
real tournament progression, earliest first.

GROUP is included as the earliest round so a group-stage fixture can be a research
target too. With GROUP as the target, every knockout round is strictly later, so
any knockout result or advancement is leakage of a future outcome and is caught.
The group stage runs concurrently, so a different group match dated before the
cutoff is legitimate form, while the target fixture's own result is caught by the
fixture-pair rule.
"""

from __future__ import annotations

# Earliest round first. Index order is the progression order. GROUP precedes the
# knockout rounds.
ROUND_ORDER = ["GROUP", "R32", "R16", "QF", "SF", "F"]
ROUND_INDEX = {code: index for index, code in enumerate(ROUND_ORDER)}

# Human round names, used by the quarantine gate to spot round mentions in text.
ROUND_LABELS = {
    "GROUP": "group stage",
    "R32": "round of 32",
    "R16": "round of 16",
    "QF": "quarter-final",
    "SF": "semi-final",
    "F": "final",
}


def require_round(code: str) -> str:
    """Return the code if it is a known round, else raise a clear error."""
    if code not in ROUND_INDEX:
        raise ValueError(
            f"Unknown knockout round {code!r}. Expected one of {ROUND_ORDER}."
        )
    return code


def is_target_or_later(code: str, target: str) -> bool:
    """True if code is the target round or a later one in the progression."""
    return ROUND_INDEX[require_round(code)] >= ROUND_INDEX[require_round(target)]


def is_strictly_later(code: str, target: str) -> bool:
    """True if code is a strictly later round than the target."""
    return ROUND_INDEX[require_round(code)] > ROUND_INDEX[require_round(target)]
