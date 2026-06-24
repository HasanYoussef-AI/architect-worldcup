"""Knockout round codes and their ordering.

Shared by the fixtures loader, the quarantine gate, and the schemas so the meaning
of "the target round and later" is defined in exactly one place. The order is the
real bracket progression, earliest first.
"""

from __future__ import annotations

# Earliest knockout round first. Index order is the progression order.
ROUND_ORDER = ["R32", "R16", "QF", "SF", "F"]
ROUND_INDEX = {code: index for index, code in enumerate(ROUND_ORDER)}

# Human round names, used by the quarantine gate to spot round mentions in text.
ROUND_LABELS = {
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
