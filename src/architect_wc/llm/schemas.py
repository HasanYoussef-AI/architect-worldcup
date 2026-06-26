"""JSON Schema validation for the dossier and Predictions A, B, C.

Mirrors the existing schema gate (tests/test_schema.py): load the published schema
file and validate the document against it with jsonschema. Centralising the kind
to filename map here keeps every consumer validating against the same contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

SCHEMA_DIR = Path(__file__).resolve().parents[3] / "schema"

SCHEMA_FILES = {
    "dossier": "llm_dossier.schema.json",
    "A": "llm_prediction_a.schema.json",
    "B": "llm_prediction_b.schema.json",
    "C": "llm_prediction_c.schema.json",
    # The real per-match Prediction B contract (Phase 2). The "B" entry above is the
    # earlier per-round-with-ties placeholder, kept until A and C move to per-match.
    "prediction_b": "prediction_b.schema.json",
}


def load_schema(kind: str) -> dict[str, Any]:
    """Load one published schema by kind: dossier, A, B, or C."""
    if kind not in SCHEMA_FILES:
        raise ValueError(
            f"Unknown schema kind {kind!r}. Expected {list(SCHEMA_FILES)}."
        )
    path = SCHEMA_DIR / SCHEMA_FILES[kind]
    return json.loads(path.read_text(encoding="utf-8"))


def validate_document(document: dict[str, Any], kind: str) -> None:
    """Validate a document against its schema, raising on any mismatch."""
    jsonschema.validate(instance=document, schema=load_schema(kind))
