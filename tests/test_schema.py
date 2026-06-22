"""First verification gate: schema and probability mass.

Validates the most recent predictions JSON against the published schema and
asserts the win probabilities sum to 1.0 within tolerance. If no artifact
exists yet (for example on a fresh CI checkout), the pipeline is run once so
the gate is self-contained.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from architect_wc import pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schema" / "predictions.schema.json"
PREDICTIONS_DIR = REPO_ROOT / "outputs" / "predictions"
SUM_TOLERANCE = 1e-9


def _latest_predictions_file() -> Path | None:
    files = sorted(PREDICTIONS_DIR.glob("predictions_*.json"))
    return files[-1] if files else None


def test_latest_predictions_conform_to_schema() -> None:
    latest = _latest_predictions_file()
    if latest is None:
        pipeline.main()
        latest = _latest_predictions_file()
    assert latest is not None, "pipeline did not produce a predictions artifact"

    document = json.loads(latest.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.validate(instance=document, schema=schema)

    total = sum(item["p_win"] for item in document["predictions"])
    assert abs(total - 1.0) <= SUM_TOLERANCE, f"p_win sums to {total}, expected 1.0"
