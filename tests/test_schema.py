"""First verification gate: schema and probability mass.

Validates the most recent predictions JSON against the published schema and
asserts the win probabilities sum to 1.0 within tolerance. If no artifact
exists yet (for example on a fresh CI checkout), one is generated in place
without the network so the gate stays self-contained and hermetic.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from architect_wc import artifact, pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schema" / "predictions.schema.json"
PREDICTIONS_DIR = REPO_ROOT / "outputs" / "predictions"
SUM_TOLERANCE = 1e-9


def _latest_predictions_file() -> Path | None:
    files = sorted(PREDICTIONS_DIR.glob("predictions_*.json"))
    return files[-1] if files else None


def test_latest_predictions_conform_to_schema(tmp_path: Path) -> None:
    latest = _latest_predictions_file()
    if latest is None:
        # No real artifact yet. Generate one hermetically, without touching the
        # network, so the schema gate does not depend on data ingestion.
        config = pipeline.load_config()
        predictions = pipeline.build_placeholder_predictions(pipeline.PLACEHOLDER_TEAMS)
        paths = artifact.write_artifact(
            predictions,
            config,
            predictions_dir=tmp_path / "predictions",
            logs_dir=tmp_path / "logs",
        )
        latest = paths["predictions"]

    document = json.loads(latest.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    jsonschema.validate(instance=document, schema=schema)

    total = sum(item["p_win"] for item in document["predictions"])
    assert abs(total - 1.0) <= SUM_TOLERANCE, f"p_win sums to {total}, expected 1.0"
