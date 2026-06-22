"""Entry point.

The single command-line entry point for a run. Reads config, runs the layered
model, and writes a versioned artifact with provenance. For Phase 0 this is a
walking skeleton: it emits placeholder predictions with equal win
probabilities so the output contract and verification gates are exercised end
to end before any model logic exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from architect_wc import artifact

CONFIG_PATH = Path("config.yaml")

# Placeholder slate until the model layers are built. Equal win probabilities
# exercise the output contract without implying any real forecast.
PLACEHOLDER_TEAMS = [
    "Argentina",
    "Brazil",
    "France",
    "England",
    "Spain",
    "Germany",
    "Portugal",
    "Netherlands",
]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the run configuration from YAML."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_placeholder_predictions(teams: list[str]) -> list[dict[str, Any]]:
    """Build equal-probability placeholder predictions that sum to 1.0."""
    p_win = 1.0 / len(teams)
    return [{"team": team, "p_win": p_win} for team in teams]


def main() -> None:
    """Run the dummy pipeline pass and write the artifact."""
    config = load_config()
    predictions = build_placeholder_predictions(PLACEHOLDER_TEAMS)
    paths = artifact.write_artifact(predictions, config)
    print(f"Wrote predictions: {paths['predictions']}")
    print(f"Wrote run log: {paths['log']}")


if __name__ == "__main__":
    main()
