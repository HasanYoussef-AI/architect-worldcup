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

from architect_wc import artifact, ingest, ratings

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
    """Run the pipeline pass and write the artifact.

    Real match data flows in through Layer 1, is frozen as an immutable
    snapshot, and is filtered by the leakage guard. Layer 2 computes Elo
    ratings from those guarded matches. The win-probability simulation arrives
    in a later phase, so the pipeline still emits equal-probability placeholder
    predictions, while the run log records the data provenance and a ratings
    summary so every run is traceable.
    """
    config = load_config()
    matches, provenance = ingest.load_matches(config)
    team_ratings = ratings.compute_elo(matches, config)

    ratings_summary = {
        "n_teams": len(team_ratings),
        "top_teams": [
            {"team": team, "rating": round(rating, 1)}
            for team, rating in team_ratings[:10]
        ],
    }

    predictions = build_placeholder_predictions(PLACEHOLDER_TEAMS)
    paths = artifact.write_artifact(
        predictions,
        config,
        provenance=provenance,
        ratings_summary=ratings_summary,
    )

    print(
        f"Loaded {provenance['n_matches']} matches from {provenance['snapshot_path']}"
    )
    print(f"Latest match date on or before the cutoff: {provenance['max_date']}")
    print(f"Computed Elo ratings for {len(team_ratings)} teams.")
    print("Top 20 teams by Elo rating:")
    for rank, (team, rating) in enumerate(team_ratings[:20], start=1):
        print(f"{rank:>2}. {team:<24} {rating:8.1f}")
    print(f"Wrote predictions: {paths['predictions']}")
    print(f"Wrote run log: {paths['log']}")


if __name__ == "__main__":
    main()
