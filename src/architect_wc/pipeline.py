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

from architect_wc import artifact, ingest, model, ratings, squad

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

# Fixtures between strong teams, printed as an eyeball check on the goal model.
SAMPLE_FIXTURES = [
    ("Argentina", "France"),
    ("Brazil", "Spain"),
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
    ratings, Layer 4 nudges those ratings by current squad value, and Layer 3
    fits the Dixon-Coles goal model on the guarded matches. The order is compute
    Elo, then apply the squad adjustment, then fit the goal model. Full
    tournament simulation is the next layer, so the pipeline still emits
    equal-probability placeholder predictions, while the run log records the data
    provenance and a ratings summary so every run is traceable.
    """
    config = load_config()
    matches, provenance = ingest.load_matches(config)

    team_ratings = ratings.compute_elo(matches, config)

    squad_config = config.get("squad", {}) or {}
    squad_values = squad.load_squad_values(squad_config["snapshot"])
    adjusted_ratings = squad.adjust_ratings(team_ratings, squad_values, config)
    adjusted_by_team = dict(adjusted_ratings)

    ratings_summary = {
        "n_teams": len(team_ratings),
        "n_squad_values": len(squad_values),
        "top_teams": [
            {
                "team": team,
                "elo": round(rating, 1),
                "squad_nudge": round(adjusted_by_team[team] - rating, 1),
                "adjusted": round(adjusted_by_team[team], 1),
            }
            for team, rating in team_ratings[:10]
        ],
    }

    print(
        f"Loaded {provenance['n_matches']} matches from {provenance['snapshot_path']}"
    )
    print(f"Latest match date on or before the cutoff: {provenance['max_date']}")
    print(f"Computed Elo ratings for {len(team_ratings)} teams.")
    print(f"Loaded squad values for {len(squad_values)} teams.")
    print("Squad-value adjustment, top 20 teams by Elo (Elo, nudge, adjusted):")
    print(f"    {'team':<24} {'elo':>8} {'nudge':>8} {'adjusted':>9}")
    for rank, (team, rating) in enumerate(team_ratings[:20], start=1):
        nudge = adjusted_by_team[team] - rating
        print(
            f"{rank:>2}. {team:<24} {rating:8.1f} {nudge:+8.1f} "
            f"{adjusted_by_team[team]:9.1f}"
        )

    print("Fitting Dixon-Coles goal model on the guarded matches...")
    goal_model = model.fit_model(matches, config)
    print("Dixon-Coles outcome probabilities for sample fixtures:")
    for home_team, away_team in SAMPLE_FIXTURES:
        probs = model.match_probabilities(goal_model, home_team, away_team)
        print(
            f"  {home_team} vs {away_team}: "
            f"home {probs['p_home_win']:.3f}, "
            f"draw {probs['p_draw']:.3f}, "
            f"away {probs['p_away_win']:.3f}"
        )

    # Full tournament simulation is the next layer. For now the artifact keeps
    # the equal-probability placeholder predictions.
    predictions = build_placeholder_predictions(PLACEHOLDER_TEAMS)
    paths = artifact.write_artifact(
        predictions,
        config,
        provenance=provenance,
        ratings_summary=ratings_summary,
    )
    print(f"Wrote predictions: {paths['predictions']}")
    print(f"Wrote run log: {paths['log']}")


if __name__ == "__main__":
    main()
