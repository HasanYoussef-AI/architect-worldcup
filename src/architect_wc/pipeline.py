"""Entry point.

The single command-line entry point for a run. Reads config, runs the layered
model end to end, and writes a versioned artifact with provenance. The pipeline
loads and leakage-guards the match data (Layer 1), computes Elo ratings (Layer
2), nudges them by current squad value (Layer 4), fits the Dixon-Coles goal
model (Layer 3), and runs the Monte Carlo tournament simulation (Layer 5) to
produce each team's stage and win probabilities, the real output contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from architect_wc import artifact, ingest, model, ratings, simulate, squad

CONFIG_PATH = Path("config.yaml")

# Retained only as a hermetic fixture for the schema gate: when no real artifact
# exists yet, the schema test builds an equal-probability placeholder so it can
# validate the output contract without the network. The live pipeline below uses
# real simulated predictions.
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


def _build_prob_fn(goal_model: Any) -> simulate.ProbFn:
    """Wrap the goal model in a cached prob_fn for the simulator.

    Outcome probabilities for a given ordered fixture and venue are
    deterministic, so they are computed once per unique matchup and reused. That
    keeps thousands of simulated tournaments from re-querying the same fixtures.
    """
    cache: dict[tuple[str, str, bool], tuple[float, float, float]] = {}

    def prob_fn(home: str, away: str, neutral: bool) -> tuple[float, float, float]:
        key = (home, away, neutral)
        cached = cache.get(key)
        if cached is None:
            probs = model.match_probabilities(goal_model, home, away, neutral=neutral)
            cached = (probs["p_home_win"], probs["p_draw"], probs["p_away_win"])
            cache[key] = cached
        return cached

    return prob_fn


def main() -> None:
    """Run the full pipeline and write the artifact.

    Match data flows in through Layer 1, frozen as an immutable snapshot and
    filtered by the leakage guard. Layer 2 computes Elo, Layer 4 nudges those
    ratings by current squad value, and Layer 3 fits the Dixon-Coles goal model.
    Layer 5 then simulates the tournament many times on the squad-adjusted
    strengths and the goal model, with neutral venues except for host nations,
    and aggregates each team's stage and win probabilities. Those real
    predictions are the artifact, and the run log records provenance and a
    ratings summary so every run is traceable.
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

    structure = simulate.load_structure(config["tournament"]["structure"])
    n_sims = int(config.get("n_sims", 10000))
    seed = int(config.get("random_seed", 42))
    print(
        f"Simulating {n_sims} tournaments over {len(structure)} groups, "
        f"neutral venues except hosts, seed {seed}..."
    )
    prob_fn = _build_prob_fn(goal_model)
    predictions = simulate.run_simulations(
        structure, prob_fn, adjusted_by_team, n_sims=n_sims, seed=seed
    )

    ranked = sorted(predictions, key=lambda record: record["p_win"], reverse=True)
    print("Tournament forecast, top 10 by win probability:")
    print(f"    {'team':<24} {'R16':>7} {'QF':>7} {'SF':>7} {'final':>7} {'win':>7}")
    for rank, record in enumerate(ranked[:10], start=1):
        print(
            f"{rank:>2}. {record['team']:<24} "
            f"{record['p_round_of_16']:7.3f} "
            f"{record['p_quarter_final']:7.3f} "
            f"{record['p_semi_final']:7.3f} "
            f"{record['p_final']:7.3f} "
            f"{record['p_win']:7.3f}"
        )

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
