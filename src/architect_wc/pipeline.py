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

from architect_wc import (
    ablation,
    artifact,
    audit,
    calibrate,
    ensemble,
    ingest,
    model,
    ratings,
    simulate,
    squad,
)

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


def build_score_fn(goal_model: Any) -> simulate.ScoreFn:
    """Wrap the goal model in a cached scoreline sampler for the simulator.

    The simulator needs goals, not just outcomes, for the group tiebreakers, so
    this samples a scoreline from the Dixon-Coles joint goal matrix. The matrix
    for a given fixture and venue is deterministic, so it is computed once per
    unique matchup and reused as a cumulative distribution that the random draw
    indexes into.
    """
    import numpy as np

    cache: dict[tuple[str, str, bool], tuple[int, np.ndarray]] = {}

    def score_fn(home: str, away: str, neutral: bool, rng: Any) -> tuple[int, int]:
        key = (home, away, neutral)
        cached = cache.get(key)
        if cached is None:
            grid = goal_model.predict(home, away, neutral_venue=neutral)
            matrix = np.asarray(grid.goal_matrix, dtype=np.float64)
            flat = matrix.ravel()
            flat = flat / flat.sum()
            cache[key] = (matrix.shape[1], np.cumsum(flat))
            cached = cache[key]
        n_columns, cumulative = cached
        index = int(np.searchsorted(cumulative, rng.random()))
        index = min(index, cumulative.size - 1)
        home_goals, away_goals = divmod(index, n_columns)
        return int(home_goals), int(away_goals)

    return score_fn


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

    tournament_config = config.get("tournament", {}) or {}
    structure = simulate.load_structure(tournament_config["structure"])
    assignment_table = simulate.load_assignment_table(
        tournament_config["r32_assignment"]
    )
    n_sims = int(config.get("n_sims", 10000))
    seed = int(config.get("random_seed", 42))
    print(
        f"Simulating {n_sims} tournaments over {len(structure)} groups on the real "
        f"2026 bracket, hosts at home in the group stage, seed {seed}..."
    )
    score_fn = build_score_fn(goal_model)
    predictions, sim_meta = simulate.run_simulations(
        structure,
        score_fn,
        adjusted_by_team,
        assignment_table,
        n_sims=n_sims,
        seed=seed,
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
    print(
        f"Fair-play fallback was reached in "
        f"{sim_meta['fair_play_fallback_fraction'] * 100:.2f} percent of simulations."
    )

    paths = artifact.write_artifact(
        predictions,
        config,
        provenance=provenance,
        ratings_summary=ratings_summary,
    )
    print(f"Wrote predictions: {paths['predictions']}")
    print(f"Wrote run log: {paths['log']}")


def calibrate_main() -> None:
    """Run the calibration backtest and write a calibration log.

    A separate reporting path from the forecast: it scores the goal model out of
    sample with the Ranked Probability Score against real results, compares it to
    a naive base-rate baseline on the same matches under the same leakage
    discipline, and prints and logs the model mean RPS, the baseline mean RPS, and
    the number of matches scored. A lower score is better. If the model does not
    beat the baseline that is a real finding, reported as is.
    """
    config = load_config()
    matches, provenance = ingest.load_matches(config)

    print(
        f"Loaded {provenance['n_matches']} matches from {provenance['snapshot_path']}"
    )
    print("Backtesting the goal model out of sample with RPS...")
    results = calibrate.run_backtest(matches, config)

    print(
        f"Scored {results['n_matches']} matches in the holdout window "
        f"{results['holdout_start']} to {results['holdout_end']}."
    )
    print(
        f"Leakage check: model trained on data up to {results['train_max_date']}, "
        f"strictly before the holdout start {results['holdout_start']}."
    )
    print(f"Model mean RPS:    {results['model_mean_rps']:.4f}  (lower is better)")
    print(f"Baseline mean RPS: {results['baseline_mean_rps']:.4f}  (base rates)")
    verdict = (
        "Model beats the naive baseline."
        if results["model_beats_baseline"]
        else "Model does NOT beat the naive baseline. Reported as is."
    )
    print(verdict)

    log_path = artifact.write_calibration_log(results, config, provenance=provenance)
    print(f"Wrote calibration log: {log_path}")


def ablation_main() -> None:
    """Run the ablation harness and write a versioned ablation artifact.

    A separate reporting path that mirrors wc-calibrate. It scores the
    Dixon-Coles goal model against a transparent Elo win, draw, loss generator,
    and ablates squad value inside that Elo generator, all on the one shared
    leakage-enforced backtest window so every delta is attributable to a single
    toggled layer. The forecast and calibration paths are untouched. The full
    Dixon-Coles configuration reproduces the Phase 6 calibration RPS, which is
    what makes every delta trustworthy.
    """
    config = load_config()
    matches, provenance = ingest.load_matches(config)

    print(
        f"Loaded {provenance['n_matches']} matches from {provenance['snapshot_path']}"
    )
    print("Running the ablation on the shared backtest window...")
    report = ablation.run_ablation(matches, config)

    print(ablation.format_report(report))

    log_path = artifact.write_ablation_log(report, config, provenance=provenance)
    print(f"Wrote ablation log: {log_path}")


def walk_backtest_main() -> None:
    """Run the walk-forward backtest and write a versioned artifact.

    A separate reporting path that mirrors wc-calibrate and wc-ablate. It replaces
    the single 150-match holdout with a rolling-origin evaluation: non-overlapping
    windows stepping backward, each scored by the same leakage-enforced machinery,
    so the result is a mean RPS with a real spread rather than one fragile point.
    The most recent window reproduces the frozen calibration RPS. wc-calibrate and
    its single-window 0.1611 are untouched; this is an additional command.
    """
    config = load_config()
    matches, provenance = ingest.load_matches(config)

    print(
        f"Loaded {provenance['n_matches']} matches from {provenance['snapshot_path']}"
    )
    print("Running the walk-forward backtest (one Dixon-Coles fit per window)...")
    report = calibrate.run_walk_forward(matches, config)

    print(calibrate.format_walk_forward(report))

    log_path = artifact.write_walk_forward_log(report, config, provenance=provenance)
    print(f"Wrote walk-forward log: {log_path}")


def audit_main() -> None:
    """Run the leakage and calibration audit and print its findings.

    A read-only diagnostic that mirrors the other commands. It checks the
    training data for duplicate fixtures and inspects the probabilities the model
    emits for the most recent walk-forward window, the calibration anchor, for
    overconfidence and validity. It refits only that one window, changes no model,
    and leaves wc-calibrate, the 0.1611, and the walk-forward 0.1575 untouched.
    """
    config = load_config()
    matches, provenance = ingest.load_matches(config)

    print(
        f"Loaded {provenance['n_matches']} matches from {provenance['snapshot_path']}"
    )
    print("Running the leakage and calibration audit (one window fit)...")
    report = audit.run_audit(matches, config)
    print(audit.format_audit(report))


def ensemble_main() -> None:
    """Run the hybrid ensemble on the walk-forward splits and write an artifact.

    Mirrors the other commands. It builds the point-in-time features once, then
    trains and scores the gradient boosting ensemble on the exact Phase 7.6
    walk-forward windows, reporting it against Dixon-Coles and the baseline with
    the feature importances. It does not alter the Dixon-Coles results, which are
    recomputed here only as the comparison columns.
    """
    config = load_config()
    matches, provenance = ingest.load_matches(config)

    print(
        f"Loaded {provenance['n_matches']} matches from {provenance['snapshot_path']}"
    )
    print("Building point-in-time features and scoring the ensemble per window...")
    report = ensemble.run_ensemble(matches, config)

    print(ensemble.format_ensemble(report))

    log_path = artifact.write_ensemble_log(report, config, provenance=provenance)
    print(f"Wrote ensemble log: {log_path}")


if __name__ == "__main__":
    main()
