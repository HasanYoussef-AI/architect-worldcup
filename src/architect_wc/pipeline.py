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

# Cutoff for the live dated forecast, end of 2026-06-22, kept separate from the
# frozen pre-tournament cutoff in config so the calibration numbers do not move.
LIVE_AS_OF_DATE = "2026-06-22"

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


def run_forecast(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the forecast at the config cutoff and write a dated frozen artifact.

    Loads and leakage-guards the data at config.as_of_date, computes the
    squad-adjusted ratings, fits the goal model, anchors on the real World Cup
    matches already played at the cutoff, and simulates the remaining matches and
    the knockout bracket. Writes a dated artifact carrying the cutoff, the leakage
    proof, the snapshot, and the current standings. Returns the predictions so a
    caller can compare two dated forecasts. With no played matches at the cutoff
    this is the pre-tournament forecast.
    """
    import pandas as pd

    matches, provenance = ingest.load_matches(config)

    as_of = pd.Timestamp(config["as_of_date"])
    boundary = as_of + pd.Timedelta(days=1)
    training_max = pd.to_datetime(matches["date"]).max()
    # Same hard leakage assertion as always: nothing at or after the cutoff
    # boundary entered training, so the forecast covers only the future.
    if training_max >= boundary:
        raise ValueError(
            f"Leakage at forecast cutoff: training max date {training_max.date()} "
            f"is not before the cutoff boundary {boundary.date()}."
        )

    team_ratings = ratings.compute_elo(matches, config)
    squad_values = squad.load_squad_values(config["squad"]["snapshot"])
    adjusted_by_team = dict(squad.adjust_ratings(team_ratings, squad_values, config))
    ratings_summary = {
        "n_teams": len(team_ratings),
        "n_squad_values": len(squad_values),
    }

    goal_model = model.fit_model(matches, config)
    score_fn = build_score_fn(goal_model)

    tournament_config = config.get("tournament", {}) or {}
    structure = simulate.load_structure(tournament_config["structure"])
    assignment_table = simulate.load_assignment_table(
        tournament_config["r32_assignment"]
    )
    known_results = simulate.build_known_results(matches, structure)
    standings = simulate.current_standings(structure, known_results)

    n_sims = int(config.get("n_sims", 10000))
    seed = int(config.get("random_seed", 42))
    print(
        f"Forecast as_of {config['as_of_date']}: trained on {provenance['n_matches']} "
        f"matches up to {provenance['max_date']}, {len(known_results)} real group "
        f"results anchored, simulating {n_sims} tournaments, seed {seed}..."
    )
    predictions, sim_meta = simulate.run_simulations(
        structure,
        score_fn,
        adjusted_by_team,
        assignment_table,
        n_sims=n_sims,
        seed=seed,
        known_results=known_results,
    )

    forecast_summary = {
        "as_of_date": config["as_of_date"],
        "cutoff_boundary": boundary.date().isoformat(),
        "training_max_date": training_max.date().isoformat(),
        "leakage_ok": bool(training_max < boundary),
        "snapshot_path": provenance["snapshot_path"],
        "n_real_group_results": len(known_results),
        "n_sims": n_sims,
        "seed": seed,
        "fair_play_fallback_fraction": sim_meta["fair_play_fallback_fraction"],
        "current_standings": {
            group: [
                {"team": t, "played": pl, "points": p, "gd": gd, "gf": gf}
                for t, pl, p, gd, gf in rows
            ]
            for group, rows in standings.items()
        },
    }

    ranked = sorted(predictions, key=lambda record: record["p_win"], reverse=True)
    print("Top 10 by title probability:")
    for rank, record in enumerate(ranked[:10], start=1):
        print(f"{rank:>2}. {record['team']:<24} win {record['p_win']:.3f}")
    print(
        f"Leakage proof: trained up to {training_max.date()}, "
        f"strictly before the cutoff boundary {boundary.date()}."
    )
    print(
        f"Fair-play fallback reached in "
        f"{sim_meta['fair_play_fallback_fraction'] * 100:.2f} percent of simulations."
    )

    paths = artifact.write_artifact(
        predictions,
        config,
        provenance=provenance,
        ratings_summary=ratings_summary,
        forecast_summary=forecast_summary,
    )
    print(f"Wrote predictions: {paths['predictions']}")
    print(f"Wrote run log: {paths['log']}")
    return predictions


def main() -> None:
    """Run the forecast at the config cutoff, the pre-tournament dated forecast."""
    run_forecast(load_config())


def live_forecast_main() -> None:
    """Run the live dated forecast at as_of 2026-06-22 on the real structure.

    A separate command from wc-predict. It overrides the cutoff to the end of
    2026-06-22 so every real match played through that date is training data and
    is anchored as a known result, and the forecast covers the remaining group
    matches and the knockout bracket. The pre-tournament forecast and its frozen
    numbers are untouched; this writes its own dated artifact.
    """
    config = load_config()
    config["as_of_date"] = LIVE_AS_OF_DATE
    run_forecast(config)


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


def llm_smoke_main() -> None:
    """Run the offline LLM plumbing smoke test and print a report.

    Proves the Phase 8 plumbing with no API call: the quarantine gate catches a
    planted target-round result, Predictions A, B, and C validate against their
    schemas, and the RPS scorer reproduces a hand-checked value. Exits nonzero if
    any check fails. The same checks run as offline pytest gates in the suite.
    """
    import sys

    from architect_wc.llm import smoke

    report = smoke.run_smoke()
    print("LLM phase offline smoke test")
    for check in report["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        print(f"  [{mark}] {check['name']}: {check['detail']}")
    if report["all_passed"]:
        print("All plumbing checks passed.")
    else:
        print("Plumbing checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
