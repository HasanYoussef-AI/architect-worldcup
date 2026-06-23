"""Hybrid ensemble, ability estimates plus covariates into a tree ensemble.

A separate per-match win, draw, loss probability model in the Groll and Zeileis
spirit, competing against Dixon-Coles on the same metric and the same walk-forward
splits. It does not touch the Dixon-Coles scoreline model, which stays the
simulator's engine. The model is scikit-learn's HistGradientBoostingClassifier,
the standard library's gradient boosting, chosen for deterministic seeded
reproducibility and native handling of missing features, with no extra native
dependency to manage.

Four leakage-safe features per match, all as home-minus-away differentials so the
set stays at exactly four columns:
  ability_diff  point-in-time Dixon-Coles net strength (attack minus defence)
  squad_diff    squad market value from the dated snapshot
  elo_diff      point-in-time Elo rating
  rest_diff     days since each team's previous match

The features are point-in-time: for any match at date d, every value uses only
data strictly before d. The ability feature is the highest risk, so its
enforcement is spelled out in point_in_time_abilities. The features are computed
once over the full data, since a match's point-in-time values do not depend on
which window scores it, and reused across all windows.

Round-two feature candidates, deliberately not built yet, the pipeline can accept
them later if v1 importances and RPS justify: a big-match record feature from the
tournament column, and a recent-form feature over the last N matches.

Deliberately excluded by design: bookmaker odds, to keep the model independent so
it can be benchmarked against the market rather than copying it, and socio-economic
features like GDP and population, as near-noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from architect_wc import calibrate, model, ratings, squad

FEATURE_COLUMNS = ["ability_diff", "squad_diff", "elo_diff", "rest_diff"]

DEFAULT_GRID_CADENCE_DAYS = 120
DEFAULT_GRID_LOOKBACK_DAYS = 2920  # about eight years, beyond which xi decay is tiny
DEFAULT_GRID_YEARS = 12
DEFAULT_MIN_FIT_MATCHES = 200
DEFAULT_REST_DAYS = 30.0
REST_CAP_DAYS = 365.0
DEFAULT_MAX_ITER = 300
DEFAULT_LEARNING_RATE = 0.05


def _canonical_fit_config(config: dict[str, Any], as_of: str) -> dict[str, Any]:
    """Config for a grid Dixon-Coles fit: canonical model, anchored at as_of."""
    fit_config = dict(config)
    fit_config["as_of_date"] = as_of
    factors = dict(config.get("factors", {}) or {})
    factors["friendly_downweight"] = False
    fit_config["factors"] = factors
    return fit_config


def _net_abilities(fitted: Any) -> dict[str, float]:
    """Per-team net strength, attack minus defence, from a fitted Dixon-Coles."""
    params = fitted.get_params()
    abilities: dict[str, float] = {}
    for team in fitted.teams:
        attack = params.get(f"attack_{team}")
        defence = params.get(f"defence_{team}")
        if defence is None:
            defence = params.get(f"defense_{team}")
        if attack is not None and defence is not None:
            abilities[str(team)] = float(attack) - float(defence)
    return abilities


def point_in_time_abilities(played: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    """Point-in-time Dixon-Coles net-strength differential for each match.

    Leakage enforcement, the critical part: a grid of refit dates is laid down at
    a fixed cadence. At each grid date g, Dixon-Coles is fit on matches with date
    strictly less than g, within a rolling lookback. A match at date d is then
    assigned the abilities from the latest grid date g with g <= d. Since that
    fit used only data with date < g <= d, the ability for the match uses only
    data strictly before d. The match's own row, and every row dated d or later,
    is excluded by construction. Matches before the first grid date get NaN, an
    explicit unknown, rather than a fabricated value.

    Returns an array of home-minus-away net strength aligned to played row order.
    """
    ensemble_config = config.get("ensemble", {}) or {}
    cadence = int(ensemble_config.get("grid_cadence_days", DEFAULT_GRID_CADENCE_DAYS))
    lookback = int(
        ensemble_config.get("grid_lookback_days", DEFAULT_GRID_LOOKBACK_DAYS)
    )
    grid_years = int(ensemble_config.get("grid_years", DEFAULT_GRID_YEARS))
    min_fit = int(ensemble_config.get("min_fit_matches", DEFAULT_MIN_FIT_MATCHES))

    dates = pd.to_datetime(played["date"])
    max_date = dates.max()
    grid_start = max_date - pd.Timedelta(days=365 * grid_years)
    grid = pd.date_range(grid_start, max_date, freq=f"{cadence}D")

    grid_dates: list[np.datetime64] = []
    grid_abilities: list[dict[str, float]] = []
    for g in grid:
        low = g - pd.Timedelta(days=lookback)
        window = played[(dates >= low) & (dates < g)]
        if len(window) < min_fit:
            continue
        try:
            fit_config = _canonical_fit_config(config, g.isoformat())
            fitted = model.fit_model(window, fit_config)
        except Exception:
            # A degenerate slice must not crash the run; skip this grid date and
            # the matches that would have used it fall back to an earlier one.
            continue
        grid_dates.append(np.datetime64(g))
        grid_abilities.append(_net_abilities(fitted))

    ability_diff = np.full(len(played), np.nan, dtype=np.float64)
    if not grid_dates:
        return ability_diff

    grid_array = np.array(grid_dates)
    home = played["home_team"].to_numpy()
    away = played["away_team"].to_numpy()
    match_dates = dates.to_numpy()
    for i in range(len(played)):
        # Latest grid date g with g <= this match's date. side="right" counts
        # grid dates <= d, so position-1 is that latest g.
        position = int(np.searchsorted(grid_array, match_dates[i], side="right")) - 1
        if position < 0:
            continue
        abilities = grid_abilities[position]
        home_ability = abilities.get(home[i])
        away_ability = abilities.get(away[i])
        if home_ability is not None and away_ability is not None:
            ability_diff[i] = home_ability - away_ability
    return ability_diff


def chronological_features(
    played: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.Series, pd.Series]:
    """Point-in-time Elo and rest-days differentials in one chronological pass.

    For each match both values are read before the match is processed, so they use
    only prior matches. The Elo update reuses the Layer 2 logic. Rest is the days
    since each team's previous match, capped, with a documented default for a
    team's first appearance. Returns Elo and rest differentials indexed like
    played.
    """
    elo_config = config.get("elo", {}) or {}
    base_rating = float(elo_config.get("base_rating", ratings.BASE_RATING))
    k_factor = float(elo_config.get("k_factor", ratings.K_FACTOR))
    home_advantage = float(elo_config.get("home_advantage", ratings.HOME_ADVANTAGE))

    ordered = played.sort_values("date", kind="stable")
    elo: dict[str, float] = {}
    last_played: dict[str, pd.Timestamp] = {}
    elo_diff = pd.Series(np.nan, index=ordered.index, dtype=np.float64)
    rest_diff = pd.Series(np.nan, index=ordered.index, dtype=np.float64)

    for row in ordered.itertuples():
        home, away = row.home_team, row.away_team
        rating_home = elo.get(home, base_rating)
        rating_away = elo.get(away, base_rating)
        elo_diff.at[row.Index] = rating_home - rating_away

        match_date = pd.Timestamp(row.date)
        rest_home = (
            (match_date - last_played[home]).days
            if home in last_played
            else DEFAULT_REST_DAYS
        )
        rest_away = (
            (match_date - last_played[away]).days
            if away in last_played
            else DEFAULT_REST_DAYS
        )
        rest_home = min(float(rest_home), REST_CAP_DAYS)
        rest_away = min(float(rest_away), REST_CAP_DAYS)
        rest_diff.at[row.Index] = rest_home - rest_away

        # Update after recording, so the values above stay strictly point-in-time.
        advantage = 0.0 if ratings._is_neutral(row.neutral) else home_advantage
        expected_home = ratings.expected_score(rating_home, rating_away, advantage)
        if row.home_score > row.away_score:
            actual_home = 1.0
        elif row.home_score == row.away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0
        delta = k_factor * (actual_home - expected_home)
        elo[home] = rating_home + delta
        elo[away] = rating_away - delta
        last_played[home] = match_date
        last_played[away] = match_date

    return elo_diff.reindex(played.index), rest_diff.reindex(played.index)


def build_features(matches: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build the four point-in-time features for every played match.

    Returns a frame indexed like the played matches with the four feature columns,
    the ordered outcome, and the date. Squad value is the single dated snapshot
    applied as a static team attribute, so it is missing for non-snapshot teams
    and a mild anachronism for historical rows, but it carries no match outcome,
    so it is not result leakage.
    """
    played = matches.dropna(subset=["home_score", "away_score"]).copy()
    played["date"] = pd.to_datetime(played["date"])

    squad_values = squad.load_squad_values(config["squad"]["snapshot"])
    elo_diff, rest_diff = chronological_features(played, config)

    features = pd.DataFrame(index=played.index)
    features["ability_diff"] = point_in_time_abilities(played, config)
    features["squad_diff"] = played["home_team"].map(squad_values) - played[
        "away_team"
    ].map(squad_values)
    features["elo_diff"] = elo_diff
    features["rest_diff"] = rest_diff
    features["outcome"] = [
        calibrate.outcome_index(home, away)
        for home, away in zip(played["home_score"], played["away_score"], strict=True)
    ]
    features["date"] = played["date"].to_numpy()
    return features


def _make_classifier(config: dict[str, Any]):
    """Build the gradient boosting classifier with a fixed seed for determinism."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    ensemble_config = config.get("ensemble", {}) or {}
    seed = int(config.get("random_seed", 42))
    return HistGradientBoostingClassifier(
        random_state=seed,
        early_stopping=False,
        max_iter=int(ensemble_config.get("max_iter", DEFAULT_MAX_ITER)),
        learning_rate=float(
            ensemble_config.get("learning_rate", DEFAULT_LEARNING_RATE)
        ),
    )


def _probability_vectors(classifier: Any, features: pd.DataFrame) -> list[list[float]]:
    """Map predict_proba onto fixed home, draw, away columns by class index."""
    proba = classifier.predict_proba(features)
    classes = [int(c) for c in classifier.classes_]
    vectors: list[list[float]] = []
    for row in proba:
        vector = [0.0, 0.0, 0.0]
        for class_index, probability in zip(classes, row, strict=True):
            vector[class_index] = float(probability)
        vectors.append(vector)
    return vectors


def fit_and_predict(
    features: pd.DataFrame, window: calibrate.BacktestWindow, config: dict[str, Any]
) -> tuple[list[list[float]], Any]:
    """Train the ensemble on a window's training rows, predict its holdout.

    Trains on the window's full training set, the same training data Dixon-Coles
    used, and predicts the identical holdout matches, so the comparison is on the
    same splits. Returns the holdout probability vectors and the fitted classifier
    for importance.
    """
    train = features.loc[window.training.index]
    x_train = train[FEATURE_COLUMNS]
    y_train = train["outcome"].to_numpy()

    classifier = _make_classifier(config)
    classifier.fit(x_train, y_train)

    x_holdout = features.loc[window.holdout_index, FEATURE_COLUMNS]
    return _probability_vectors(classifier, x_holdout), classifier


def _feature_importances(
    classifier: Any,
    features: pd.DataFrame,
    window: calibrate.BacktestWindow,
    config: dict[str, Any],
) -> dict[str, float]:
    """Permutation importance on the holdout, scored by log loss. Empty on error."""
    from sklearn.inspection import permutation_importance

    x_holdout = features.loc[window.holdout_index, FEATURE_COLUMNS]
    y_holdout = np.array([match.outcome for match in window.matches])
    try:
        result = permutation_importance(
            classifier,
            x_holdout,
            y_holdout,
            scoring="neg_log_loss",
            n_repeats=5,
            random_state=int(config.get("random_seed", 42)),
        )
    except Exception:
        return {}
    return {
        column: float(mean)
        for column, mean in zip(FEATURE_COLUMNS, result.importances_mean, strict=True)
    }


def run_ensemble(matches: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Train and score the ensemble on the exact walk-forward splits.

    Builds the features once, then for each walk-forward window trains the
    ensemble on that window's training set and scores its win, draw, loss RPS on
    that window's holdout, identical to how Dixon-Coles was scored. Reports the
    ensemble against Dixon-Coles and the base-rate baseline per window, the
    aggregate mean and standard deviation, and the permutation feature importances
    aggregated across windows.
    """
    features = build_features(matches, config)

    window_rows: list[dict[str, Any]] = []
    importance_runs: dict[str, list[float]] = {column: [] for column in FEATURE_COLUMNS}
    for window in calibrate.walk_forward_windows(matches, config):
        # The leakage guard fires for every window the ensemble scores too.
        calibrate.assert_no_leakage(
            pd.Timestamp(window.train_max_date), pd.Timestamp(window.holdout_start)
        )
        outcomes = [match.outcome for match in window.matches]

        ensemble_vectors, classifier = fit_and_predict(features, window, config)
        ensemble_rps = calibrate.mean_rps(ensemble_vectors, outcomes)

        # Dixon-Coles and the baseline on the same window, the comparison columns.
        dc_rps, baseline_rps, _rates = calibrate.score_window(window, config)

        importances = _feature_importances(classifier, features, window, config)
        for column, value in importances.items():
            importance_runs[column].append(value)

        window_rows.append(
            {
                "index": len(window_rows) + 1,
                "holdout_start": window.holdout_start,
                "holdout_end": window.holdout_end,
                "train_max_date": window.train_max_date,
                "n_matches": len(window.matches),
                "ensemble_rps": ensemble_rps,
                "dixon_coles_rps": dc_rps,
                "baseline_rps": baseline_rps,
            }
        )

    if not window_rows:
        raise ValueError("Could not form any walk-forward window for the ensemble.")

    ensemble_values = np.array(
        [r["ensemble_rps"] for r in window_rows], dtype=np.float64
    )
    dc_values = np.array([r["dixon_coles_rps"] for r in window_rows], dtype=np.float64)
    baseline_values = np.array(
        [r["baseline_rps"] for r in window_rows], dtype=np.float64
    )
    n = len(window_rows)
    ensemble_std = float(np.std(ensemble_values, ddof=1)) if n > 1 else 0.0
    dc_std = float(np.std(dc_values, ddof=1)) if n > 1 else 0.0

    importances = {
        column: (float(np.mean(runs)) if runs else float("nan"))
        for column, runs in importance_runs.items()
    }
    ensemble_mean = float(np.mean(ensemble_values))
    dc_mean = float(np.mean(dc_values))
    delta = ensemble_mean - dc_mean
    cleared_bar = delta < -dc_std

    return {
        "as_of_date": str(config.get("as_of_date")),
        "random_seed": config.get("random_seed"),
        "n_windows": n,
        "window_size": int(
            (config.get("calibrate", {}) or {}).get("backtest_size", 150)
        ),
        "model": "HistGradientBoostingClassifier",
        "features": FEATURE_COLUMNS,
        "windows": window_rows,
        "aggregate": {
            "ensemble_mean_rps": ensemble_mean,
            "ensemble_std_rps": ensemble_std,
            "ensemble_min_rps": float(np.min(ensemble_values)),
            "ensemble_max_rps": float(np.max(ensemble_values)),
            "dixon_coles_mean_rps": dc_mean,
            "dixon_coles_std_rps": dc_std,
            "baseline_mean_rps": float(np.mean(baseline_values)),
            "delta_ensemble_minus_dc": delta,
            "noise_floor": dc_std,
            "cleared_noise_floor": cleared_bar,
        },
        "feature_importances": importances,
        "verdict_rule": (
            "Lower RPS is better. The ensemble counts as a real improvement only if "
            "its aggregate mean is below the Dixon-Coles mean by more than the "
            f"noise floor of {dc_std:.4f}."
        ),
    }


def format_ensemble(report: dict[str, Any]) -> str:
    """Render the ensemble comparison table and importances for stdout."""
    aggregate = report["aggregate"]
    lines = [
        "World Cup hybrid ensemble, evaluated on the Phase 7.6 walk-forward splits",
        (
            f"Model {report['model']}, {report['n_windows']} windows of "
            f"{report['window_size']} matches, seed {report['random_seed']}."
        ),
        "",
        f"  {'win':>3} {'holdout start':>13} {'holdout end':>13} {'n':>4}"
        f" {'ensemble':>9} {'dixon_coles':>12} {'baseline':>9}",
    ]
    for row in report["windows"]:
        lines.append(
            f"  {row['index']:>3} {row['holdout_start']:>13} {row['holdout_end']:>13}"
            f" {row['n_matches']:>4} {row['ensemble_rps']:9.4f}"
            f" {row['dixon_coles_rps']:12.4f} {row['baseline_rps']:9.4f}"
        )
    lines.extend(
        [
            "",
            "Aggregate:",
            f"  ensemble    mean {aggregate['ensemble_mean_rps']:.4f}  "
            f"std {aggregate['ensemble_std_rps']:.4f}  "
            f"min {aggregate['ensemble_min_rps']:.4f}  "
            f"max {aggregate['ensemble_max_rps']:.4f}",
            f"  dixon-coles mean {aggregate['dixon_coles_mean_rps']:.4f}  "
            f"std {aggregate['dixon_coles_std_rps']:.4f}",
            f"  baseline    mean {aggregate['baseline_mean_rps']:.4f}",
            "",
            f"  ensemble minus dixon-coles: {aggregate['delta_ensemble_minus_dc']:+.4f}"
            f"   (noise floor {aggregate['noise_floor']:.4f})",
            report["verdict_rule"],
            (
                "  Verdict: ensemble cleared the noise floor."
                if aggregate["cleared_noise_floor"]
                else "  Verdict: ensemble did NOT clear the noise floor."
            ),
            "",
            "Permutation feature importances (mean log-loss increase across windows):",
        ]
    )
    ordered = sorted(
        report["feature_importances"].items(),
        key=lambda item: (np.isnan(item[1]), -item[1]),
    )
    for column, value in ordered:
        lines.append(f"    {column:<14} {value:+.4f}")
    return "\n".join(lines)
