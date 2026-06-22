"""Layer 6, calibration.

Owns the honesty layer: it scores the forecast against real results with a real
metric, so the project is a verification system and not just another predictor.
The metric is the Ranked Probability Score (RPS), the right score for ordered
three-outcome football predictions, home win, draw, away win, where lower is
better. The scoring uses penaltyblog's ranked probability scoring rather than a
hand-rolled version.

The pure scoring logic is kept separate from the backtest orchestration. rps,
mean_rps, outcome_index, and base_rates are pure and need no model or network,
which is what lets the verification gate prove the scorer itself is correct
before any number it produces is trusted. run_backtest does the model work.

Leakage discipline is non-negotiable here: when a match is scored, the model must
never have seen that match or anything after it. run_backtest enforces this with
a single training cutoff and treats any overlap as a failure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from penaltyblog.metrics import rps_array, rps_average

# Ordered outcome indices for home win, draw, away win scoring.
HOME_WIN = 0
DRAW = 1
AWAY_WIN = 2

DEFAULT_BACKTEST_SIZE = 150


def outcome_index(home_score: Any, away_score: Any) -> int:
    """Return the ordered outcome index: 0 home win, 1 draw, 2 away win."""
    if home_score > away_score:
        return HOME_WIN
    if home_score == away_score:
        return DRAW
    return AWAY_WIN


def rps(probabilities: tuple[float, float, float], outcome: int) -> float:
    """Ranked Probability Score for one ordered three-outcome prediction.

    probabilities is (p_home_win, p_draw, p_away_win) and outcome is the index 0,
    1, or 2 of the result that actually happened. Lower is better: 0 is a perfect
    confident prediction and 1 is the worst possible. Wraps penaltyblog's
    rps_array. Pure function with no I/O.
    """
    return float(rps_array([list(probabilities)], [outcome])[0])


def mean_rps(
    probabilities: list[list[float]] | np.ndarray, outcomes: list[int] | np.ndarray
) -> float:
    """Mean RPS over many predictions via penaltyblog's rps_average. Pure."""
    return float(
        rps_average(
            np.asarray(probabilities, dtype=np.float64),
            np.asarray(outcomes, dtype=np.int32),
        )
    )


def base_rates(matches: pd.DataFrame) -> tuple[float, float, float]:
    """Home win, draw, away win frequencies over matches with known results.

    Pure function with no I/O. This is the naive baseline forecast: the long-run
    outcome rates, applied as a constant prediction regardless of who is playing.
    """
    played = matches.dropna(subset=["home_score", "away_score"])
    total = len(played)
    if total == 0:
        raise ValueError("Cannot compute base rates from zero played matches.")
    home = int((played["home_score"] > played["away_score"]).sum())
    draw = int((played["home_score"] == played["away_score"]).sum())
    away = total - home - draw
    return (home / total, draw / total, away / total)


def _is_neutral(value: Any) -> bool:
    """Interpret the neutral column robustly across bool and text encodings."""
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1")
    return bool(value)


def run_backtest(matches: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Backtest the model out of sample and compare it to a naive baseline.

    Takes the most recent backtest_size matches with known results as the holdout
    window, fits the goal model once on data strictly before the window starts,
    predicts each holdout match honouring its actual venue, and scores it with
    RPS. The model is also compared against the historical base-rate baseline on
    the exact same matches. Returns the model mean RPS, the baseline mean RPS, the
    number of matches scored, and the cutoff dates that prove no leakage.

    A single training cutoff keeps the backtest to one model fit, which matters
    because fitting the full model for every historical match is expensive. The
    holdout window is recent and short, so the single fit is only slightly stale
    for its most recent matches, a conservative bias rather than a flattering one.
    backtest_size trades cost against confidence and is configurable; widen later.
    """
    from architect_wc import model

    calibrate_config = config.get("calibrate", {}) or {}
    backtest_size = int(calibrate_config.get("backtest_size", DEFAULT_BACKTEST_SIZE))
    as_of_date = pd.Timestamp(config["as_of_date"])

    played = matches.dropna(subset=["home_score", "away_score"]).copy()
    played["date"] = pd.to_datetime(played["date"])
    played = played[played["date"] <= as_of_date].sort_values("date", kind="stable")
    if played.empty:
        raise ValueError("No played matches available for the backtest.")

    holdout = played.tail(backtest_size)
    cutoff = holdout["date"].min()
    training = played[played["date"] < cutoff]
    if training.empty:
        raise ValueError("No training data strictly before the backtest window.")

    # Leakage guard: the model must never have seen a scored match or anything
    # after it. Treat any overlap as a failure, not a warning.
    train_max = training["date"].max()
    holdout_min = holdout["date"].min()
    if train_max >= holdout_min:
        raise ValueError(
            f"Leakage in backtest: training max date {train_max.date()} is not "
            f"strictly before the holdout start {holdout_min.date()}."
        )

    # The model can only score teams it has seen, so drop holdout matches with an
    # unseen team rather than crash or guess.
    train_teams = set(training["home_team"]) | set(training["away_team"])
    scorable = holdout[
        holdout["home_team"].isin(train_teams) & holdout["away_team"].isin(train_teams)
    ]
    if scorable.empty:
        raise ValueError("No holdout matches have both teams present in training.")

    # Fit once, anchored as of the cutoff so the time decay reference is correct.
    fit_config = dict(config)
    fit_config["as_of_date"] = cutoff.date().isoformat()
    fitted = model.fit_model(training, fit_config)

    model_probs: list[list[float]] = []
    outcomes: list[int] = []
    for row in scorable.itertuples(index=False):
        probs = model.match_probabilities(
            fitted, row.home_team, row.away_team, neutral=_is_neutral(row.neutral)
        )
        model_probs.append([probs["p_home_win"], probs["p_draw"], probs["p_away_win"]])
        outcomes.append(outcome_index(row.home_score, row.away_score))

    # Naive baseline: training base rates as a constant prediction on every scored
    # match. Computed from training only, so both sides are leakage-safe.
    rates = base_rates(training)
    baseline_probs = [list(rates)] * len(outcomes)

    model_mean = mean_rps(model_probs, outcomes)
    baseline_mean = mean_rps(baseline_probs, outcomes)

    return {
        "n_matches": len(outcomes),
        "backtest_size": backtest_size,
        "cutoff_date": cutoff.date().isoformat(),
        "train_max_date": train_max.date().isoformat(),
        "holdout_start": holdout_min.date().isoformat(),
        "holdout_end": holdout["date"].max().date().isoformat(),
        "model_mean_rps": model_mean,
        "baseline_mean_rps": baseline_mean,
        "baseline_base_rates": {
            "home": rates[0],
            "draw": rates[1],
            "away": rates[2],
        },
        "model_beats_baseline": model_mean < baseline_mean,
    }
