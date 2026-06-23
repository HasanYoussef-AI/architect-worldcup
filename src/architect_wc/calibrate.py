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

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from penaltyblog.metrics import rps_array, rps_average

# Ordered outcome indices for home win, draw, away win scoring.
HOME_WIN = 0
DRAW = 1
AWAY_WIN = 2

DEFAULT_BACKTEST_SIZE = 150
DEFAULT_WALK_WINDOWS = 8


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


class ScorableMatch(NamedTuple):
    """One holdout match to be scored: its date, teams, venue, and result.

    The date and teams together identify the match, so a walk-forward window can
    record exactly which matches it scored and a later phase can reuse the
    identical split without regenerating it by guesswork.
    """

    date: str
    home_team: str
    away_team: str
    neutral: bool
    outcome: int


class BacktestWindow(NamedTuple):
    """The shared, leakage-enforced holdout window every configuration scores.

    Centralising the split here is what guarantees calibration and ablation use
    the identical matches, training cutoff, and venue handling, so any difference
    between configurations comes only from the toggled layer, never the data.
    """

    training: pd.DataFrame
    matches: list[ScorableMatch]
    backtest_size: int
    cutoff_date: str
    train_max_date: str
    holdout_start: str
    holdout_end: str


def assert_no_leakage(train_max: pd.Timestamp, holdout_min: pd.Timestamp) -> None:
    """Raise if training is not strictly before the holdout. The leakage guard.

    The model must never have seen a scored match or anything after it. Any
    overlap is a failure, not a warning, so this raises rather than logs.
    """
    if train_max >= holdout_min:
        raise ValueError(
            f"Leakage in backtest: training max date {train_max.date()} is not "
            f"strictly before the holdout start {holdout_min.date()}."
        )


def prepare_backtest(matches: pd.DataFrame, config: dict[str, Any]) -> BacktestWindow:
    """Build the shared holdout window under strict leakage discipline.

    Takes the most recent backtest_size matches with known results as the holdout,
    sets the training cutoff at the holdout start, asserts training is strictly
    before that start, and keeps only holdout matches whose teams both appear in
    training. Returns the training frame and the scorable matches with their
    venue and result, the single source of truth all configurations score on.

    A single training cutoff keeps each configuration to one model fit, which
    matters because fitting the full model for every historical match is
    expensive. The holdout window is recent and short, so a single fit is only
    slightly stale for its most recent matches, a conservative bias rather than a
    flattering one. backtest_size trades cost against confidence and is tunable.
    """
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

    train_max = training["date"].max()
    holdout_min = holdout["date"].min()
    assert_no_leakage(train_max, holdout_min)

    # The model can only score teams it has seen, so drop holdout matches with an
    # unseen team rather than crash or guess.
    train_teams = set(training["home_team"]) | set(training["away_team"])
    scorable = holdout[
        holdout["home_team"].isin(train_teams) & holdout["away_team"].isin(train_teams)
    ]
    if scorable.empty:
        raise ValueError("No holdout matches have both teams present in training.")

    scorable_matches = [
        ScorableMatch(
            date=pd.Timestamp(row.date).date().isoformat(),
            home_team=row.home_team,
            away_team=row.away_team,
            neutral=_is_neutral(row.neutral),
            outcome=outcome_index(row.home_score, row.away_score),
        )
        for row in scorable.itertuples(index=False)
    ]

    return BacktestWindow(
        training=training,
        matches=scorable_matches,
        backtest_size=backtest_size,
        cutoff_date=cutoff.date().isoformat(),
        train_max_date=train_max.date().isoformat(),
        holdout_start=holdout_min.date().isoformat(),
        holdout_end=holdout["date"].max().date().isoformat(),
    )


def score_window(
    window: BacktestWindow, config: dict[str, Any]
) -> tuple[float, float, tuple[float, float, float]]:
    """Fit Dixon-Coles on a window's training and score its holdout with RPS.

    Returns the model mean RPS, the baseline mean RPS from the window's own
    training base rates, and those base rates. Shared by the single-window
    backtest and the walk-forward harness so both score a window the identical
    way, which is what lets the walk-forward anchor reproduce the calibration RPS.
    The fit is anchored as of the window cutoff so the time decay reference is
    correct, and the baseline is computed from training only, so it is leakage
    safe.
    """
    from architect_wc import model

    fit_config = dict(config)
    fit_config["as_of_date"] = window.cutoff_date
    fitted = model.fit_model(window.training, fit_config)

    model_probs: list[list[float]] = []
    outcomes: list[int] = []
    for match in window.matches:
        probs = model.match_probabilities(
            fitted, match.home_team, match.away_team, neutral=match.neutral
        )
        model_probs.append([probs["p_home_win"], probs["p_draw"], probs["p_away_win"]])
        outcomes.append(match.outcome)

    rates = base_rates(window.training)
    baseline_probs = [list(rates)] * len(outcomes)
    return mean_rps(model_probs, outcomes), mean_rps(baseline_probs, outcomes), rates


def run_backtest(matches: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Backtest the goal model out of sample and compare it to a naive baseline.

    Prepares the shared leakage-enforced window, fits the Dixon-Coles goal model
    once on training, scores each holdout match with RPS honouring its venue, and
    compares it to the historical base-rate baseline on the exact same matches.
    Returns the model mean RPS, the baseline mean RPS, the number of matches
    scored, and the cutoff dates that prove no leakage.
    """
    window = prepare_backtest(matches, config)
    model_mean, baseline_mean, rates = score_window(window, config)

    return {
        "n_matches": len(window.matches),
        "backtest_size": window.backtest_size,
        "cutoff_date": window.cutoff_date,
        "train_max_date": window.train_max_date,
        "holdout_start": window.holdout_start,
        "holdout_end": window.holdout_end,
        "model_mean_rps": model_mean,
        "baseline_mean_rps": baseline_mean,
        "baseline_base_rates": {
            "home": rates[0],
            "draw": rates[1],
            "away": rates[2],
        },
        "model_beats_baseline": model_mean < baseline_mean,
    }


def run_walk_forward(matches: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Walk-forward backtest: non-overlapping holdouts stepping backward.

    Anchors the most recent window on the calibration window exactly, then steps
    backward in consecutive holdouts of the same size, each trained on everything
    strictly before its own holdout start. The step is simple: the next window
    ends the day this window's training ends, so the holdouts are contiguous and
    do not share matches. Every window is built by the same prepare_backtest and
    guarded by the same assert_no_leakage, so the windowing and the leakage rule
    are the trusted single-window machinery, looped, not a parallel copy.

    Returns the per-window results, the explicit window definitions including the
    holdout match identifiers so a later phase can reuse the identical splits, and
    the aggregate statistics whose standard deviation across windows is the noise
    floor for judging later changes. Non-overlapping windows are deliberate:
    overlapping holdouts share matches and would fake a tighter spread than the
    data supports.
    """
    calibrate_config = config.get("calibrate", {}) or {}
    n_windows = int(calibrate_config.get("walk_windows", DEFAULT_WALK_WINDOWS))
    window_size = int(calibrate_config.get("backtest_size", DEFAULT_BACKTEST_SIZE))

    windows: list[dict[str, Any]] = []
    current_as_of = str(config["as_of_date"])
    for _ in range(n_windows):
        window_config = dict(config)
        window_config["as_of_date"] = current_as_of
        try:
            window = prepare_backtest(matches, window_config)
        except ValueError:
            # Ran out of data to form another full window. Stop honestly here and
            # report the real number of windows reached rather than padding.
            break

        # The leakage guard already fired inside prepare_backtest. Assert again so
        # the guarantee is explicit for every walk-forward window, not just the
        # anchor.
        assert_no_leakage(
            pd.Timestamp(window.train_max_date), pd.Timestamp(window.holdout_start)
        )

        model_mean, baseline_mean, _rates = score_window(window, config)
        windows.append(
            {
                "index": len(windows) + 1,
                "cutoff_date": window.cutoff_date,
                "train_max_date": window.train_max_date,
                "holdout_start": window.holdout_start,
                "holdout_end": window.holdout_end,
                "n_matches": len(window.matches),
                "train_size": len(window.training),
                "model_mean_rps": model_mean,
                "baseline_mean_rps": baseline_mean,
                "holdout_matches": [
                    {
                        "date": match.date,
                        "home_team": match.home_team,
                        "away_team": match.away_team,
                    }
                    for match in window.matches
                ],
            }
        )
        # Step backward to the next, older, non-overlapping window.
        current_as_of = window.train_max_date

    if not windows:
        raise ValueError("Could not form any walk-forward window from the data.")

    model_rps = np.array([w["model_mean_rps"] for w in windows], dtype=np.float64)
    baseline_rps = np.array([w["baseline_mean_rps"] for w in windows], dtype=np.float64)
    n = len(windows)
    # Sample standard deviation across windows, the window-to-window spread. Zero
    # when only one window could be formed.
    std = float(np.std(model_rps, ddof=1)) if n > 1 else 0.0

    aggregate = {
        "n_windows": n,
        "mean_rps": float(np.mean(model_rps)),
        "std_rps": std,
        "min_rps": float(np.min(model_rps)),
        "max_rps": float(np.max(model_rps)),
        "mean_baseline_rps": float(np.mean(baseline_rps)),
        "earliest_holdout_start": windows[-1]["holdout_start"],
        "earliest_train_size": windows[-1]["train_size"],
    }

    return {
        "as_of_date": str(config.get("as_of_date")),
        "random_seed": config.get("random_seed"),
        "window_size": window_size,
        "requested_windows": n_windows,
        "windows": windows,
        "aggregate": aggregate,
        "noise_floor_rule": (
            "The standard deviation across windows is the noise floor. No future "
            "change counts as real unless it moves the aggregate mean RPS by more "
            f"than this spread ({std:.4f})."
        ),
    }


def format_walk_forward(report: dict[str, Any]) -> str:
    """Render the walk-forward report as a clean stdout table with aggregates."""
    aggregate = report["aggregate"]
    lines = [
        "World Cup walk-forward backtest",
        (
            f"Window size {report['window_size']} matches, seed "
            f"{report['random_seed']}, {aggregate['n_windows']} non-overlapping "
            "windows stepping backward."
        ),
        "",
        f"  {'win':>3} {'holdout start':>13} {'holdout end':>13} {'trained to':>11}"
        f" {'n':>4} {'model_rps':>10} {'base_rps':>9}",
    ]
    for w in report["windows"]:
        lines.append(
            f"  {w['index']:>3} {w['holdout_start']:>13} {w['holdout_end']:>13}"
            f" {w['train_max_date']:>11} {w['n_matches']:>4}"
            f" {w['model_mean_rps']:10.4f} {w['baseline_mean_rps']:9.4f}"
        )
    lines.extend(
        [
            "",
            f"Aggregate over {aggregate['n_windows']} windows:",
            f"  mean model RPS   {aggregate['mean_rps']:.4f}",
            f"  std (noise floor){aggregate['std_rps']:>8.4f}",
            f"  min model RPS    {aggregate['min_rps']:.4f}",
            f"  max model RPS    {aggregate['max_rps']:.4f}",
            f"  mean baseline    {aggregate['mean_baseline_rps']:.4f}",
            "",
            report["noise_floor_rule"],
        ]
    )
    return "\n".join(lines)
