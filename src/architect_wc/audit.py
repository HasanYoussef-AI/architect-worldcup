"""Leakage and calibration audit.

Read-only integrity checks over the loaded data and the model's emitted
probabilities. This is an audit, not a modeling change: it refits only the most
recent walk-forward window, the one the checks need, and it does not touch the
calibration, the walk-forward result, or any frozen number. Its job is to close
three reviewer concerns with evidence: that no real match is duplicated in the
training data, that the model is not pathologically overconfident, and that every
probability vector it emits is a valid distribution.

The pure predicates valid_probability_vector and confidence_stats are reused by
the structural gates so they guard every future change, including the ensemble.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from architect_wc import calibrate, ingest

PROBABILITY_TOLERANCE = 1e-9

# Result labels indexed by the ordered outcome: 0 home win, 1 draw, 2 away win.
OUTCOME_LABELS = ("H", "D", "A")

# Roughly this many holdout matches are printed for the eyeball check.
SAMPLE_SIZE = 15


def valid_probability_vector(
    probs: list[float], tolerance: float = PROBABILITY_TOLERANCE
) -> bool:
    """True if probs are non-negative, none above 1, and sum to 1 within tolerance.

    The basic correctness invariant for any per-match prediction. Pure function.
    """
    if any(p < -tolerance for p in probs):
        return False
    if any(p > 1.0 + tolerance for p in probs):
        return False
    return abs(sum(probs) - 1.0) <= tolerance


def confidence_stats(prob_vectors: list[list[float]]) -> dict[str, float]:
    """Max-class probability distribution across many predictions.

    Reports the fraction of predictions whose top class exceeds 0.90 and 0.80, and
    the mean top-class probability. A model that is rarely above 0.80 with a
    moderate mean is producing spread probabilities, not pathological near-certain
    ones. Pure function.
    """
    tops = [max(vector) for vector in prob_vectors]
    n = len(tops)
    if n == 0:
        raise ValueError("Cannot compute confidence stats from zero predictions.")
    return {
        "n": n,
        "frac_above_0_90": sum(t > 0.90 for t in tops) / n,
        "frac_above_0_80": sum(t > 0.80 for t in tops) / n,
        "mean_max_prob": sum(tops) / n,
    }


def run_audit(matches: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Run the data and calibration audit and return a structured report.

    Checks the leakage-guarded training data for duplicate matches, then refits
    the most recent walk-forward window, the calibration anchor, and inspects the
    probabilities it emits for that window's holdout: a sample with actuals, the
    overconfidence statistics, and a count of any invalid probability vectors.
    """
    duplicates = ingest.duplicate_matches(matches)
    redundant = int(matches.duplicated(subset=ingest.DUPLICATE_KEYS).sum())
    duplicate_sample = [
        {"date": str(row.date), "home_team": row.home_team, "away_team": row.away_team}
        for row in duplicates.head(8).itertuples(index=False)
    ]

    # The most recent walk-forward window is the calibration anchor.
    window = calibrate.prepare_backtest(matches, config)
    prob_vectors, outcomes = calibrate.window_predictions(window, config)
    confidence = confidence_stats(prob_vectors)
    n_invalid = sum(not valid_probability_vector(vector) for vector in prob_vectors)

    # Evenly spaced sample so the table is not all from a single date.
    n = len(prob_vectors)
    step = max(1, n // SAMPLE_SIZE)
    sample = []
    for index in range(0, n, step):
        if len(sample) >= SAMPLE_SIZE:
            break
        match = window.matches[index]
        vector = prob_vectors[index]
        sample.append(
            {
                "home_team": match.home_team,
                "away_team": match.away_team,
                "p_home": vector[0],
                "p_draw": vector[1],
                "p_away": vector[2],
                "actual": OUTCOME_LABELS[outcomes[index]],
            }
        )

    overconfident = (
        confidence["frac_above_0_90"] > 0.20 or confidence["mean_max_prob"] > 0.80
    )
    return {
        "n_training_matches": int(len(matches)),
        "duplicate_rows": int(len(duplicates)),
        "redundant_rows": redundant,
        "duplicate_sample": duplicate_sample,
        "window": {
            "holdout_start": window.holdout_start,
            "holdout_end": window.holdout_end,
            "n_matches": len(window.matches),
        },
        "n_probability_vectors": n,
        "n_invalid_vectors": n_invalid,
        "confidence": confidence,
        "overconfident": overconfident,
        "sample_predictions": sample,
    }


def format_audit(report: dict[str, Any]) -> str:
    """Render the audit report as readable stdout: duplicates, sample, confidence."""
    lines = ["World Cup leakage and calibration audit", ""]

    lines.append("1. Duplicate fixtures (date, home_team, away_team):")
    if report["duplicate_rows"] == 0:
        lines.append(
            f"   None. All {report['n_training_matches']} training matches are "
            "unique on date, home, and away. Clean."
        )
    else:
        lines.append(
            f"   FOUND {report['redundant_rows']} redundant rows across "
            f"{report['duplicate_rows']} rows in duplicate groups. Sample:"
        )
        for row in report["duplicate_sample"]:
            lines.append(
                f"     {row['date']}  {row['home_team']} vs {row['away_team']}"
            )

    window = report["window"]
    lines.extend(
        [
            "",
            (
                f"2. Calibration sanity on the most recent window "
                f"({window['holdout_start']} to {window['holdout_end']}, "
                f"{window['n_matches']} matches)."
            ),
            "   Sample predicted probabilities, home/draw/away, with the actual:",
            f"     {'home':<18} {'away':<18} {'p_home':>7} {'p_draw':>7}"
            f" {'p_away':>7}  actual",
        ]
    )
    for row in report["sample_predictions"]:
        lines.append(
            f"     {row['home_team']:<18.18} {row['away_team']:<18.18}"
            f" {row['p_home']:7.3f} {row['p_draw']:7.3f} {row['p_away']:7.3f}"
            f"  {row['actual']}"
        )

    confidence = report["confidence"]
    lines.extend(
        [
            "",
            f"   Across all {confidence['n']} scored predictions in the window:",
            f"     max class probability above 0.90: "
            f"{confidence['frac_above_0_90'] * 100:.1f} percent",
            f"     max class probability above 0.80: "
            f"{confidence['frac_above_0_80'] * 100:.1f} percent",
            f"     mean of the max class probability: "
            f"{confidence['mean_max_prob']:.3f}",
            (
                "   Interpretation: elevated confidence, inspect further."
                if report["overconfident"]
                else "   Interpretation: moderate, spread probabilities, not "
                "pathological overconfidence."
            ),
            "",
            (
                f"3. Probability validity: {report['n_invalid_vectors']} of "
                f"{report['n_probability_vectors']} emitted vectors are invalid."
            ),
        ]
    )
    return "\n".join(lines)
