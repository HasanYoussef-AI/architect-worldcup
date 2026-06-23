"""Layer 7, ablation.

Owns the with-and-without harness, the project's differentiator. It scores
several model configurations on one shared, leakage-enforced backtest window so
every delta is attributable to a single toggled layer and nothing else.

An honest decoupling drove the design. The Phase 6 per-match backtest scores
probabilities that come only from the Dixon-Coles goal model. Elo (Layer 2) and
squad value (Layer 4) feed the tournament simulator, not the per-match goal
model, so a naive leave-one-out toggling them through this backtest shows them at
exactly zero delta. That is not a harness bug, it is the architecture. So the
ablation is built as a model-versus-model comparison, Dixon-Coles against a
transparent Elo win, draw, loss generator, and squad value is ablated inside the
Elo generator, the path where it actually moves the probabilities.

Configurations, all scored on the identical window from calibrate.prepare_backtest:
  dc_model               Dixon-Coles goal model, the reference, reproduces 0.1611.
  elo_model              Elo win, draw, loss generator with squad value on.
  elo_model_minus_squad  the same Elo generator with squad value off.
  baseline               the Phase 6 base-rate floor, reused for context.

The flags come from the config factors block and default all on, so the forecast
and calibration paths are untouched. The harness flips them, one layer at a time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from architect_wc import calibrate, model, ratings, squad

REFERENCE = "dc_model"

CONVENTION = (
    "Ranked Probability Score, lower is better. The delta is each configuration "
    "minus the dc_model reference. For the squad ablation, if removing squad "
    "value raises the Elo model RPS, that positive delta is the value squad was "
    "adding."
)

DECOUPLING_NOTE = (
    "Elo and squad value are not in the Dixon-Coles per-match scoring path, so a "
    "naive leave-one-out under dc_model would show them at zero delta. That is "
    "why this is a model-versus-model comparison rather than a single-model "
    "leave-one-out, and why squad value is ablated inside the Elo generator, the "
    "path where it actually moves the probabilities."
)

# Each configuration is a name, the layer flags it runs under, and a description.
# The flags are the config factors: elo, dixon_coles, squad_value, and
# friendly_downweight. dc_model runs Dixon-Coles, so its elo and squad flags do
# not touch its probabilities, which is the decoupling stated above. The
# friendly_downweight flag does move the Dixon-Coles fit, so it is toggled on a
# second Dixon-Coles configuration to measure what the downweight is worth.
CONFIGS = [
    {
        "name": "dc_model",
        "flags": {
            "elo": True,
            "dixon_coles": True,
            "squad_value": True,
            "friendly_downweight": False,
        },
        "description": "Dixon-Coles goal model, reference",
    },
    {
        "name": "dc_model_friendly_downweight",
        "flags": {
            "elo": True,
            "dixon_coles": True,
            "squad_value": True,
            "friendly_downweight": True,
        },
        "description": "Dixon-Coles with friendlies downweighted",
    },
    {
        "name": "elo_model",
        "flags": {
            "elo": True,
            "dixon_coles": False,
            "squad_value": True,
            "friendly_downweight": False,
        },
        "description": "Elo win, draw, loss generator, squad value on",
    },
    {
        "name": "elo_model_minus_squad",
        "flags": {
            "elo": True,
            "dixon_coles": False,
            "squad_value": False,
            "friendly_downweight": False,
        },
        "description": "Elo generator, squad value off",
    },
]


def _dixon_coles_probs(
    window: calibrate.BacktestWindow,
    fit_config: dict[str, Any],
) -> list[list[float]]:
    """Score the holdout with the Dixon-Coles goal model. The calibration path."""
    fitted = model.fit_model(window.training, fit_config)
    probs: list[list[float]] = []
    for match in window.matches:
        p = model.match_probabilities(
            fitted, match.home_team, match.away_team, neutral=match.neutral
        )
        probs.append([p["p_home_win"], p["p_draw"], p["p_away_win"]])
    return probs


def _elo_probs(
    window: calibrate.BacktestWindow,
    flags: dict[str, bool],
    config: dict[str, Any],
    squad_values: dict[str, float],
    base_ratings: dict[str, float],
) -> list[list[float]]:
    """Score the holdout with the Elo win, draw, loss generator.

    Starts from the training Elo ratings, optionally flattens them when Elo is
    off, optionally applies the squad-value adjustment, fits the Davidson tie
    parameter on training only, then turns each holdout match into win, draw, loss
    probabilities honouring its venue. Squad value enters here, so toggling it
    produces a real measured delta.
    """
    elo_config = config.get("elo", {}) or {}
    base_rating = float(elo_config.get("base_rating", ratings.BASE_RATING))
    home_advantage = float(elo_config.get("home_advantage", ratings.HOME_ADVANTAGE))

    rating_map = dict(base_ratings)
    if not flags["elo"]:
        rating_map = dict.fromkeys(rating_map, base_rating)
    if flags["squad_value"]:
        adjusted = squad.adjust_ratings(list(rating_map.items()), squad_values, config)
        rating_map = dict(adjusted)

    nu = ratings.fit_draw_parameter(window.training, rating_map, config)

    probs: list[list[float]] = []
    for match in window.matches:
        advantage = 0.0 if match.neutral else home_advantage
        p_home, p_draw, p_away = ratings.elo_win_draw_loss(
            rating_map.get(match.home_team, base_rating),
            rating_map.get(match.away_team, base_rating),
            advantage,
            nu,
        )
        probs.append([p_home, p_draw, p_away])
    return probs


def run_ablation(matches: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Run the ablation on one shared backtest window and return the report.

    Builds the leakage-enforced window once so every configuration scores the
    identical matches, cutoff, venues, and seed, then scores dc_model, the Elo
    generator with and without squad value, and the base-rate baseline. Returns a
    structured report with each configuration mean RPS, the delta against
    dc_model, the within-Elo squad delta, the fixed run parameters, the leakage
    cutoff dates, and the convention and decoupling notes. Deterministic: the
    backtest path has no random component, so the fixed seed and fixed data give
    identical results every run.
    """
    window = calibrate.prepare_backtest(matches, config)
    outcomes = [match.outcome for match in window.matches]

    fit_config = dict(config)
    fit_config["as_of_date"] = window.cutoff_date

    # Shared inputs, computed once. The Elo ratings and squad values are reused
    # across the Elo configurations so only the toggled layer differs.
    base_ratings = dict(ratings.compute_elo(window.training, fit_config))
    squad_values = squad.load_squad_values(config["squad"]["snapshot"])

    train_max = pd.Timestamp(window.train_max_date)
    holdout_start = pd.Timestamp(window.holdout_start)

    scored: dict[str, float] = {}
    for entry in CONFIGS:
        # The leakage guard fires for every configuration, not just the first,
        # including the friendly-downweighted Dixon-Coles fit.
        calibrate.assert_no_leakage(train_max, holdout_start)
        flags = entry["flags"]
        if flags["dixon_coles"]:
            # Thread the friendly-downweight factor into this fit only. The shared
            # fit_config is left untouched so the dc_model reference stays the
            # unweighted path that reproduces the calibration RPS.
            dc_fit_config = dict(fit_config)
            dc_fit_config["factors"] = {
                **(config.get("factors") or {}),
                "friendly_downweight": flags["friendly_downweight"],
            }
            probs = _dixon_coles_probs(window, dc_fit_config)
        else:
            probs = _elo_probs(window, flags, config, squad_values, base_ratings)
        scored[entry["name"]] = calibrate.mean_rps(probs, outcomes)

    rates = calibrate.base_rates(window.training)
    baseline_rps = calibrate.mean_rps([list(rates)] * len(outcomes), outcomes)

    dc_rps = scored[REFERENCE]
    config_rows = [
        {
            "name": entry["name"],
            "mean_rps": scored[entry["name"]],
            "delta_vs_dc": scored[entry["name"]] - dc_rps,
            "description": entry["description"],
        }
        for entry in CONFIGS
    ]
    config_rows.append(
        {
            "name": "baseline",
            "mean_rps": baseline_rps,
            "delta_vs_dc": baseline_rps - dc_rps,
            "description": "base rates, floor",
        }
    )

    squad_delta = scored["elo_model_minus_squad"] - scored["elo_model"]
    squad_interpretation = (
        "Removing squad value raised the Elo model RPS, so squad value was adding "
        "this much."
        if squad_delta > 0
        else "Removing squad value did not raise the Elo model RPS, so squad value "
        "was not adding measured value on this window."
    )

    # Lower RPS is better, so a negative delta means the downweight helped.
    downweight_delta = scored["dc_model_friendly_downweight"] - scored["dc_model"]
    downweight_interpretation = (
        "Downweighting friendlies lowered the Dixon-Coles RPS, so it helped."
        if downweight_delta < 0
        else "Downweighting friendlies did not lower the Dixon-Coles RPS on this "
        "window, so it did not help."
    )

    return {
        "as_of_date": str(config.get("as_of_date")),
        "random_seed": config.get("random_seed"),
        "backtest_size": window.backtest_size,
        "n_matches": len(outcomes),
        "cutoff_date": window.cutoff_date,
        "train_max_date": window.train_max_date,
        "holdout_start": window.holdout_start,
        "holdout_end": window.holdout_end,
        "reference": REFERENCE,
        "convention": CONVENTION,
        "decoupling_note": DECOUPLING_NOTE,
        "configs": config_rows,
        "squad_value_within_elo": {
            "delta_rps": squad_delta,
            "interpretation": squad_interpretation,
        },
        "friendly_downweight_within_dc": {
            "delta_rps": downweight_delta,
            "interpretation": downweight_interpretation,
        },
    }


def format_report(report: dict[str, Any]) -> str:
    """Render the ablation report as a clean stdout table with the convention."""
    lines = [
        "World Cup ablation report",
        (
            f"Backtest: {report['n_matches']} matches, holdout "
            f"{report['holdout_start']} to {report['holdout_end']}, trained up to "
            f"{report['train_max_date']}, seed {report['random_seed']}."
        ),
        f"Convention: {report['convention']}",
        "",
        f"  {'config':<30} {'mean_rps':>9} {'delta_vs_dc':>12}   description",
    ]
    for row in report["configs"]:
        lines.append(
            f"  {row['name']:<30} {row['mean_rps']:9.4f} {row['delta_vs_dc']:+12.4f}"
            f"   {row['description']}"
        )
    squad = report["squad_value_within_elo"]
    downweight = report["friendly_downweight_within_dc"]
    lines.extend(
        [
            "",
            (
                "Squad value inside the Elo model: removing it changes RPS by "
                f"{squad['delta_rps']:+.4f}."
            ),
            squad["interpretation"],
            "",
            (
                "Friendly downweight inside Dixon-Coles: turning it on changes RPS "
                f"by {downweight['delta_rps']:+.4f}."
            ),
            downweight["interpretation"],
            "",
            f"Decoupling note: {report['decoupling_note']}",
        ]
    )
    return "\n".join(lines)
