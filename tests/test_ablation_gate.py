"""Verification gate: the ablation harness, structural only.

Hermetic test, no network, on a small synthetic league. These gates prove the
harness is trustworthy without asserting the empirical ordering, because the
ordering is the finding, not a property to enforce. They check that the dc_model
configuration reproduces the calibration backtest exactly, that the harness is
deterministic, that the leakage guard fires, and one soft sanity check that the
goal model beats the base-rate baseline on this synthetic data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from architect_wc import ablation, calibrate, model

COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]

# A clear strength order A > B > C > D, scores moderate and deterministic by
# pairing so a fitted model beats a flat base rate. Kept draw-free, mirroring the
# proven model-gate league: perfectly repeated 1-1 draws stress the Dixon-Coles
# low-score correction into negative cells on this tiny synthetic set. The draw
# model is exercised by the real backtest, which has real draws; here the gates
# are structural, so a clean fit matters more than draw coverage.
_BLOCK = [
    ("A", "B", 2, 0),
    ("A", "C", 3, 1),
    ("A", "D", 2, 0),
    ("B", "A", 1, 2),
    ("B", "C", 2, 0),
    ("B", "D", 3, 1),
    ("C", "A", 1, 3),
    ("C", "B", 0, 2),
    ("C", "D", 2, 0),
    ("D", "A", 0, 2),
    ("D", "B", 1, 3),
    ("D", "C", 0, 2),
]


def _matches(repeats: int = 12) -> pd.DataFrame:
    fixtures = _BLOCK * repeats
    dates = pd.date_range("2024-01-01", periods=len(fixtures), freq="D")
    rows = [
        {
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "tournament": "League",
            "city": "Citytown",
            "country": "Countryland",
            "neutral": False,
        }
        for (home, away, home_score, away_score), date in zip(
            fixtures, dates, strict=True
        )
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> tuple[pd.DataFrame, dict]:
    squad_path = tmp_path_factory.mktemp("squad") / "squad_values.csv"
    pd.DataFrame(
        {
            "team": ["A", "B", "C", "D"],
            "market_value_eur_m": [900, 600, 300, 100],
        }
    ).to_csv(squad_path, index=False)
    config = {
        "as_of_date": "2025-06-01",
        "random_seed": 42,
        "elo": {"base_rating": 1500, "k_factor": 32, "home_advantage": 65},
        "dixon_coles": {"xi": 0.0018},
        "squad": {
            "snapshot": str(squad_path),
            "weight": 40,
            "max_adjustment": 120,
        },
        "calibrate": {"backtest_size": 12},
    }
    return _matches(), config


def _config_rps(report: dict, name: str) -> float:
    return next(row["mean_rps"] for row in report["configs"] if row["name"] == name)


def test_dc_model_reproduces_the_calibration_backtest(synthetic) -> None:
    matches, config = synthetic
    calibration = calibrate.run_backtest(matches, config)
    report = ablation.run_ablation(matches, config)
    # The harness full path must be the exact path calibration scored, so the
    # dc_model RPS equals the calibration model RPS to floating precision. If it
    # does not, the harness has diverged and every delta is suspect.
    assert abs(_config_rps(report, "dc_model") - calibration["model_mean_rps"]) <= 1e-12


def test_ablation_is_deterministic(synthetic) -> None:
    matches, config = synthetic
    first = ablation.run_ablation(matches, config)
    second = ablation.run_ablation(matches, config)
    assert first == second


def test_assert_no_leakage_fires_on_overlap() -> None:
    with pytest.raises(ValueError, match="Leakage"):
        calibrate.assert_no_leakage(
            pd.Timestamp("2024-03-02"), pd.Timestamp("2024-03-01")
        )


def test_prepare_backtest_keeps_training_before_holdout(synthetic) -> None:
    matches, config = synthetic
    window = calibrate.prepare_backtest(matches, config)
    assert pd.Timestamp(window.train_max_date) < pd.Timestamp(window.holdout_start)


def test_report_records_no_leakage(synthetic) -> None:
    matches, config = synthetic
    report = ablation.run_ablation(matches, config)
    train_max = pd.Timestamp(report["train_max_date"])
    holdout_start = pd.Timestamp(report["holdout_start"])
    assert train_max < holdout_start


def test_dc_model_beats_baseline(synthetic) -> None:
    matches, config = synthetic
    report = ablation.run_ablation(matches, config)
    # Soft sanity only, on the goal model versus the base-rate floor. The
    # elo-versus-dc ordering and the squad sign are findings, so not asserted.
    assert _config_rps(report, "dc_model") < _config_rps(report, "baseline")


def test_importance_weights_off_is_uniform() -> None:
    tournaments = pd.Series(["Friendly", "FIFA World Cup", "Friendly"])
    config = {
        "dixon_coles": {"friendly_weight": 0.5},
        "factors": {"friendly_downweight": False},
    }
    # Off means every match keeps full weight, which is what makes the unweighted
    # dc_model reproduce the calibration RPS exactly.
    assert list(model.importance_weights(tournaments, config)) == [1.0, 1.0, 1.0]


def test_importance_weights_downweights_only_friendlies() -> None:
    tournaments = pd.Series(["Friendly", "FIFA World Cup", "Friendly"])
    config = {
        "dixon_coles": {"friendly_weight": 0.5},
        "factors": {"friendly_downweight": True},
    }
    # Friendlies drop to friendly_weight; competitive matches stay at 1.0.
    assert list(model.importance_weights(tournaments, config)) == [0.5, 1.0, 0.5]


def test_weighted_fit_preserves_no_leakage(synthetic) -> None:
    matches, config = synthetic
    weighted = dict(config)
    weighted["factors"] = {"friendly_downweight": True}
    weighted["dixon_coles"] = {**config["dixon_coles"], "friendly_weight": 0.5}
    # The friendly-downweighted fit goes through the same leakage-guarded window,
    # so training still ends strictly before the holdout starts.
    result = calibrate.run_backtest(matches, weighted)
    train_max = pd.Timestamp(result["train_max_date"])
    holdout_start = pd.Timestamp(result["holdout_start"])
    assert train_max < holdout_start
