"""Verification gate: the hybrid ensemble, structural only.

Hermetic, on a small synthetic league. The highest-risk part is the point-in-time
Dixon-Coles ability feature, so it is gated hardest: the ability and the other
features for a match must not change when the outcomes of that match and every
later match are tampered, which proves they use no data from the match's date or
later. The other gates check that the ensemble emits valid probability vectors and
that it is deterministic under a fixed seed. The empirical RPS is not gated.
"""

from __future__ import annotations

import pandas as pd
import pytest

from architect_wc import audit, ensemble

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

# Draw-free, clear gradient, so the grid Dixon-Coles fits stay stable.
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


def _matches(repeats: int = 70) -> pd.DataFrame:
    fixtures = _BLOCK * repeats
    dates = pd.date_range("2023-03-01", periods=len(fixtures), freq="D")
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
        {"team": ["A", "B", "C", "D"], "market_value_eur_m": [900, 600, 300, 100]}
    ).to_csv(squad_path, index=False)
    config = {
        "as_of_date": "2025-06-01",
        "random_seed": 42,
        "elo": {"base_rating": 1500, "k_factor": 32, "home_advantage": 65},
        "dixon_coles": {"xi": 0.0018},
        "squad": {"snapshot": str(squad_path), "weight": 40, "max_adjustment": 120},
        "calibrate": {"backtest_size": 12, "walk_windows": 2},
        "ensemble": {
            "grid_cadence_days": 120,
            "grid_lookback_days": 3650,
            "grid_years": 2,
            "min_fit_matches": 12,
            "max_iter": 60,
            "learning_rate": 0.1,
        },
    }
    return _matches(), config


@pytest.fixture(scope="module")
def features(synthetic):
    matches, config = synthetic
    return ensemble.build_features(matches, config)


def test_features_are_leakage_safe_against_tampering(synthetic, features) -> None:
    matches, config = synthetic
    # A target with a defined ability and later matches still to come.
    defined = features[features["ability_diff"].notna()]
    target_index = defined.index[len(defined) // 2]
    target_date = features.loc[target_index, "date"]
    before = features.loc[target_index, ensemble.FEATURE_COLUMNS]

    tampered = matches.copy()
    mask = pd.to_datetime(tampered["date"]) >= target_date
    assert mask.sum() > 1  # the target and genuine later matches are tampered
    tampered.loc[mask, "home_score"] = 7
    tampered.loc[mask, "away_score"] = 0
    after = ensemble.build_features(tampered, config).loc[
        target_index, ensemble.FEATURE_COLUMNS
    ]

    # Identical features prove they used no data from the match's date or later,
    # the core leakage guarantee for the point-in-time ability.
    assert before.equals(after)


def test_ability_grid_actually_produced_values(features) -> None:
    # Guard the gate above against vacuity: the ability feature must be defined for
    # a real share of matches, or leakage-safety would be trivially true on NaN.
    assert features["ability_diff"].notna().sum() > 50


def test_ensemble_emits_valid_probabilities(synthetic, features) -> None:
    matches, config = synthetic
    from architect_wc import calibrate

    for window in calibrate.walk_forward_windows(matches, config):
        vectors, _classifier = ensemble.fit_and_predict(features, window, config)
        assert vectors
        for vector in vectors:
            assert audit.valid_probability_vector(vector)


def test_ensemble_is_deterministic(synthetic, features) -> None:
    matches, config = synthetic
    from architect_wc import calibrate

    window = next(calibrate.walk_forward_windows(matches, config))
    first, _ = ensemble.fit_and_predict(features, window, config)
    second, _ = ensemble.fit_and_predict(features, window, config)
    assert first == second
