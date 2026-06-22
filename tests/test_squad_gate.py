"""Verification gate: squad-value adjustment.

Hermetic test, no network, on a small synthetic set. Checks the core
properties of Layer 4: an above-average squad value lifts a team and a
below-average one tempers it down, a team at the field mean is left roughly
unchanged, no nudge exceeds max_adjustment in absolute value, a team missing
from the squad data is returned unchanged, and the adjustment is deterministic.
Also checks that load_squad_values fails clearly on a missing file or column.
"""

from __future__ import annotations

import pandas as pd
import pytest

from architect_wc import squad

CONFIG = {"squad": {"weight": 30, "max_adjustment": 75}}

# Three valued teams with a clean mean of 100, plus one team with no value.
# High sits above the field, Low below, Mid exactly on the mean.
RATINGS = [
    ("High", 1600.0),
    ("Mid", 1500.0),
    ("Low", 1400.0),
    ("NoValue", 1550.0),
]
SQUAD_VALUES = {"High": 200.0, "Mid": 100.0, "Low": 0.0}


def _adjusted_by_team(config: dict) -> dict[str, float]:
    return dict(squad.adjust_ratings(RATINGS, SQUAD_VALUES, config))


def test_above_average_rises_and_below_average_falls() -> None:
    result = _adjusted_by_team(CONFIG)
    assert result["High"] > 1600.0
    assert result["Low"] < 1400.0


def test_team_at_mean_is_unchanged() -> None:
    result = _adjusted_by_team(CONFIG)
    # Mid sits exactly on the field mean, so its z-score, and its nudge, is zero.
    assert result["Mid"] == pytest.approx(1500.0, abs=1e-9)


def test_no_nudge_exceeds_max_adjustment() -> None:
    # A large weight against a small cap forces the clamp to bind for High and
    # Low, so the cap is what is actually under test here.
    config = {"squad": {"weight": 1000, "max_adjustment": 50}}
    base = dict(RATINGS)
    for team, adjusted in squad.adjust_ratings(RATINGS, SQUAD_VALUES, config):
        assert abs(adjusted - base[team]) <= config["squad"]["max_adjustment"]


def test_team_without_squad_value_is_unchanged() -> None:
    result = _adjusted_by_team(CONFIG)
    assert result["NoValue"] == 1550.0


def test_adjustment_is_deterministic() -> None:
    first = squad.adjust_ratings(RATINGS, SQUAD_VALUES, CONFIG)
    second = squad.adjust_ratings(RATINGS, SQUAD_VALUES, CONFIG)
    assert first == second


def test_load_squad_values_reads_two_columns(tmp_path) -> None:
    path = tmp_path / "squad_values.csv"
    pd.DataFrame({"team": ["Alpha", "Beta"], "market_value_eur_m": [500, 250]}).to_csv(
        path, index=False
    )
    values = squad.load_squad_values(path)
    assert values == {"Alpha": 500.0, "Beta": 250.0}


def test_load_squad_values_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        squad.load_squad_values(tmp_path / "absent.csv")


def test_load_squad_values_missing_column_raises(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"team": ["Alpha"], "value": [500]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        squad.load_squad_values(path)
