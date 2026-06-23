"""Layer 1, data integrity.

Owns ingestion of match data into dated, immutable raw snapshots, and the
leakage guard that ensures no information dated after as_of_date enters the
model. Input and output are kept separate from the pure leakage logic so the
core rule is unit-testable without the network.
"""

from __future__ import annotations

import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)
RAW_DIR = Path("data/raw")
SNAPSHOT_GLOB = "results_snapshot_*.csv"
EXPECTED_COLUMNS = [
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
DOWNLOAD_TIMEOUT_SECONDS = 60


def get_snapshot(raw_dir: Path = RAW_DIR) -> Path:
    """Return the path to an immutable results snapshot.

    If a snapshot already exists in raw_dir, return the most recent one and do
    not re-download. Otherwise download results.csv, save it under a UTC-dated
    filename, and return that path. Existing snapshots are never overwritten so
    runs stay reproducible.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(raw_dir.glob(SNAPSHOT_GLOB))
    if existing:
        return existing[-1]

    snapshot_date = datetime.now(UTC).strftime("%Y-%m-%d")
    snapshot_path = raw_dir / f"results_snapshot_{snapshot_date}.csv"
    with urllib.request.urlopen(  # noqa: S310 trusted https source
        RESULTS_URL, timeout=DOWNLOAD_TIMEOUT_SECONDS
    ) as response:
        payload = response.read()
    snapshot_path.write_bytes(payload)
    return snapshot_path


def load_raw(path: Path) -> pd.DataFrame:
    """Read a results snapshot, validate columns, and parse dates.

    Raise a clear error if any expected column is missing.
    """
    df = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Snapshot {path} is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df


def apply_leakage_guard(df: pd.DataFrame, as_of_date: Any) -> pd.DataFrame:
    """Return only the rows dated on or before as_of_date.

    Pure function with no I/O. This is the core data integrity rule: when
    predicting as of a date, nothing dated after that date may enter the data.
    Rows dated exactly on as_of_date are kept.
    """
    cutoff = pd.Timestamp(as_of_date)
    dates = pd.to_datetime(df["date"])
    return df[dates <= cutoff]


def load_matches(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Orchestrate snapshot, load, and leakage guard using config.as_of_date.

    Return the filtered dataframe and a provenance dict with snapshot_path,
    n_matches, and max_date in the filtered set.
    """
    as_of_date = config["as_of_date"]
    snapshot_path = get_snapshot()
    df = load_raw(snapshot_path)
    filtered = apply_leakage_guard(df, as_of_date)

    if len(filtered):
        max_timestamp = pd.to_datetime(filtered["date"]).max()
        max_date = None if pd.isna(max_timestamp) else max_timestamp.date().isoformat()
    else:
        max_date = None

    provenance = {
        "snapshot_path": str(snapshot_path),
        "n_matches": int(len(filtered)),
        "max_date": max_date,
    }
    return filtered, provenance


DUPLICATE_KEYS = ["date", "home_team", "away_team"]


def duplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Return the rows that duplicate another on date, home team, and away team.

    A real fixture is a single row. Two rows sharing the same date, home team, and
    away team are the same match stored twice, which would let the model train on
    a result it is also scored against. keep=False returns every row in any
    duplicate group, so they can be inspected rather than silently dropped.
    """
    return df[df.duplicated(subset=DUPLICATE_KEYS, keep=False)]
