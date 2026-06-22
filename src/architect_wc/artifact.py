"""Versioned output and provenance.

Owns the output contract. Every run writes two files: a timestamped
predictions JSON to outputs/predictions/ and a run log to outputs/logs/. The
run log captures the config used, the as_of_date, the random_seed, the current
git commit sha, and the UTC timestamp. This provenance pattern is mandatory and
every future run must reuse it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREDICTIONS_DIR = Path("outputs/predictions")
LOGS_DIR = Path("outputs/logs")
MODEL_VERSION = "0.0.0"


def get_git_sha() -> str:
    """Return the current git commit sha, or "unknown" outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return result.stdout.strip()


def write_artifact(
    predictions: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
    ratings_summary: dict[str, Any] | None = None,
    model_version: str = MODEL_VERSION,
    predictions_dir: Path = PREDICTIONS_DIR,
    logs_dir: Path = LOGS_DIR,
) -> dict[str, Path]:
    """Write the predictions JSON and run log, returning both paths.

    The predictions document is the single output contract that every
    downstream consumer reads. The run log records provenance so any run can be
    reproduced from a fixed seed and dated data. The optional provenance dict
    records which data snapshot fed the run, and the optional ratings summary
    records a short view of the Elo ratings used.
    """
    predictions_dir = Path(predictions_dir)
    logs_dir = Path(logs_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    generated_at = now.isoformat()
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    as_of_date = str(config.get("as_of_date"))
    git_sha = get_git_sha()

    document = {
        "generated_at": generated_at,
        "as_of_date": as_of_date,
        "model_version": model_version,
        "git_sha": git_sha,
        "predictions": predictions,
    }
    predictions_path = predictions_dir / f"predictions_{stamp}.json"
    predictions_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    run_log = {
        "generated_at": generated_at,
        "as_of_date": as_of_date,
        "random_seed": config.get("random_seed"),
        "git_sha": git_sha,
        "model_version": model_version,
        "config": config,
        "data_provenance": provenance,
        "ratings_summary": ratings_summary,
        "predictions_file": str(predictions_path),
    }
    log_path = logs_dir / f"run_{stamp}.json"
    log_path.write_text(
        json.dumps(run_log, indent=2, default=str) + "\n", encoding="utf-8"
    )

    return {"predictions": predictions_path, "log": log_path}
