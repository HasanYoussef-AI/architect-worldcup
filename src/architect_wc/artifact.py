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
    forecast_summary: dict[str, Any] | None = None,
    model_version: str = MODEL_VERSION,
    predictions_dir: Path = PREDICTIONS_DIR,
    logs_dir: Path = LOGS_DIR,
) -> dict[str, Path]:
    """Write the predictions JSON and run log, returning both paths.

    The predictions document is the single output contract that every downstream
    consumer reads. The run log records provenance so any run can be reproduced
    from a fixed seed and dated data. The optional provenance dict records which
    data snapshot fed the run, the ratings summary a short view of the ratings,
    and the forecast summary the cutoff, the leakage proof, and the current
    standings for a live dated forecast. The filename carries the as_of date so a
    dated forecast is a separate, identifiable frozen artifact.
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
    tag = f"asof{as_of_date}_{stamp}"

    document = {
        "generated_at": generated_at,
        "as_of_date": as_of_date,
        "model_version": model_version,
        "git_sha": git_sha,
        "forecast_summary": forecast_summary,
        "predictions": predictions,
    }
    predictions_path = predictions_dir / f"predictions_{tag}.json"
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
        "forecast_summary": forecast_summary,
        "predictions_file": str(predictions_path),
    }
    log_path = logs_dir / f"run_{tag}.json"
    log_path.write_text(
        json.dumps(run_log, indent=2, default=str) + "\n", encoding="utf-8"
    )

    return {"predictions": predictions_path, "log": log_path}


def write_calibration_log(
    results: dict[str, Any],
    config: dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
    model_version: str = MODEL_VERSION,
    logs_dir: Path = LOGS_DIR,
) -> Path:
    """Write a calibration run log and return its path.

    Calibration is a reporting path, not a forecast, so it writes only a run log,
    not a predictions document. The same provenance pattern applies: the log
    records the config, the random seed, the git sha, the UTC timestamp, the data
    snapshot that fed the backtest, and the RPS results.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    calibration_log = {
        "generated_at": now.isoformat(),
        "as_of_date": str(config.get("as_of_date")),
        "random_seed": config.get("random_seed"),
        "git_sha": get_git_sha(),
        "model_version": model_version,
        "config": config,
        "data_provenance": provenance,
        "calibration": results,
    }
    log_path = logs_dir / f"calibration_{stamp}.json"
    log_path.write_text(
        json.dumps(calibration_log, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return log_path


def write_ablation_log(
    report: dict[str, Any],
    config: dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
    model_version: str = MODEL_VERSION,
    logs_dir: Path = LOGS_DIR,
) -> Path:
    """Write a versioned ablation artifact with provenance and return its path.

    The ablation is a reporting path, so it writes a single self-describing JSON,
    not a predictions document. It carries the per-config RPS, the deltas, the
    fixed run parameters, and the leakage cutoff dates inside the report, wrapped
    in the same provenance the rest of the project uses: the config, the random
    seed, the git sha, the UTC timestamp, and the data snapshot that fed the
    backtest. Presentation and the README read this file, they do not recompute.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    ablation_log = {
        "generated_at": now.isoformat(),
        "as_of_date": str(config.get("as_of_date")),
        "random_seed": config.get("random_seed"),
        "git_sha": get_git_sha(),
        "model_version": model_version,
        "config": config,
        "data_provenance": provenance,
        "ablation": report,
    }
    log_path = logs_dir / f"ablation_{stamp}.json"
    log_path.write_text(
        json.dumps(ablation_log, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return log_path


def write_walk_forward_log(
    report: dict[str, Any],
    config: dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
    model_version: str = MODEL_VERSION,
    logs_dir: Path = LOGS_DIR,
) -> Path:
    """Write a versioned walk-forward artifact with provenance and return its path.

    Carries the per-window results, the aggregate statistics, the fixed run
    parameters, the per-window leakage cutoff dates, and the explicit window
    definitions including the holdout match identifiers, all inside the report,
    wrapped in the same provenance the rest of the project uses. The next phase
    reuses these exact splits as its evaluation protocol, so the splits must be
    readable and reproducible from this file rather than regenerated by guesswork.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    walk_forward_log = {
        "generated_at": now.isoformat(),
        "as_of_date": str(config.get("as_of_date")),
        "random_seed": config.get("random_seed"),
        "git_sha": get_git_sha(),
        "model_version": model_version,
        "config": config,
        "data_provenance": provenance,
        "walk_forward": report,
    }
    log_path = logs_dir / f"walk_forward_{stamp}.json"
    log_path.write_text(
        json.dumps(walk_forward_log, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return log_path


def write_ensemble_log(
    report: dict[str, Any],
    config: dict[str, Any],
    *,
    provenance: dict[str, Any] | None = None,
    model_version: str = MODEL_VERSION,
    logs_dir: Path = LOGS_DIR,
) -> Path:
    """Write a versioned ensemble artifact with provenance and return its path.

    Carries the per-window ensemble RPS against Dixon-Coles and the baseline, the
    aggregate mean and standard deviation, the feature importances, the fixed run
    parameters, and the leakage cutoff dates, wrapped in the same provenance the
    rest of the project uses. Presentation and the README read this file.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    ensemble_log = {
        "generated_at": now.isoformat(),
        "as_of_date": str(config.get("as_of_date")),
        "random_seed": config.get("random_seed"),
        "git_sha": get_git_sha(),
        "model_version": model_version,
        "config": config,
        "data_provenance": provenance,
        "ensemble": report,
    }
    log_path = logs_dir / f"ensemble_{stamp}.json"
    log_path.write_text(
        json.dumps(ensemble_log, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return log_path
