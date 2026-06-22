# Session log

Continuation notes for the Architect WorldCup build. Each session reads the
last "Next" line and resumes from there. See CLAUDE.md for the protocol.

## 2026-06-22  Phase 0  Scaffold verification-first skeleton
- Did: Scaffolded the repo end to end. Created src-layout package architect_wc with layered module stubs (ingest, ratings, model, squad, simulate, calibrate, ablation), the provenance-writing artifact layer, and pipeline.py as the single entry point exposed as the wc-predict console script. Added config.yaml with ablation factor toggles, the predictions JSON schema, the first verification gate (tests/test_schema.py), a uv project pinned to Python 3.12, ruff config, MIT license, README stub, .gitignore with kept data and output dirs, and a CI workflow running ruff and pytest. Initialised git and committed.
- Commit: c1895f5 Scaffold verification-first pipeline skeleton
- Verified: uv sync on Python 3.12.13. ruff check and ruff format both clean. uv run wc-predict writes a timestamped predictions JSON and a provenance run log. uv run pytest passes 1 of 1 (schema validation plus p_win sums to 1.0 within tolerance).
- Next: Phase 1, implement ingest.py data integrity. Load match data into dated immutable raw snapshots under data/raw, add the leakage guard that drops any record dated after as_of_date, and add a no-leakage pytest gate.

## 2026-06-22  Phase 0  Harden verification gate
- Did: Found that the editable install can stop being honored after a uv resync, which broke pytest collection with ModuleNotFoundError mid-session. Set pythonpath = ["src"] in the pytest config so the gate imports architect_wc from src directly, independent of the editable install. Confirmed that a clean uv sync still produces a working editable install and that wc-predict runs.
- Commit: 80e6404 Make schema verification gate independent of editable install
- Verified: With the editable .pth file removed on purpose, uv run pytest still passes 1 of 1. ruff check and ruff format both clean. wc-predict writes a predictions JSON and a run log.
- Next: Phase 1, implement ingest.py data integrity. Load match data into dated immutable raw snapshots under data/raw, add the leakage guard that drops any record dated after as_of_date, and add a no-leakage pytest gate.

## 2026-06-22  Phase 0  Correct license and track session-log
- Did: Replaced the MIT license with proprietary, all rights reserved, view-only terms because the repo is public for evaluation only and must not be reusable. Pointed the pyproject license declaration at the LICENSE file and kept no OSI or MIT classifier. Added a License line to the README. Started tracking session-log.md so a fresh clone keeps the history and session continuity. From now on this file is committed, never left untracked.
- Commit: f3cce08 Replace permissive license with evaluation-only terms. This session-log file is committed for the first time in the immediately following commit, which begins tracking it.
- Verified: uv run ruff check clean. uv run pytest passes 1 of 1. The package still builds under the new license declaration.
- Next: Phase 1, implement ingest.py data integrity. Load match data into dated immutable raw snapshots under data/raw, add the leakage guard that drops any record dated after as_of_date, and add a no-leakage pytest gate.

## 2026-06-22  Phase 1  Build Layer 1 data integrity
- Did: Implemented ingest.py as the data integrity layer with input and output separated from pure logic. get_snapshot downloads the martj42 international results.csv once into an immutable, UTC-dated snapshot under data/raw and never re-downloads or overwrites it. load_raw validates the expected columns and parses dates, raising a clear error on a missing column. apply_leakage_guard is a pure, I/O-free function that keeps only rows dated on or before as_of_date. load_matches orchestrates the three using config.as_of_date and returns the filtered frame plus a provenance dict of snapshot_path, n_matches, and max_date. Wired pipeline.py to load matches and record that provenance in the run log via a new optional provenance argument to write_artifact, while still emitting the placeholder predictions. Added tests/test_ingest_no_leakage.py and made the schema gate hermetic so neither gate touches the network.
- Commit: e328eca Build Layer 1 data integrity with a no-leakage gate. This session-log entry is committed in the immediately following commit, which keeps the file tracked.
- Verified: uv run ruff check clean. uv run ruff format clean. uv run pytest passes 2 of 2, including the new no-leakage gate. A real run loaded 49449 matches from results_snapshot_2026-06-22.csv, filtered to as_of_date 2026-06-22 with max_date 2026-06-22, and recorded the provenance in the run log.
- Next: Phase 2, implement ratings.py Elo. Compute team Elo ratings from the leakage-guarded matches, expose them for the model layer, and add a ratings sanity test.
