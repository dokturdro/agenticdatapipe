from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agenticdatapipe.config import Settings


def _absolute(path: Path) -> str:
    return str(path.resolve())


def _dbt_executable(environment: dict[str, str]) -> str:
    executable = shutil.which("dbt", path=environment.get("PATH"))
    if executable:
        return executable
    adjacent = Path(sys.executable).with_name("dbt.exe" if os.name == "nt" else "dbt")
    if adjacent.is_file():
        return str(adjacent)
    raise RuntimeError("dbt executable is not installed in the active Python environment")


def build_training_dataset(settings: Settings) -> dict[str, Any]:
    settings.dbt_path.parent.mkdir(parents=True, exist_ok=True)
    settings.dbt_target_path.mkdir(parents=True, exist_ok=True)
    settings.dbt_log_path.mkdir(parents=True, exist_ok=True)
    settings.feast_history_path.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "BIKE_DBT_PATH": _absolute(settings.dbt_path),
            "BIKE_DBT_TARGET_PATH": _absolute(settings.dbt_target_path),
            "BIKE_DBT_LOG_PATH": _absolute(settings.dbt_log_path),
            "DBT_TARGET_PATH": _absolute(settings.dbt_target_path),
            "DBT_LOG_PATH": _absolute(settings.dbt_log_path),
            "PYTHONUTF8": "1",
            "BIKE_TRAINING_SOURCE": settings.training_source,
            "BIKE_SYNTHETIC_RAW_HISTORY_PATH": _absolute(
                settings.synthetic_raw_history_path
            ),
            "BIKE_FEAST_HISTORY_PATH": _absolute(settings.feast_history_path),
            "BIKE_OBSERVATIONS_URI": settings.s3_uri(settings.observations_table),
            "BIKE_S3_ACCESS_KEY_ID": settings.s3_access_key_id,
            "BIKE_S3_SECRET_ACCESS_KEY": settings.s3_secret_access_key,
            "BIKE_S3_REGION": settings.s3_region,
            "BIKE_DUCKDB_S3_ENDPOINT": environment.get(
                "BIKE_DUCKDB_S3_ENDPOINT", urlsplit(settings.s3_endpoint).netloc
            ),
        }
    )
    command = [
        _dbt_executable(environment),
        "build",
        "--project-dir",
        _absolute(settings.dbt_project_dir),
        "--profiles-dir",
        _absolute(settings.dbt_profiles_dir),
        "--target",
        "dev",
    ]
    completed = subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="")
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"dbt build failed with exit code {completed.returncode}: {detail}")
    if not settings.feast_history_path.is_file():
        raise RuntimeError(f"dbt succeeded without creating {settings.feast_history_path}")

    results_path = settings.dbt_target_path / "run_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    statuses: dict[str, int] = {}
    for result in results.get("results", []):
        status = str(result.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "source": settings.training_source,
        "history_path": str(settings.feast_history_path),
        "run_results_path": str(results_path),
        "statuses": statuses,
    }
