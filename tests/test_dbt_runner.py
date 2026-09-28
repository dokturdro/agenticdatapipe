import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agenticdatapipe.config import Settings
from agenticdatapipe.dbt_runner import build_training_dataset


@pytest.fixture(autouse=True)
def dbt_executable(monkeypatch):
    monkeypatch.setattr("agenticdatapipe.dbt_runner._dbt_executable", lambda _environment: "dbt")


def settings_for(tmp_path: Path) -> Settings:
    project = tmp_path / "dbt"
    project.mkdir()
    (project / "profiles.yml").write_text("profiles", encoding="utf-8")
    return Settings(
        dbt_project_dir=project,
        dbt_profiles_dir=project,
        dbt_path=tmp_path / "state/bike.duckdb",
        dbt_target_path=tmp_path / "target",
        dbt_log_path=tmp_path / "logs",
        synthetic_raw_history_path=tmp_path / "synthetic.parquet",
        feast_history_path=tmp_path / "feast/history.parquet",
    )


def test_dbt_runner_returns_artifact_summary(monkeypatch, tmp_path):
    settings = settings_for(tmp_path)

    def run(command, **kwargs):
        Path(kwargs["env"]["BIKE_FEAST_HISTORY_PATH"]).touch()
        target = Path(kwargs["env"]["BIKE_DBT_TARGET_PATH"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "run_results.json").write_text(
            json.dumps({"results": [{"status": "success"}, {"status": "pass"}]}),
            encoding="utf-8",
        )
        assert Path(command[0]).name in {"dbt", "dbt.exe"}
        assert command[1] == "build"
        assert kwargs["env"]["BIKE_TRAINING_SOURCE"] == "synthetic"
        return SimpleNamespace(returncode=0, stdout="built\n", stderr="")

    monkeypatch.setattr("agenticdatapipe.dbt_runner.subprocess.run", run)
    result = build_training_dataset(settings)
    assert result["statuses"] == {"success": 1, "pass": 1}


def test_dbt_runner_propagates_failure(monkeypatch, tmp_path):
    settings = settings_for(tmp_path)
    monkeypatch.setattr(
        "agenticdatapipe.dbt_runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad model"),
    )
    with pytest.raises(RuntimeError, match="bad model"):
        build_training_dataset(settings)
