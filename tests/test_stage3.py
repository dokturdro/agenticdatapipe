from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from agenticdatapipe.config import Settings
from agenticdatapipe.training import (
    chronological_split,
    generate_synthetic_observations,
    should_promote,
)


def test_synthetic_observations_are_raw_and_bounded(tmp_path):
    path = tmp_path / "synthetic.parquet"
    settings = Settings(
        synthetic_raw_history_path=path,
        synthetic_history_days=7,
        synthetic_station_count=3,
        synthetic_seed=7,
    )
    result = generate_synthetic_observations(
        settings, end_at=datetime(2026, 9, 17, 12, 7, tzinfo=UTC)
    )
    frame = pd.read_parquet(path)

    assert result["source"] == "synthetic_raw"
    assert result["stations"] == 3
    assert list(frame.columns) == [
        "station_id",
        "last_reported",
        "num_bikes_available",
        "capacity",
        "num_docks_available",
    ]
    assert (frame["num_bikes_available"] >= 0).all()
    assert (frame["num_bikes_available"] <= frame["capacity"]).all()
    assert (
        frame["num_docks_available"] == frame["capacity"] - frame["num_bikes_available"]
    ).all()


def test_synthetic_history_is_reproducible(tmp_path):
    end = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    first = Settings(
        synthetic_raw_history_path=tmp_path / "first.parquet",
        synthetic_history_days=7,
        synthetic_station_count=2,
        synthetic_seed=17,
    )
    second = first.model_copy(
        update={"synthetic_raw_history_path": tmp_path / "second.parquet"}
    )
    generate_synthetic_observations(first, end_at=end)
    generate_synthetic_observations(second, end_at=end)
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.synthetic_raw_history_path),
        pd.read_parquet(second.synthetic_raw_history_path),
    )


def test_chronological_split_keeps_timestamps_disjoint():
    timestamps = pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        [
            {"station_id": station, "event_timestamp": timestamp}
            for timestamp in timestamps
            for station in ("one", "two")
        ]
    )
    train, validation, test = chronological_split(frame)
    train_times = set(train["event_timestamp"])
    validation_times = set(validation["event_timestamp"])
    test_times = set(test["event_timestamp"])
    assert train_times.isdisjoint(validation_times)
    assert train_times.isdisjoint(test_times)
    assert validation_times.isdisjoint(test_times)
    assert max(train_times) < min(validation_times) < min(test_times)


def test_promotion_requires_strict_baseline_improvement():
    assert should_promote(1.49, 1.5)
    assert not should_promote(1.5, 1.5)
    assert not should_promote(1.51, 1.5)


def test_training_dag_is_manual_and_ordered():
    source = Path("airflow/dags/bike_model_training.py").read_text(encoding="utf-8")
    assert 'dag_id="bike_model_training"' in source
    assert "schedule=None" in source
    assert "prepared >> built >> materialized >> trained" in source
