from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from agenticdatapipe.config import Settings
from agenticdatapipe.training import (
    TARGET_COLUMN,
    chronological_split,
    generate_synthetic_history,
    should_promote,
)


def test_synthetic_history_is_bounded_and_shifted(tmp_path):
    path = tmp_path / "history.parquet"
    settings = Settings(
        feast_history_path=path,
        synthetic_history_days=7,
        synthetic_station_count=3,
        synthetic_seed=7,
    )
    result = generate_synthetic_history(
        settings, end_at=datetime(2026, 9, 17, 12, 7, tzinfo=UTC)
    )
    frame = pd.read_parquet(path)

    assert result["source"] == "synthetic"
    assert result["stations"] == 3
    assert (frame["available_bikes"] >= 0).all()
    assert (frame["available_bikes"] <= frame["capacity"]).all()
    assert (frame["available_docks"] == frame["capacity"] - frame["available_bikes"]).all()
    for _, station in frame.groupby("station_id"):
        assert station.iloc[:-1][TARGET_COLUMN].tolist() == station.iloc[1:][
            "available_bikes"
        ].astype(float).tolist()
        assert pd.isna(station.iloc[-1][TARGET_COLUMN])


def test_synthetic_history_is_reproducible(tmp_path):
    end = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    first = Settings(
        feast_history_path=tmp_path / "first.parquet",
        synthetic_history_days=7,
        synthetic_station_count=2,
        synthetic_seed=17,
    )
    second = first.model_copy(update={"feast_history_path": tmp_path / "second.parquet"})
    generate_synthetic_history(first, end_at=end)
    generate_synthetic_history(second, end_at=end)
    pd.testing.assert_frame_equal(
        pd.read_parquet(first.feast_history_path),
        pd.read_parquet(second.feast_history_path),
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
    assert "prepared >> materialized >> trained" in source
