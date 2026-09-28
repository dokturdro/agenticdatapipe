from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from deltalake import DeltaTable

from agenticdatapipe.audit import persist_prediction_audit_batch
from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import Batch, PredictionAudit
from agenticdatapipe.monitoring import (
    evaluate_monitoring_window,
    join_predictions_to_actuals,
    population_stability_index,
)
from agenticdatapipe.training import FEATURE_COLUMNS


def audit_event(
    event_id: str,
    station_id: str,
    observed_at: datetime,
    *,
    available_bikes: int,
    prediction: int,
) -> PredictionAudit:
    return PredictionAudit(
        event_id=event_id,
        request_id=event_id,
        station_id=station_id,
        served_at=observed_at + timedelta(minutes=1),
        feature_observed_at=observed_at,
        target_at=observed_at + timedelta(minutes=15),
        model_name="bike-availability-15m",
        model_version="7",
        model_alias="champion",
        raw_prediction=float(prediction),
        predicted_available_bikes_15m=prediction,
        hour_utc=observed_at.hour,
        day_of_week=observed_at.weekday(),
        available_bikes=available_bikes,
        available_docks=10 - available_bikes,
        capacity=10,
        availability_ratio=available_bikes / 10,
    )


def test_prediction_audit_delta_upsert_is_idempotent(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        prediction_audits_table=str(tmp_path / "prediction-audits"),
    )
    event = audit_event(
        "request-1",
        "station-1",
        datetime(2026, 1, 1, tzinfo=UTC),
        available_bikes=5,
        prediction=7,
    )
    batch = Batch(batch_id="batch-1", records=[event.model_dump(mode="json")])

    persist_prediction_audit_batch(settings, batch)
    persist_prediction_audit_batch(settings, batch)

    frame = DeltaTable(settings.prediction_audits_table).to_pandas()
    assert len(frame) == 1
    assert frame.iloc[0]["event_id"] == "request-1"


def test_label_join_uses_nearest_observation_within_tolerance():
    observed_at = datetime(2026, 1, 1, tzinfo=UTC)
    audits = pd.DataFrame(
        [audit_event("one", "station-1", observed_at, available_bikes=5, prediction=7).model_dump()]
    )
    observations = pd.DataFrame(
        [
            {
                "station_id": "station-1",
                "last_reported": int((observed_at + timedelta(minutes=15, seconds=45)).timestamp()),
                "num_bikes_available": 7,
            }
        ]
    )

    joined = join_predictions_to_actuals(audits, observations, tolerance_seconds=120)
    assert joined.iloc[0]["actual_available_bikes"] == 7
    assert join_predictions_to_actuals(
        audits, observations, tolerance_seconds=30
    ).iloc[0]["actual_available_bikes"] != 7


def test_psi_distinguishes_stable_shifted_and_constant_inputs():
    reference = np.arange(100)
    assert population_stability_index(reference, reference.copy()) == 0
    assert population_stability_index(reference, reference + 100) > 0.25
    assert population_stability_index(np.ones(10), np.ones(10)) == 0
    assert population_stability_index(np.ones(10), np.full(10, 2)) == 1


def test_monitoring_window_passes_when_model_beats_baseline():
    observed_at = datetime(2026, 1, 1, tzinfo=UTC)
    events = [
        audit_event("one", "station-1", observed_at, available_bikes=5, prediction=7),
        audit_event(
            "two",
            "station-2",
            observed_at + timedelta(minutes=1),
            available_bikes=8,
            prediction=6,
        ),
    ]
    audits = pd.DataFrame([event.model_dump() for event in events])
    observations = pd.DataFrame(
        [
            {
                "station_id": event.station_id,
                "last_reported": int(event.target_at.timestamp()),
                "num_bikes_available": event.predicted_available_bikes_15m,
            }
            for event in events
        ]
    )
    reference = audits[FEATURE_COLUMNS].copy()
    settings = Settings(
        monitoring_min_samples=2,
        monitoring_psi_warning=10,
        monitoring_psi_failure=20,
    )
    report, joined, stations = evaluate_monitoring_window(
        audits,
        observations,
        reference,
        audits["raw_prediction"].to_numpy(),
        settings,
        model_version="7",
        window_start=observed_at - timedelta(minutes=1),
        window_end=observed_at + timedelta(hours=1),
        training_test_mae=0.5,
    )

    assert report.status == "passed"
    assert report.metrics["production_mae"] == 0
    assert report.metrics["persistence_baseline_mae"] == 2
    assert len(joined) == 2
    assert set(stations) == {"station-1", "station-2"}


def test_monitoring_dags_are_scheduled_and_bounded():
    audit_source = Path("airflow/dags/bike_prediction_audit.py").read_text(encoding="utf-8")
    monitor_source = Path("airflow/dags/bike_quality_monitoring.py").read_text(encoding="utf-8")
    assert 'dag_id="bike_prediction_audit"' in audit_source
    assert 'schedule="* * * * *"' in audit_source
    assert 'dag_id="bike_quality_monitoring"' in monitor_source
    assert 'schedule="0 * * * *"' in monitor_source
    assert "max_active_runs=1" in audit_source
    assert "max_active_runs=1" in monitor_source
