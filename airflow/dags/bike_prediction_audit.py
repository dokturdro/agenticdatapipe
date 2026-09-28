"""Persist prediction audit events before committing their Kafka offsets."""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.sensors.python import PythonSensor
from airflow.sdk import dag, task

from agenticdatapipe.airflow_tasks import (
    ingest_prediction_audits,
    prediction_audit_has_backlog,
)


@dag(
    dag_id="bike_prediction_audit",
    schedule="* * * * *",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(seconds=30)},
    tags=["bike-sharing", "audit", "kafka", "delta"],
)
def bike_prediction_audit():
    wait_for_events = PythonSensor(
        task_id="wait_for_prediction_audits",
        python_callable=prediction_audit_has_backlog,
        mode="reschedule",
        poke_interval=10,
        timeout=50,
        soft_fail=True,
    )

    @task(task_id="persist_prediction_audits")
    def persist() -> dict:
        return ingest_prediction_audits()

    wait_for_events >> persist()


bike_prediction_audit()
