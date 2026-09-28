"""Hourly production quality and drift monitoring recorded in MLflow."""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.sensors.python import PythonSensor
from airflow.sdk import dag, task

from agenticdatapipe.airflow_tasks import monitor_model_quality, monitoring_is_ready


@dag(
    dag_id="bike_quality_monitoring",
    schedule="0 * * * *",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["bike-sharing", "quality", "drift", "mlflow"],
)
def bike_quality_monitoring():
    ready = PythonSensor(
        task_id="wait_for_monitoring_inputs",
        python_callable=monitoring_is_ready,
        mode="reschedule",
        poke_interval=30,
        timeout=300,
        soft_fail=True,
    )

    @task(task_id="evaluate_quality_and_drift")
    def monitor() -> dict:
        return monitor_model_quality()

    ready >> monitor()


bike_quality_monitoring()
