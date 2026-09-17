"""Reliable outer scheduling for the stateful LangGraph station pipeline."""

from datetime import UTC, datetime, timedelta

from airflow.providers.standard.sensors.python import PythonSensor
from airflow.sdk import dag, task

from agenticdatapipe.airflow_tasks import kafka_has_backlog, process_station_batch


@dag(
    dag_id="bike_station_pipeline",
    schedule="* * * * *",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(seconds=30),
        "retry_exponential_backoff": True,
    },
    tags=["bike-sharing", "langgraph", "mlops"],
)
def bike_station_pipeline():
    wait_for_events = PythonSensor(
        task_id="wait_for_station_events",
        python_callable=kafka_has_backlog,
        mode="reschedule",
        poke_interval=10,
        timeout=50,
        soft_fail=True,
    )

    @task(task_id="run_langgraph_batch")
    def run_graph() -> dict:
        return process_station_batch()

    wait_for_events >> run_graph()


bike_station_pipeline()
