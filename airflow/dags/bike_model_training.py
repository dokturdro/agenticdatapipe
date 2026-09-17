"""Manual Stage 3 workflow for Feast materialization and MLflow training."""

from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task

from agenticdatapipe.airflow_tasks import (
    materialize_training_features,
    prepare_training_history,
    train_registered_model,
)


@dag(
    dag_id="bike_model_training",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(seconds=30),
    },
    tags=["bike-sharing", "feast", "mlflow", "training"],
)
def bike_model_training():
    @task(task_id="prepare_synthetic_history")
    def prepare() -> dict:
        return prepare_training_history()

    @task(task_id="apply_and_materialize_feast")
    def materialize() -> dict:
        return materialize_training_features()

    @task(task_id="train_register_and_promote")
    def train() -> dict:
        return train_registered_model()

    prepared = prepare()
    materialized = materialize()
    trained = train()
    prepared >> materialized >> trained


bike_model_training()
