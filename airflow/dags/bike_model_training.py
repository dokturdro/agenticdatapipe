from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task

from agenticdatapipe.airflow_tasks import (
    build_training_features,
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
    tags=["bike-sharing", "dbt", "feast", "mlflow", "training"],
)
def bike_model_training():
    @task(task_id="prepare_training_source")
    def prepare() -> dict:
        return prepare_training_history()

    @task(task_id="apply_and_materialize_feast")
    def materialize() -> dict:
        return materialize_training_features()

    @task(task_id="build_and_test_dbt_features")
    def build_features() -> dict:
        return build_training_features()

    @task(task_id="train_register_and_promote")
    def train() -> dict:
        return train_registered_model()

    prepared = prepare()
    built = build_features()
    materialized = materialize()
    trained = train()
    prepared >> built >> materialized >> trained


bike_model_training()
