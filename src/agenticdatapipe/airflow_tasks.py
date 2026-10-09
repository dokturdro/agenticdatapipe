from typing import Any

from confluent_kafka import Consumer, KafkaException, TopicPartition

from agenticdatapipe.cli import run_batch
from agenticdatapipe.config import Settings


def _kafka_has_backlog(topic_name: str, group_id: str) -> bool:
    settings = Settings()
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": group_id,
            "enable.auto.commit": False,
            "session.timeout.ms": 10000,
        }
    )
    try:
        metadata = consumer.list_topics(topic_name, timeout=10)
        topic = metadata.topics.get(topic_name)
        if topic is None or topic.error is not None:
            detail = topic.error if topic is not None else "topic missing"
            raise KafkaException(detail)
        partitions = [TopicPartition(topic_name, number) for number in topic.partitions]
        committed = consumer.committed(partitions, timeout=10)
        for partition in committed:
            low, high = consumer.get_watermark_offsets(
                TopicPartition(topic_name, partition.partition), timeout=10
            )
            current = partition.offset if partition.offset >= 0 else low
            if high > current:
                return True
        return False
    finally:
        consumer.close()


def kafka_has_backlog() -> bool:
    settings = Settings()
    return _kafka_has_backlog(settings.kafka_topic, settings.kafka_group_id)


def prediction_audit_has_backlog() -> bool:
    from agenticdatapipe.ingestion import ensure_topic

    settings = Settings()
    ensure_topic(settings.model_copy(update={"kafka_topic": settings.prediction_audit_topic}))
    return _kafka_has_backlog(
        settings.prediction_audit_topic, settings.prediction_audit_group_id
    )


def monitoring_is_ready() -> bool:
    from mlflow import MlflowClient

    settings = Settings()
    if not settings.feast_history_path.exists():
        return False
    try:
        MlflowClient(tracking_uri=settings.mlflow_tracking_uri).get_model_version_by_alias(
            settings.mlflow_model_name, settings.mlflow_model_alias
        )
    except Exception:  # noqa: BLE001
        return False
    return True


def process_station_batch() -> dict[str, Any]:
    settings = Settings()
    return run_batch(settings, fixture=settings.fixture_mode)


def prepare_training_history() -> dict[str, Any]:
    from agenticdatapipe.training import generate_synthetic_observations

    settings = Settings()
    if settings.training_source == "collected":
        return {"source": "collected", "status": "not_required"}
    return generate_synthetic_observations(settings)


def build_training_features() -> dict[str, Any]:
    from agenticdatapipe.dbt_runner import build_training_dataset

    return build_training_dataset(Settings())


def materialize_training_features() -> dict[str, Any]:
    from agenticdatapipe.training import apply_and_materialize_features

    return apply_and_materialize_features(Settings())


def train_registered_model() -> dict[str, Any]:
    from agenticdatapipe.training import train_and_register_model

    return train_and_register_model(Settings())


def ingest_prediction_audits() -> dict[str, Any]:
    from agenticdatapipe.audit import process_prediction_audit_batch

    return process_prediction_audit_batch(Settings())


def monitor_model_quality() -> dict[str, Any]:
    from agenticdatapipe.monitoring import run_monitoring

    result = run_monitoring(Settings())
    if result["status"] == "failed":
        raise RuntimeError("Quality/drift monitoring thresholds failed")
    return result
