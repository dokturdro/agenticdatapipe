"""Small callables used by Airflow without coupling the core package to Airflow."""

from typing import Any

from confluent_kafka import Consumer, KafkaException, TopicPartition

from agenticdatapipe.cli import run_batch
from agenticdatapipe.config import Settings


def kafka_has_backlog() -> bool:
    """Return whether the pipeline consumer group has unprocessed topic offsets.

    This performs metadata and offset queries only. It does not consume a message or
    mutate the pipeline consumer group's offsets.
    """
    settings = Settings()
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": settings.kafka_group_id,
            "enable.auto.commit": False,
            "session.timeout.ms": 10000,
        }
    )
    try:
        metadata = consumer.list_topics(settings.kafka_topic, timeout=10)
        topic = metadata.topics.get(settings.kafka_topic)
        if topic is None or topic.error is not None:
            detail = topic.error if topic is not None else "topic missing"
            raise KafkaException(detail)
        partitions = [TopicPartition(settings.kafka_topic, number) for number in topic.partitions]
        committed = consumer.committed(partitions, timeout=10)
        for partition in committed:
            low, high = consumer.get_watermark_offsets(
                TopicPartition(settings.kafka_topic, partition.partition), timeout=10
            )
            current = partition.offset if partition.offset >= 0 else low
            if high > current:
                return True
        return False
    finally:
        consumer.close()


def process_station_batch() -> dict[str, Any]:
    """Run one bounded LangGraph batch for an Airflow task."""
    settings = Settings()
    return run_batch(settings, fixture=settings.fixture_mode)
