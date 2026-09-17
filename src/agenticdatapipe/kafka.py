"""Bounded consumption; a pending manifest freezes batch membership across retries."""

import json
import time
from pathlib import Path
from typing import Any

from confluent_kafka import Consumer, TopicPartition

from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import Batch
from agenticdatapipe.ingestion import digest
from agenticdatapipe.storage import atomic_json


class KafkaBatchSource:
    """One active process per consumer group (local prototype, not a distributed lock)."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": settings.kafka_group_id,
                "enable.auto.commit": False,
                "enable.auto.offset.store": False,
                "auto.offset.reset": "earliest",
                "max.poll.interval.ms": 900000,
            }
        )
        identity = digest(
            [settings.kafka_bootstrap_servers, settings.kafka_topic, settings.kafka_group_id]
        )[:16]
        self.pending: Path = settings.data_dir / "pending" / f"{identity}.json"
        self.consumer.subscribe([settings.kafka_topic])

    def read(self) -> Batch:
        if self.pending.exists():
            return Batch.model_validate_json(self.pending.read_text(encoding="utf-8"))
        records: list[dict[str, Any]] = []
        offsets: dict[str, dict[str, int]] = {}
        deadline = time.monotonic() + self.settings.batch_timeout_seconds
        while len(records) < self.settings.batch_size and time.monotonic() < deadline:
            message = self.consumer.poll(min(1, max(0, deadline - time.monotonic())))
            if message is None:
                continue
            if message.error():
                raise RuntimeError(str(message.error()))
            try:
                record = json.loads(message.value())
                if not isinstance(record, dict):
                    raise TypeError("Event must be a JSON object")
            except (ValueError, TypeError, UnicodeDecodeError) as exc:
                record = {
                    "raw": (message.value() or b"").decode(errors="replace"),
                    "decode_error": str(exc),
                }
            records.append(record)
            partition = str(message.partition())
            offsets.setdefault(partition, {"start": message.offset(), "end": message.offset()})
            offsets[partition]["end"] = message.offset()
        batch_id = digest([self.settings.kafka_topic, self.settings.kafka_group_id, offsets])[:24]
        batch = Batch(batch_id=batch_id, records=records, offsets=offsets)
        if records:
            atomic_json(self.pending, batch.model_dump())
        return batch

    def commit(self, batch: Batch) -> None:
        offsets = [
            TopicPartition(self.settings.kafka_topic, int(p), value["end"] + 1)
            for p, value in batch.offsets.items()
        ]
        if offsets:
            results = self.consumer.commit(offsets=offsets, asynchronous=False)
            for result in results or []:
                if result.error:
                    raise RuntimeError(str(result.error))
        self.pending.unlink(missing_ok=True)

    def close(self) -> None:
        self.consumer.close()
