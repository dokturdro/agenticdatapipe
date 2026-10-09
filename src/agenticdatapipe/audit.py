from __future__ import annotations

import json
from typing import Any, Protocol

from confluent_kafka import Producer

from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import Batch, PredictionAudit
from agenticdatapipe.ingestion import ensure_topic
from agenticdatapipe.kafka import KafkaBatchSource
from agenticdatapipe.storage import atomic_json, delta_upsert, delta_uri


class AuditPublisher(Protocol):
    failed_events: int

    def publish(self, event: PredictionAudit) -> None: ...

    def close(self) -> None: ...


class PredictionAuditPublisher:
    def __init__(self, settings: Settings) -> None:
        self.topic = settings.prediction_audit_topic
        self.failed_events = 0
        ensure_topic(settings.model_copy(update={"kafka_topic": self.topic}))
        self.producer = Producer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "enable.idempotence": True,
                "acks": "all",
                "delivery.timeout.ms": 30000,
            }
        )

    def _delivered(self, error: Any, message: Any) -> None:
        if error is not None:
            self.failed_events += 1

    def publish(self, event: PredictionAudit) -> None:
        try:
            self.producer.produce(
                self.topic,
                key=event.station_id,
                value=event.model_dump_json().encode(),
                on_delivery=self._delivered,
            )
            self.producer.poll(0)
        except Exception:
            self.failed_events += 1
            raise

    def close(self) -> None:
        remaining = self.producer.flush(5)
        self.failed_events += remaining


def persist_prediction_audit_batch(settings: Settings, batch: Batch) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    for record in batch.records:
        try:
            valid.append(PredictionAudit.model_validate(record).model_dump(mode="python"))
        except Exception as exc:  # noqa: BLE001
            invalid.append({"record": record, "error": f"{type(exc).__name__}: {exc}"})

    uri = delta_uri(settings, settings.prediction_audits_table)
    options = settings.delta_storage_options() if settings.storage_backend == "delta" else None
    version = delta_upsert(uri, valid, options)
    report = {
        "batch_id": batch.batch_id,
        "accepted": len(valid),
        "quarantined": len(invalid),
        "table_uri": uri,
        "table_version": version,
    }
    directory = settings.data_dir / "prediction_audits" / batch.batch_id
    atomic_json(directory / "quarantine.json", invalid)
    atomic_json(directory / "report.json", report)
    return report


def process_prediction_audit_batch(settings: Settings) -> dict[str, Any]:
    source = KafkaBatchSource(
        settings,
        topic=settings.prediction_audit_topic,
        group_id=settings.prediction_audit_group_id,
    )
    try:
        batch = source.read()
        if not batch.records:
            return {"status": "empty"}
        marker = settings.data_dir / "prediction_audits" / batch.batch_id / "report.json"
        report = (
            json.loads(marker.read_text(encoding="utf-8"))
            if marker.exists()
            else persist_prediction_audit_batch(settings, batch)
        )
        source.commit(batch)
        return report
    finally:
        source.close()
