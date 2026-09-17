"""GBFS snapshots, recording, and acknowledged Kafka publishing."""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from agenticdatapipe.config import Settings


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class GBFSClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.Client(timeout=settings.http_timeout_seconds, follow_redirects=True)
        self.feeds: dict[str, str] = {}
        self.metadata: dict[str, Any] = {}
        self.metadata_at = 0.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=8),
        retry=retry_if_exception_type(httpx.HTTPError),
        reraise=True,
    )
    def get(self, url: str) -> dict[str, Any]:
        response = self.client.get(url)
        response.raise_for_status()
        payload = response.json()
        if payload.get("version") != "2.3":
            raise ValueError("Stage one supports GBFS 2.3 only")
        return payload

    def snapshot(self) -> dict[str, Any]:
        if not self.feeds:
            discovery = self.get(self.settings.feed_url)
            self.feeds = {f["name"]: f["url"] for f in discovery["data"]["en"]["feeds"]}
        if (
            not self.metadata
            or time.monotonic() - self.metadata_at >= self.settings.metadata_refresh_seconds
        ):
            self.metadata = self.get(self.feeds["station_information"])
            self.metadata_at = time.monotonic()
        status = self.get(self.feeds["station_status"])
        return {
            "system_id": self.settings.system_id,
            "ingested_at": int(time.time()),
            "station_information": self.metadata,
            "station_status": status,
        }

    def close(self) -> None:
        self.client.close()


def snapshot_events(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve bad source values for the guardian rather than silently dropping them."""
    metadata = {
        str(s["station_id"]): s for s in snapshot["station_information"]["data"]["stations"]
    }
    status = snapshot["station_status"]
    events = []
    for station in status["data"]["stations"]:
        station_id = str(station.get("station_id", ""))
        event = {
            "schema_version": 1,
            "system_id": snapshot["system_id"],
            "station_id": station_id,
            "source_updated_at": status["last_updated"],
            "ingested_at": snapshot["ingested_at"],
            "status": station,
            "metadata": metadata.get(station_id, {}),
        }
        # Stable across retries/replay; ingestion time is intentionally excluded.
        event["event_id"] = digest({k: v for k, v in event.items() if k != "ingested_at"})
        events.append(event)
    return events


def record_snapshot(
    snapshot: dict[str, Any], directory: Path, settings: Settings | None = None
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{snapshot['ingested_at']}-{digest(snapshot)[:16]}.json"
    target.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    if settings is not None:
        from agenticdatapipe.storage import upload_snapshot

        upload_snapshot(settings, snapshot)
    return target


def ensure_topic(settings: Settings) -> None:
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    futures = admin.create_topics([NewTopic(settings.kafka_topic, 1, 1)], request_timeout=15)
    for future in futures.values():
        try:
            future.result(timeout=20)
        except Exception as exc:
            from confluent_kafka import KafkaError, KafkaException

            if (
                not isinstance(exc, KafkaException)
                or exc.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS
            ):
                raise


class EventProducer:
    def __init__(self, settings: Settings) -> None:
        self.topic = settings.kafka_topic
        self.producer = Producer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "enable.idempotence": True,
                "acks": "all",
                "delivery.timeout.ms": 30000,
            }
        )

    def publish(self, events: list[dict[str, Any]]) -> int:
        errors: list[str] = []

        def delivered(error: Any, message: Any) -> None:
            if error:
                errors.append(str(error))

        for event in events:
            self.producer.produce(
                self.topic,
                key=event["station_id"],
                value=json.dumps(event).encode(),
                on_delivery=delivered,
            )
            self.producer.poll(0)
        remaining = self.producer.flush(35)
        if errors or remaining:
            raise RuntimeError(f"Kafka delivery failed: {errors}; undelivered={remaining}")
        return len(events)
