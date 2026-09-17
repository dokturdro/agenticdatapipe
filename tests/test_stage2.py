import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from confluent_kafka import TopicPartition
from deltalake import DeltaTable

from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import Batch
from agenticdatapipe.storage import delta_upsert, inspect_delta, persist_batch, upload_snapshot


def test_delta_upsert_is_idempotent_and_updates(tmp_path):
    table_uri = str(tmp_path / "events")
    first = {"event_id": "one", "value": 1}
    assert delta_upsert(table_uri, [first]) == 0
    delta_upsert(table_uri, [first])
    delta_upsert(table_uri, [{"event_id": "one", "value": 2}])
    data = DeltaTable(table_uri).to_pandas()
    assert len(data) == 1
    assert data.iloc[0]["value"] == 2


def test_inspect_local_delta(tmp_path):
    observations = tmp_path / "observations"
    features = tmp_path / "features"
    delta_upsert(str(observations), [{"event_id": "one", "value": 1}])
    settings = Settings(
        storage_backend="local",
        observations_table=str(observations),
        features_table=str(features),
    )
    result = inspect_delta(settings)
    assert result["tables"]["observations"]["rows"] == 1
    assert result["tables"]["features"]["status"] == "missing"


def test_delta_failure_does_not_create_completion_marker(monkeypatch, tmp_path):
    from agenticdatapipe import storage

    monkeypatch.setattr(storage, "_persist_delta", Mock(side_effect=OSError("MinIO unavailable")))
    settings = Settings(storage_backend="delta", data_dir=tmp_path)
    result = {
        "accepted": [],
        "features": [],
        "quarantine": [],
        "quality": {},
        "profile": {},
        "plan": {},
        "attempts": 1,
        "mode": "fixture",
    }
    batch = Batch(batch_id="failed-delta", records=[])
    try:
        persist_batch(settings, batch, result)
    except OSError:
        pass
    else:
        raise AssertionError("Expected the storage error to propagate")
    assert not (tmp_path / "batches/failed-delta/report.json").exists()


def test_raw_snapshot_uses_deterministic_partitioned_key(monkeypatch):
    from agenticdatapipe import storage

    put = Mock(return_value="s3://bike-lake/raw.json")
    monkeypatch.setattr(storage, "put_json", put)
    settings = Settings(storage_backend="delta")
    snapshot = {"ingested_at": 1_788_802_920, "value": 3}
    assert upload_snapshot(settings, snapshot) == "s3://bike-lake/raw.json"
    key = put.call_args.args[1]
    assert key.startswith("raw/gbfs/system_id=citibike-nyc/event_date=")
    assert key.endswith(".json")


def test_kafka_backlog_sensor_does_not_consume(monkeypatch):
    from agenticdatapipe import airflow_tasks

    consumer = Mock()
    consumer.list_topics.return_value = SimpleNamespace(
        topics={"bike.station_status.v1": SimpleNamespace(error=None, partitions={0: object()})}
    )
    consumer.committed.return_value = [TopicPartition("bike.station_status.v1", 0, 5)]
    consumer.get_watermark_offsets.return_value = (0, 6)
    monkeypatch.setattr(airflow_tasks, "Consumer", Mock(return_value=consumer))
    assert airflow_tasks.kafka_has_backlog()
    consumer.poll.assert_not_called()
    consumer.commit.assert_not_called()
    consumer.close.assert_called_once()


def test_airflow_dag_declares_sensor_before_graph():
    source = Path("airflow/dags/bike_station_pipeline.py").read_text(encoding="utf-8")
    assert 'dag_id="bike_station_pipeline"' in source
    assert "mode=\"reschedule\"" in source
    assert "max_active_runs=1" in source
    assert "wait_for_events >> run_graph()" in source


@pytest.mark.integration
@pytest.mark.minio
@pytest.mark.skipif(
    os.getenv("BIKE_TEST_STACK") != "1",
    reason="Set BIKE_TEST_STACK=1 with Kafka and MinIO running",
)
def test_kafka_to_minio_delta_stack(tmp_path):
    from agenticdatapipe.cli import run_batch
    from agenticdatapipe.ingestion import EventProducer, ensure_topic, snapshot_events

    suffix = uuid.uuid4().hex
    settings = Settings(
        data_dir=tmp_path,
        storage_backend="delta",
        kafka_topic=f"bike-stack-{suffix}",
        kafka_group_id=f"bike-stack-{suffix}",
        observations_table=f"tests/{suffix}/observations",
        features_table=f"tests/{suffix}/features",
        audit_prefix=f"tests/{suffix}/audit",
        batch_size=2,
    )
    snapshot = json.loads(Path("fixtures/stations.json").read_text(encoding="utf-8-sig"))
    ensure_topic(settings)
    events = snapshot_events(snapshot)
    EventProducer(settings).publish(events)
    report = run_batch(settings, fixture=True)
    EventProducer(settings).publish(events)
    retry_report = run_batch(settings, fixture=True)
    lake = inspect_delta(settings)
    assert report["storage"]["backend"] == "delta"
    assert retry_report["storage"]["backend"] == "delta"
    assert lake["tables"]["observations"]["rows"] == 1
    assert lake["tables"]["features"]["rows"] == 1
