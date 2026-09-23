import copy
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import ValidationError

from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import Batch, TransformPlan
from agenticdatapipe.graph import build_graph
from agenticdatapipe.ingestion import snapshot_events
from agenticdatapipe.operations import execute_plan, profile_records


@pytest.fixture
def records():
    return snapshot_events(
        json.loads(Path("fixtures/stations.json").read_text(encoding="utf-8-sig"))
    )


@pytest.fixture
def varied_records():
    return snapshot_events(
        json.loads(Path("fixtures/stations_variation.json").read_text(encoding="utf-8"))
    )


def plan(alias=False, grace=0):
    return TransformPlan(
        allow_bikes_available_alias=alias,
        freshness_grace_seconds=grace,
        rationale="test",
    )


def test_graph_outputs(tmp_path, records):
    settings = Settings(data_dir=tmp_path)
    graph = build_graph(settings, fixture=True)
    batch = Batch(batch_id="example", records=records + [records[0]])
    report = graph.invoke({"batch": batch.model_dump()})["report"]
    assert report["quality"]["accepted_rows"] == 1
    assert report["quality"]["quarantined_rows"] == 1
    assert report["quality"]["duplicate_rows"] == 1
    assert report["quality"]["alias_recovered_rows"] == 0
    assert report["quality"]["grace_accepted_rows"] == 0
    assert "retrieved_docs" not in report
    output = tmp_path / "batches/example"
    assert pd.read_parquet(output / "features.parquet").iloc[0].available_bikes == 12
    assert graph.invoke({"batch": batch.model_dump()})["report"] == report


@pytest.mark.parametrize("change", ["negative", "missing", "stale", "future", "metadata"])
def test_invalid_records(records, change):
    record = copy.deepcopy(records[0])
    if change == "negative":
        record["status"]["num_bikes_available"] = -1
    elif change == "missing":
        del record["status"]["num_docks_available"]
    elif change == "stale":
        record["status"]["last_reported"] -= 10000
    elif change == "future":
        record["status"]["last_reported"] += 10000
    else:
        record["metadata"] = {}
    result = execute_plan.invoke(
        {
            "records": [record],
            "plan": plan().model_dump(),
            "stale_seconds": 900,
            "profile": profile_records.invoke({"records": [record], "stale_seconds": 900}),
        }
    )
    assert not result["accepted"]
    assert result["quarantine"][0]["record"] == record


def test_unsupported_plan():
    with pytest.raises(ValidationError):
        TransformPlan(
            allow_bikes_available_alias=False,
            freshness_grace_seconds=3600,
            rationale="bad",
        )


def test_variation_fixture_exercises_both_choices(tmp_path, varied_records):
    report = build_graph(Settings(data_dir=tmp_path), fixture=True).invoke(
        {"batch": Batch(batch_id="variation", records=varied_records).model_dump()}
    )["report"]
    assert report["plan"]["allow_bikes_available_alias"] is True
    assert report["plan"]["freshness_grace_seconds"] == 900
    assert report["profile"]["alias_candidates"] == 1
    assert report["profile"]["within_freshness_grace"] == 1
    assert report["quality"]["accepted_rows"] == 2
    assert report["quality"]["quarantined_rows"] == 1
    assert report["quality"]["alias_recovered_rows"] == 1
    assert report["quality"]["grace_accepted_rows"] == 1
    rows = pd.read_parquet(tmp_path / "batches/variation/observations.parquet")
    assert rows.loc[rows.station_id == "demo-2", "num_bikes_available"].item() == 7


def test_choices_change_results(varied_records):
    profile = profile_records.invoke({"records": varied_records, "stale_seconds": 900})

    def run(alias, grace):
        return execute_plan.invoke(
            {
                "records": varied_records,
                "plan": plan(alias, grace).model_dump(),
                "stale_seconds": 900,
                "profile": profile,
            }
        )

    assert len(run(False, 0)["accepted"]) == 1
    assert len(run(True, 0)["accepted"]) == 1
    assert len(run(False, 900)["accepted"]) == 1
    assert len(run(True, 900)["accepted"]) == 2


def test_grace_never_accepts_older_or_future_rows(records):
    too_old = copy.deepcopy(records[0])
    too_old["status"]["last_reported"] -= 1801
    future = copy.deepcopy(records[0])
    future["status"]["last_reported"] += 71
    batch = [too_old, future, records[0]]
    profile = profile_records.invoke({"records": batch, "stale_seconds": 900})
    result = execute_plan.invoke(
        {
            "records": batch,
            "plan": plan(grace=900).model_dump(),
            "stale_seconds": 900,
            "profile": profile,
        }
    )
    assert result["issues"]  # No grace candidates: the planner must repair this choice.

    batch = [too_old, future, records[0], copy.deepcopy(records[0])]
    batch[-1]["status"]["last_reported"] -= 1200
    batch[-1]["event_id"] = "grace-record"
    profile = profile_records.invoke({"records": batch, "stale_seconds": 900})
    result = execute_plan.invoke(
        {
            "records": batch,
            "plan": plan(grace=900).model_dump(),
            "stale_seconds": 900,
            "profile": profile,
        }
    )
    assert len(result["accepted"]) == 2
    assert len(result["quarantine"]) == 2


def test_alias_never_overrides_canonical(records):
    record = copy.deepcopy(records[0])
    record["status"]["bikes_available"] = 99
    alias_only = copy.deepcopy(records[0])
    alias_only["event_id"] = "alias-only"
    del alias_only["status"]["num_bikes_available"]
    alias_only["status"]["bikes_available"] = 7
    profile = profile_records.invoke({"records": [record, alias_only], "stale_seconds": 900})
    result = execute_plan.invoke(
        {
            "records": [record, alias_only],
            "plan": plan(alias=True).model_dump(),
            "stale_seconds": 900,
            "profile": profile,
        }
    )
    assert result["accepted"][0]["num_bikes_available"] == 12
    assert result["accepted"][1]["num_bikes_available"] == 7
    assert result["alias_recovered"] == 1


def test_repair_budget(tmp_path, records):
    planner = Mock(return_value=plan(alias=True))
    graph = build_graph(Settings(data_dir=tmp_path), True, planner)
    with pytest.raises(RuntimeError, match="Repair budget exhausted"):
        graph.invoke({"batch": Batch(batch_id="bad", records=records).model_dump()})
    assert planner.call_count == 3
    assert not (tmp_path / "batches/bad/report.json").exists()


def test_repair_success(tmp_path, records):
    planner = Mock(
        side_effect=[
            plan(alias=True),
            plan(),
        ]
    )
    report = build_graph(Settings(data_dir=tmp_path), True, planner).invoke(
        {"batch": Batch(batch_id="repair", records=records).model_dump()}
    )["report"]
    assert report["attempts"] == 2


def test_mocked_openai_plan(monkeypatch, tmp_path, varied_records):
    from agenticdatapipe import graph

    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = plan(True, 900)
    monkeypatch.setattr(graph, "ChatOpenAI", Mock(return_value=model))
    report = graph.build_graph(Settings(data_dir=tmp_path)).invoke(
        {"batch": Batch(batch_id="openai-plan", records=varied_records).model_dump()}
    )["report"]
    assert report["mode"] == "openai"
    assert report["quality"]["alias_recovered_rows"] == 1
    assert report["quality"]["grace_accepted_rows"] == 1


def test_no_commit_after_failure(monkeypatch, tmp_path, records):
    from agenticdatapipe import cli

    source = Mock()
    source.read.return_value = Batch(batch_id="failure", records=records)
    monkeypatch.setattr(cli, "KafkaBatchSource", Mock(return_value=source))
    graph = Mock()
    graph.invoke.side_effect = OSError("disk full")
    monkeypatch.setattr(cli, "build_graph", Mock(return_value=graph))
    with pytest.raises(OSError):
        cli.run_batch(Settings(data_dir=tmp_path), True)
    source.commit.assert_not_called()
    source.close.assert_called_once()


def test_pending_recovery(monkeypatch, tmp_path, records):
    from agenticdatapipe import kafka
    from agenticdatapipe.storage import atomic_json

    consumer = Mock()
    monkeypatch.setattr(kafka, "Consumer", Mock(return_value=consumer))
    source = kafka.KafkaBatchSource(Settings(data_dir=tmp_path))
    batch = Batch(batch_id="pending", records=records, offsets={"0": {"start": 5, "end": 6}})
    atomic_json(source.pending, batch.model_dump())
    assert source.read() == batch
    consumer.poll.assert_not_called()
    consumer.commit.side_effect = RuntimeError("broker unavailable")
    with pytest.raises(RuntimeError):
        source.commit(batch)
    assert source.pending.exists()


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("BIKE_TEST_KAFKA") != "1", reason="Set BIKE_TEST_KAFKA=1 with Kafka running"
)
def test_real_kafka(tmp_path, records):
    import uuid

    from agenticdatapipe.cli import run_batch
    from agenticdatapipe.ingestion import EventProducer, ensure_topic

    settings = Settings(
        data_dir=tmp_path,
        kafka_topic="bike-test-" + uuid.uuid4().hex,
        kafka_group_id=uuid.uuid4().hex,
        batch_size=2,
    )
    ensure_topic(settings)
    EventProducer(settings).publish(records)
    assert run_batch(settings, True)["quality"]["accepted_rows"] == 1


def test_openai_failure_is_not_silently_replaced(monkeypatch, tmp_path, records):
    from agenticdatapipe import graph

    model = Mock()
    model.with_structured_output.return_value.invoke.side_effect = RuntimeError("API unavailable")
    monkeypatch.setattr(graph, "ChatOpenAI", Mock(return_value=model))
    with pytest.raises(RuntimeError, match="API unavailable"):
        graph.build_graph(Settings(data_dir=tmp_path)).invoke(
            {"batch": Batch(batch_id="api-failure", records=records).model_dump()}
        )
    assert not (tmp_path / "batches/api-failure/report.json").exists()
