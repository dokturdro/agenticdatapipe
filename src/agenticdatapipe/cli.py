"""Command-line entry points for live ingestion and reproducible replay."""

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv

from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import Batch
from agenticdatapipe.graph import build_graph
from agenticdatapipe.ingestion import (
    EventProducer,
    GBFSClient,
    digest,
    ensure_topic,
    record_snapshot,
    snapshot_events,
)
from agenticdatapipe.kafka import KafkaBatchSource
from agenticdatapipe.storage import inspect_delta, verify_object_store


def run_batch(settings: Settings, fixture: bool = False) -> dict:
    source = KafkaBatchSource(settings)
    try:
        batch = source.read()
        if not batch.records:
            return {"status": "empty"}
        marker = settings.data_dir / "batches" / batch.batch_id / "report.json"
        if marker.exists():
            report = json.loads(marker.read_text(encoding="utf-8"))
        else:
            report = build_graph(settings, fixture).invoke({"batch": batch.model_dump()})["report"]
        source.commit(batch)
        return report
    finally:
        source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Citi Bike streaming data pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    producer = commands.add_parser("produce")
    producer.add_argument("--once", action="store_true")
    replay = commands.add_parser("replay")
    replay.add_argument("snapshot", type=Path)
    commands.add_parser("inspect-lake")
    commands.add_parser("verify-storage")
    batch = commands.add_parser("run-batch")
    batch.add_argument(
        "--fixture",
        action="store_true",
        help="Use deterministic offline planning; still consumes Kafka",
    )
    demo = commands.add_parser("demo", help="Run fixture graph without Kafka or API calls")
    demo.add_argument("--snapshot", type=Path, default=Path("fixtures/stations.json"))
    args = parser.parse_args()
    load_dotenv()
    settings = Settings()
    try:
        if args.command == "inspect-lake":
            print(json.dumps(inspect_delta(settings), indent=2))
        elif args.command == "verify-storage":
            print(json.dumps(verify_object_store(settings), indent=2))
        elif args.command == "run-batch":
            print(json.dumps(run_batch(settings, args.fixture), indent=2))
        elif args.command == "demo":
            events = snapshot_events(json.loads(args.snapshot.read_text(encoding="utf-8-sig")))
            batch_value = Batch(batch_id="fixture-" + digest(events)[:16], records=events)
            print(
                json.dumps(
                    build_graph(settings, True).invoke({"batch": batch_value.model_dump()})[
                        "report"
                    ],
                    indent=2,
                )
            )
        elif args.command == "replay":
            ensure_topic(settings)
            events = snapshot_events(json.loads(args.snapshot.read_text(encoding="utf-8-sig")))
            print(f"Published {EventProducer(settings).publish(events)} events")
        else:
            ensure_topic(settings)
            client, publisher = GBFSClient(settings), EventProducer(settings)
            try:
                while True:
                    snapshot = client.snapshot()
                    record_snapshot(snapshot, settings.data_dir / "snapshots", settings)
                    print(
                        f"Published {publisher.publish(snapshot_events(snapshot))} events",
                        flush=True,
                    )
                    if args.once:
                        break
                    time.sleep(
                        max(settings.poll_seconds, float(snapshot["station_status"].get("ttl", 60)))
                    )
            finally:
                client.close()
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001 -- CLI boundary reports failures with a nonzero exit.
        parser.exit(1, f"Pipeline failed ({type(exc).__name__}): {exc}\n")


if __name__ == "__main__":
    main()
