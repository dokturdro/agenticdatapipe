"""Run one complete local Kafka -> Airflow -> LangGraph -> Delta pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

from agenticdatapipe.config import Settings
from agenticdatapipe.ingestion import EventProducer, ensure_topic, snapshot_events

DAG_ID = "bike_station_pipeline"


def compose_command(*arguments: str, environment: dict[str, str], capture: bool = False) -> str:
    """Run Docker Compose and turn command failures into readable launcher errors."""
    try:
        result = subprocess.run(
            ["docker", "compose", *arguments],
            check=True,
            env=environment,
            text=True,
            capture_output=capture,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker was not found. Install and start Docker Desktop.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"Docker Compose command failed: {detail}") from exc
    return result.stdout.strip() if capture else ""


def wait_for_airflow(timeout: int) -> None:
    """Wait until the API, scheduler, database, and DAG processor are healthy."""
    deadline = time.monotonic() + timeout
    url = "http://localhost:8080/api/v2/monitor/health"
    last_error = "Airflow has not responded"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=5)
            response.raise_for_status()
            health = response.json()
            required = ("metadatabase", "scheduler", "dag_processor")
            if all(health.get(component, {}).get("status") == "healthy" for component in required):
                return
            last_error = json.dumps(health)
        except (httpx.HTTPError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Airflow did not become healthy within {timeout}s: {last_error}")


def dag_state(run_id: str, environment: dict[str, str]) -> str:
    output = compose_command(
        "exec",
        "-T",
        "airflow-scheduler",
        "airflow",
        "dags",
        "state",
        DAG_ID,
        run_id,
        environment=environment,
        capture=True,
    )
    return output.splitlines()[-1].strip().lower()


def wait_for_result(
    run_id: str,
    existing_reports: set[Path],
    timeout: int,
    environment: dict[str, str],
) -> Path:
    """Wait for a successful DAG run and its final local completion marker."""
    deadline = time.monotonic() + timeout
    reports_root = Path("data/batches")
    last_state = "queued"
    while time.monotonic() < deadline:
        last_state = dag_state(run_id, environment)
        current_reports = set(reports_root.glob("*/report.json"))
        new_reports = sorted(current_reports - existing_reports, key=lambda path: path.stat().st_mtime)
        if last_state == "failed":
            raise RuntimeError(f"Airflow DAG run {run_id} failed; inspect http://localhost:8080")
        if last_state == "success" and new_reports:
            return new_reports[-1]
        time.sleep(2)
    raise TimeoutError(
        f"Airflow DAG run {run_id} did not produce a report within {timeout}s "
        f"(last state: {last_state})"
    )


def run(snapshot_path: Path, timeout: int, fixture: bool) -> dict:
    load_dotenv()
    if not fixture and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required. Add it to .env or use --fixture.")
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"Snapshot does not exist: {snapshot_path}")

    suffix = uuid.uuid4().hex[:12]
    topic = f"bike.station_status.run.{suffix}"
    group_id = f"bike-pipeline-run-{suffix}"
    environment = os.environ.copy()
    environment.update(
        {
            "BIKE_KAFKA_TOPIC": topic,
            "BIKE_KAFKA_GROUP_ID": group_id,
            "BIKE_FIXTURE_MODE": str(fixture).lower(),
        }
    )

    print("[1/5] Starting Kafka, MinIO, PostgreSQL, and Airflow...", file=sys.stderr)
    compose_command("up", "--build", "-d", environment=environment)
    wait_for_airflow(timeout)

    reports_root = Path("data/batches")
    existing_reports = set(reports_root.glob("*/report.json"))
    print(f"[2/5] Publishing {snapshot_path} to Kafka topic {topic}...", file=sys.stderr)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
    events = snapshot_events(snapshot)
    producer_settings = Settings(
        kafka_bootstrap_servers="localhost:9092",
        kafka_topic=topic,
        kafka_group_id=group_id,
    )
    ensure_topic(producer_settings)
    EventProducer(producer_settings).publish(events)

    run_id = f"main-{suffix}"
    print(f"[3/5] Triggering Airflow DAG run {run_id}...", file=sys.stderr)
    compose_command(
        "exec",
        "-T",
        "airflow-scheduler",
        "airflow",
        "dags",
        "trigger",
        DAG_ID,
        "--run-id",
        run_id,
        environment=environment,
        capture=True,
    )

    print("[4/5] Waiting for LangGraph and Delta persistence...", file=sys.stderr)
    report_path = wait_for_result(run_id, existing_reports, timeout, environment)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"[5/5] Complete: {report_path}", file=sys.stderr)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("fixtures/stations.json"),
        help="GBFS snapshot to publish (default: fixtures/stations.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Seconds allowed for service startup and DAG execution (default: 300)",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Use the deterministic planner instead of calling OpenAI",
    )
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    try:
        report = run(args.snapshot, args.timeout, args.fixture)
    except (OSError, RuntimeError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Pipeline failed ({type(exc).__name__}): {exc}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
