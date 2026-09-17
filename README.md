# Citi Bike agentic data and MLOps pipeline

This repository is a local-first, incremental MLOps example. The current two stages stream
Citi Bike GBFS 2.3 observations through Kafka, run a stateful LangGraph pipeline under
Airflow, and store validated observations and features as Delta Lake tables in MinIO.
OpenAI produces structured transformation plans from the current batch profile and validation results.

```mermaid
flowchart LR
    GBFS[Citi Bike GBFS] --> Producer[Live producer]
    Producer --> Kafka[Kafka]
    Kafka --> Sensor[Airflow backlog sensor]
    Sensor --> Graph[LangGraph task]
    OpenAI --> Graph
    Graph -->|repair up to twice| Graph
    Graph --> Delta[(Delta Lake on MinIO)]
    Graph --> Audit[Reports and quarantine]
```

The eventual model will predict available bikes 15 minutes ahead. Training, Feast, MLflow,
FastAPI serving, and statistical drift monitoring are later stages.

## Components

- Kafka carries `bike.station_status.v1` events keyed by station ID.
- Airflow schedules one bounded batch each minute. Its rescheduling sensor checks Kafka
  watermarks and committed offsets without consuming messages.
- LangGraph executes scout, profiler, planner, executor, validator, and persist
  nodes. Invalid plans loop back to the planner at most twice.
- Deterministic tools perform transformations; the model cannot execute generated code.
- MinIO stores raw snapshots, Delta tables, quarantine records, and run reports.
- PostgreSQL stores Airflow metadata. `LocalExecutor` keeps the local stack small.

## Local Python setup

Python 3.11 or newer and `uv` are required:

```powershell
uv sync --locked
Copy-Item .env.example .env
```

Put `OPENAI_API_KEY` in `.env`. The default model is `gpt-4.1-mini` and can be changed
with `BIKE_OPENAI_MODEL`. Do not commit `.env`.

The offline fixture path remains available and requires neither Docker nor OpenAI:

```powershell
uv run bikepipe demo
uv run pytest -q
```

It produces one valid station and quarantines one inactive station under `data/batches`.

## Run the complete pipeline with one command

With Docker Desktop running and `OPENAI_API_KEY` set in `.env`, execute the complete local
Kafka, Airflow, LangGraph, OpenAI, MinIO, and Delta path:

```powershell
uv run python main.py
```

The launcher starts the Compose services, creates an isolated Kafka topic and consumer group,
publishes `fixtures/stations.json`, triggers the Airflow DAG, waits for completion, and prints
the final report. It leaves the services running so their state can be inspected afterward.

To exercise the same infrastructure without an OpenAI call:

```powershell
uv run python main.py --fixture
```

Use `--snapshot <path>` for another GBFS snapshot and `--timeout <seconds>` to change the
startup and DAG execution timeout. The lower-level `bikepipe` commands remain available for
running individual stages.

## Start Kafka, MinIO, and Airflow

Start Docker Desktop with Linux containers, then build and start the base services:

```powershell
docker compose up --build -d
docker compose ps
```

Airflow is at http://localhost:8080 and the MinIO console is at http://localhost:9001.
The Airflow user is `admin`; its generated password is stored in:

```powershell
Get-Content data/airflow/simple_auth_manager_passwords.json.generated
```

The default MinIO login is `minioadmin` / `minioadmin`. Change the corresponding values in
`.env` if the stack is reachable outside your machine. Compose creates the `bike-lake`
bucket automatically.

The `bike_station_pipeline` DAG is enabled on a one-minute schedule. With no Kafka backlog,
the sensor reschedules and the run ends as skipped without calling OpenAI.

## Send data through the pipeline

For a deterministic two-event Kafka replay from the host:

```powershell
uv run bikepipe replay fixtures/stations.json
```

Airflow detects the backlog and executes the LangGraph task with OpenAI planning. To stream
the live Citi Bike feed continuously, start the optional producer profile:

```powershell
docker compose --profile live up --build -d bike-producer
docker compose logs -f bike-producer
```

The producer respects the larger of the configured polling interval and the GBFS feed TTL,
records each raw snapshot locally, uploads it to MinIO, and publishes acknowledged Kafka
events. Stop only the live producer with:

```powershell
docker compose --profile live stop bike-producer
```

Inspect Airflow task logs in the UI or with `docker compose logs airflow-scheduler`.

## Inspect MinIO and Delta Lake

The host CLI defaults to local storage so `demo` stays self-contained. Temporarily select
the Delta backend when inspecting the running stack:

```powershell
$env:BIKE_STORAGE_BACKEND = 'delta'
uv run bikepipe verify-storage
uv run bikepipe inspect-lake
Remove-Item Env:BIKE_STORAGE_BACKEND
```

MinIO contains these logical locations:

```text
s3://bike-lake/raw/gbfs/                         raw source snapshots
s3://bike-lake/delta/station_observations       validated Delta table
s3://bike-lake/delta/station_features           feature Delta table
s3://bike-lake/audit/runs/<batch-id>/            quarantine and report JSON
```

Each local `data/batches/<batch-id>` directory still contains `observations.parquet`,
`features.parquet`, `quarantine.json`, and `report.json`. Reports include Kafka offset ranges,
quality results, the selected plan, Delta versions, and object URIs.

## Reliability model

Processing is at-least-once. A pending manifest freezes each batch's partition/offset range.
Delta tables upsert by stable `event_id`, so task retries do not duplicate rows. Raw and audit
objects use deterministic keys. The local completion marker is written only after all MinIO
and Delta operations succeed, and Kafka offsets are committed only after that marker exists.

Airflow limits the DAG to one active run. The MinIO Delta client also uses conditional writes.
Run only one consumer for a given group and shared `data` directory. Across-batch queries use
`event_id` as their identity; later feature jobs will use event time rather than ingestion time.

## Tests

Run fast tests and lint locally:

```powershell
uv run pytest -q
uv run ruff check .
```

With Kafka and MinIO running, execute the opt-in stack test:

```powershell
$env:BIKE_TEST_STACK = '1'
uv run pytest -q -m "integration and minio"
Remove-Item Env:BIKE_TEST_STACK
```

Validate Airflow DAG imports inside its actual image:

```powershell
docker compose exec airflow-scheduler airflow dags list-import-errors
```

## Configuration

All application settings use the `BIKE_` prefix and are listed in `.env.example`. Host
defaults use `localhost`; Compose overrides Kafka and MinIO endpoints with service names.
The Airflow image contains the application, while DAGs, logs, and the local recovery directory
are mounted from the repository.

## Shutdown and reset

Stop containers while retaining Kafka, MinIO, and PostgreSQL volumes:

```powershell
docker compose --profile live down
```

To erase this demo's container state, use `docker compose --profile live down -v`. Remove the
project's `data` directory separately only when you also intend to discard pending batch
recovery state. Do not reuse a pending manifest after resetting Kafka.

## Next stages

1. Feast online/offline feature materialization, time-based training, and MLflow registry.
2. FastAPI approved-model inference, quality/drift monitoring, and Kafka prediction events.

Citi Bike discovery feed: https://gbfs.citibikenyc.com/gbfs/2.3/gbfs.json  
GBFS reference: https://gbfs.org/documentation/reference/  
OpenAI structured outputs: https://developers.openai.com/api/docs/guides/structured-outputs
