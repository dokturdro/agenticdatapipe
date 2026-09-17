"""Recoverable local artifacts and optional MinIO-backed Delta Lake storage."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import pyarrow as pa
from botocore.client import BaseClient
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import TableNotFoundError

from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import Batch, Observation


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_parquet(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    temporary = path.with_suffix(".parquet.tmp")
    pd.DataFrame(rows, columns=columns).to_parquet(temporary, index=False)
    with temporary.open("rb+") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def s3_client(settings: Settings) -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
    )


def verify_object_store(settings: Settings) -> dict[str, Any]:
    """Check the configured bucket and return a secret-free status payload."""
    s3_client(settings).head_bucket(Bucket=settings.s3_bucket)
    return {"endpoint": settings.s3_endpoint, "bucket": settings.s3_bucket, "status": "ok"}


def put_json(settings: Settings, key: str, value: Any) -> str:
    payload = json.dumps(value, indent=2, allow_nan=False).encode()
    s3_client(settings).put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=payload,
        ContentType="application/json",
    )
    return settings.s3_uri(key)


def upload_snapshot(settings: Settings, snapshot: dict[str, Any]) -> str | None:
    """Upload a raw source snapshot when the Delta backend is enabled."""
    if settings.storage_backend != "delta":
        return None
    ingested_at = int(snapshot["ingested_at"])
    date = datetime.fromtimestamp(ingested_at, UTC).date().isoformat()
    from agenticdatapipe.ingestion import digest

    key = (
        f"{settings.raw_prefix.rstrip('/')}/system_id={settings.system_id}/"
        f"event_date={date}/{ingested_at}-{digest(snapshot)[:16]}.json"
    )
    return put_json(settings, key, snapshot)


def _enrich_rows(
    rows: list[dict[str, Any]], batch_id: str, timestamp_field: str
) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        timestamp = int(row[timestamp_field])
        enriched.append(
            {
                **row,
                "batch_id": batch_id,
                "event_date": datetime.fromtimestamp(timestamp, UTC).date().isoformat(),
            }
        )
    return enriched


def delta_upsert(
    table_uri: str,
    rows: list[dict[str, Any]],
    storage_options: dict[str, str] | None = None,
) -> int | None:
    """Create or idempotently upsert a Delta table by event_id."""
    if not rows:
        try:
            return DeltaTable(table_uri, storage_options=storage_options).version()
        except TableNotFoundError:
            return None

    source = pa.Table.from_pylist(rows)
    try:
        table = DeltaTable(table_uri, storage_options=storage_options)
    except TableNotFoundError:
        write_deltalake(table_uri, source, mode="error", storage_options=storage_options)
    else:
        (
            table.merge(
                source=source,
                predicate="target.event_id = source.event_id",
                source_alias="source",
                target_alias="target",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
    return DeltaTable(table_uri, storage_options=storage_options).version()


def inspect_delta(settings: Settings) -> dict[str, Any]:
    """Return table versions and row counts."""
    output: dict[str, Any] = {"backend": settings.storage_backend, "tables": {}}
    options = settings.delta_storage_options() if settings.storage_backend == "delta" else None
    for name, location in (
        ("observations", settings.observations_table),
        ("features", settings.features_table),
    ):
        uri = settings.s3_uri(location) if settings.storage_backend == "delta" else location
        try:
            table = DeltaTable(uri, storage_options=options)
            output["tables"][name] = {
                "uri": uri,
                "version": table.version(),
                "rows": table.to_pyarrow_dataset().count_rows(),
            }
        except TableNotFoundError:
            output["tables"][name] = {"uri": uri, "status": "missing"}
    return output


def _persist_delta(settings: Settings, batch: Batch, result: dict[str, Any]) -> dict[str, Any]:
    options = settings.delta_storage_options()
    observations_uri = settings.s3_uri(settings.observations_table)
    features_uri = settings.s3_uri(settings.features_table)
    observation_rows = _enrich_rows(result["accepted"], batch.batch_id, "last_reported")
    feature_rows = _enrich_rows(result["features"], batch.batch_id, "event_timestamp")
    observation_version = delta_upsert(observations_uri, observation_rows, options)
    feature_version = delta_upsert(features_uri, feature_rows, options)
    quarantine_key = f"{settings.audit_prefix.rstrip('/')}/{batch.batch_id}/quarantine.json"
    return {
        "backend": "delta",
        "observations": {"uri": observations_uri, "version": observation_version},
        "features": {"uri": features_uri, "version": feature_version},
        "quarantine_uri": put_json(settings, quarantine_key, result["quarantine"]),
    }


def persist_batch(settings: Settings, batch: Batch, result: dict[str, Any]) -> dict[str, Any]:
    """Persist every output; report.json is the final local completion marker."""
    directory = settings.data_dir / "batches" / batch.batch_id
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "report.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))

    atomic_parquet(
        directory / "observations.parquet", result["accepted"], list(Observation.model_fields)
    )
    atomic_parquet(
        directory / "features.parquet",
        result["features"],
        [
            "event_id",
            "system_id",
            "station_id",
            "event_timestamp",
            "hour_utc",
            "day_of_week",
            "available_bikes",
            "available_docks",
            "capacity",
            "availability_ratio",
        ],
    )
    atomic_json(directory / "quarantine.json", result["quarantine"])

    storage = (
        _persist_delta(settings, batch, result)
        if settings.storage_backend == "delta"
        else {"backend": "local"}
    )
    report = {
        "batch_id": batch.batch_id,
        "offsets": batch.offsets,
        "quality": result["quality"],
        "profile": result["profile"],
        "plan": result["plan"],
        "attempts": result["attempts"],
        "mode": result["mode"],
        "storage": storage,
    }
    if settings.storage_backend == "delta":
        report_key = f"{settings.audit_prefix.rstrip('/')}/{batch.batch_id}/report.json"
        report["storage"]["report_uri"] = settings.s3_uri(report_key)
        put_json(settings, report_key, report)

    # Kafka offsets may advance only after this marker exists.
    atomic_json(marker, report)
    return report
