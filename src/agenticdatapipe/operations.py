"""Deterministic data operations exposed to the graph as LangChain tools."""

from datetime import UTC, datetime
from typing import Any

import duckdb
import pandas as pd
from langchain_core.tools import tool
from pydantic import ValidationError

from agenticdatapipe.contracts import REQUIRED_OPERATIONS, Observation, Profile, TransformPlan


@tool
def profile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Profile station events using DuckDB without executing generated SQL."""
    frame = pd.DataFrame(
        [{"station_id": r.get("station_id"), "ts": r.get("source_updated_at")} for r in records]
    )
    if frame.empty:
        return Profile(rows=0, unique_stations=0, missing_fields={}).model_dump()
    with duckdb.connect() as connection:
        connection.register("events", frame)
        count = connection.execute("SELECT count(DISTINCT station_id) FROM events").fetchone()[0]
    timestamps = [
        r["source_updated_at"] for r in records if isinstance(r.get("source_updated_at"), int)
    ]
    return Profile(
        rows=len(records),
        unique_stations=count,
        missing_fields={
            k: sum(r.get(k) is None for r in records) for k in ("station_id", "status", "metadata")
        },
        timestamp_min=min(timestamps, default=None),
        timestamp_max=max(timestamps, default=None),
    ).model_dump()


@tool
def execute_plan(
    records: list[dict[str, Any]], plan: dict[str, Any], stale_seconds: int
) -> dict[str, Any]:
    """Execute allowlisted operations; preserve rejected input in quarantine."""
    parsed = TransformPlan.model_validate(plan)
    missing = set(REQUIRED_OPERATIONS) - set(parsed.operations)
    if missing:
        return {
            "accepted": [],
            "features": [],
            "quarantine": [],
            "duplicates": 0,
            "issues": ["Missing required operations: " + ", ".join(sorted(missing))],
        }
    accepted, features, quarantine = [], [], []
    seen: set[str] = set()
    duplicates = 0
    for original in records:
        try:
            status = original.get("status", {})
            metadata = original.get("metadata", {})
            row = {
                **original,
                **{
                    k: status.get(k)
                    for k in (
                        "last_reported",
                        "num_bikes_available",
                        "num_docks_available",
                        "is_installed",
                        "is_renting",
                        "is_returning",
                    )
                },
                "capacity": metadata.get("capacity"),
                "station_name": metadata.get("name"),
                "lat": metadata.get("lat"),
                "lon": metadata.get("lon"),
            }
            observation = Observation.model_validate(row)
            if not all(
                (observation.is_installed, observation.is_renting, observation.is_returning)
            ):
                raise ValueError("Station is not operational")
            if observation.source_updated_at - observation.last_reported > stale_seconds:
                raise ValueError("Stale station observation")
            if observation.last_reported > observation.source_updated_at + 60:
                raise ValueError("Station timestamp is in the future")
            if observation.event_id in seen:
                duplicates += 1
                continue
            seen.add(observation.event_id)
            accepted.append(observation.model_dump())
            timestamp = datetime.fromtimestamp(observation.last_reported, UTC)
            features.append(
                {
                    "event_id": observation.event_id,
                    "system_id": observation.system_id,
                    "station_id": observation.station_id,
                    "event_timestamp": observation.last_reported,
                    "hour_utc": timestamp.hour,
                    "day_of_week": timestamp.weekday(),
                    "available_bikes": observation.num_bikes_available,
                    "available_docks": observation.num_docks_available,
                    "capacity": observation.capacity,
                    "availability_ratio": observation.num_bikes_available / observation.capacity,
                }
            )
        except (
            ValidationError,
            ValueError,
            TypeError,
            AttributeError,
            OverflowError,
            OSError,
        ) as exc:
            quarantine.append({"record": original, "reason": str(exc)})
    return {
        "accepted": accepted,
        "features": features,
        "quarantine": quarantine,
        "duplicates": duplicates,
        "issues": [],
    }
