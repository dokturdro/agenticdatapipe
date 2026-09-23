"""Deterministic data operations exposed to the graph as LangChain tools."""

from datetime import UTC, datetime
from typing import Any

import duckdb
import pandas as pd
from langchain_core.tools import tool
from pydantic import ValidationError

from agenticdatapipe.contracts import Observation, Profile, TransformPlan


@tool
def profile_records(records: list[dict[str, Any]], stale_seconds: int) -> dict[str, Any]:
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
    alias_candidates = 0
    within_limit = within_grace = too_old = future = 0
    for record in records:
        status = record.get("status")
        if not isinstance(status, dict):
            continue
        if status.get("num_bikes_available") is None and status.get("bikes_available") is not None:
            alias_candidates += 1
        source_time = record.get("source_updated_at")
        event_time = status.get("last_reported")
        if not all(type(value) is int for value in (source_time, event_time)):
            continue
        age = source_time - event_time
        if age < -60:
            future += 1
        elif age <= stale_seconds:
            within_limit += 1
        elif age <= stale_seconds + 900:
            within_grace += 1
        else:
            too_old += 1
    return Profile(
        rows=len(records),
        unique_stations=count,
        missing_fields={
            k: sum(r.get(k) is None for r in records) for k in ("station_id", "status", "metadata")
        },
        timestamp_min=min(timestamps, default=None),
        timestamp_max=max(timestamps, default=None),
        alias_candidates=alias_candidates,
        within_freshness_limit=within_limit,
        within_freshness_grace=within_grace,
        too_old=too_old,
        future_timestamps=future,
    ).model_dump()


@tool
def execute_plan(
    records: list[dict[str, Any]],
    plan: dict[str, Any],
    stale_seconds: int,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Apply bounded policies; preserve rejected input in quarantine."""
    parsed = TransformPlan.model_validate(plan)
    issues = []
    if parsed.allow_bikes_available_alias and not profile["alias_candidates"]:
        issues.append("Alias enabled but no alias-only bike counts were profiled")
    if parsed.freshness_grace_seconds and not profile["within_freshness_grace"]:
        issues.append("Freshness grace enabled but no observations qualify")
    if issues:
        return {
            "accepted": [],
            "features": [],
            "quarantine": [],
            "duplicates": 0,
            "alias_recovered": 0,
            "grace_accepted": 0,
            "issues": issues,
        }
    accepted, features, quarantine = [], [], []
    seen: set[str] = set()
    duplicates = alias_recovered = grace_accepted = 0
    for original in records:
        try:
            status = original.get("status", {})
            metadata = original.get("metadata", {})
            used_alias = (
                parsed.allow_bikes_available_alias
                and status.get("num_bikes_available") is None
                and status.get("bikes_available") is not None
            )
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
            if used_alias:
                row["num_bikes_available"] = status["bikes_available"]
            observation = Observation.model_validate(row)
            if not all(
                (observation.is_installed, observation.is_renting, observation.is_returning)
            ):
                raise ValueError("Station is not operational")
            age = observation.source_updated_at - observation.last_reported
            if age > stale_seconds + parsed.freshness_grace_seconds:
                raise ValueError("Stale station observation")
            if observation.last_reported > observation.source_updated_at + 60:
                raise ValueError("Station timestamp is in the future")
            if observation.event_id in seen:
                duplicates += 1
                continue
            timestamp = datetime.fromtimestamp(observation.last_reported, UTC)
            feature = {
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
            seen.add(observation.event_id)
            accepted.append(observation.model_dump())
            features.append(feature)
            alias_recovered += int(used_alias)
            grace_accepted += int(age > stale_seconds)
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
        "alias_recovered": alias_recovered,
        "grace_accepted": grace_accepted,
        "issues": [],
    }
