"""Wire contracts and structured agent outputs."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Observation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal[1] = 1
    event_id: str
    system_id: str
    station_id: str = Field(min_length=1)
    source_updated_at: int = Field(gt=0)
    last_reported: int = Field(gt=0)
    ingested_at: int = Field(gt=0)
    num_bikes_available: int = Field(ge=0)
    num_docks_available: int = Field(ge=0)
    is_installed: Literal[0, 1]
    is_renting: Literal[0, 1]
    is_returning: Literal[0, 1]
    capacity: int = Field(gt=0)
    station_name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


class TransformPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allow_bikes_available_alias: bool
    freshness_grace_seconds: Literal[0, 900]
    rationale: str


class Profile(BaseModel):
    rows: int
    unique_stations: int
    missing_fields: dict[str, int]
    timestamp_min: int | None = None
    timestamp_max: int | None = None
    alias_candidates: int = 0
    within_freshness_limit: int = 0
    within_freshness_grace: int = 0
    too_old: int = 0
    future_timestamps: int = 0


class QualityReport(BaseModel):
    input_rows: int
    accepted_rows: int
    quarantined_rows: int
    duplicate_rows: int
    alias_recovered_rows: int = 0
    grace_accepted_rows: int = 0
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    repairable: bool = False


class Batch(BaseModel):
    batch_id: str
    records: list[dict[str, Any]]
    # Kafka offset ranges are inclusive; commit advances each end by one.
    offsets: dict[str, dict[str, int]] = Field(default_factory=dict)
