"""Feast feature definitions shared by training and the later serving stage."""

import os
from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64

station = Entity(name="station", join_keys=["station_id"], value_type=ValueType.STRING)

station_history = FileSource(
    name="station_history",
    path=os.environ["BIKE_FEAST_HISTORY_PATH"],
    timestamp_field="event_timestamp",
)

station_features = FeatureView(
    name="station_features",
    entities=[station],
    ttl=timedelta(days=2),
    schema=[
        Field(name="hour_utc", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="available_bikes", dtype=Int64),
        Field(name="available_docks", dtype=Int64),
        Field(name="capacity", dtype=Int64),
        Field(name="availability_ratio", dtype=Float32),
    ],
    source=station_history,
    online=True,
)

station_feature_service = FeatureService(
    name="bike_availability_service",
    features=[station_features],
)
