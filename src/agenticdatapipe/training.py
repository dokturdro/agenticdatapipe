"""Synthetic feature history, Feast materialization, and MLflow training."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from datetime import UTC, datetime, timedelta
from types import ModuleType
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from feast import FeatureStore
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from agenticdatapipe.config import Settings

FEATURE_COLUMNS = [
    "hour_utc",
    "day_of_week",
    "available_bikes",
    "available_docks",
    "capacity",
    "availability_ratio",
]
TARGET_COLUMN = "target_available_bikes_15m"


def generate_synthetic_history(
    settings: Settings, *, end_at: datetime | None = None
) -> dict[str, Any]:
    """Write labeled, multi-station history for the local training demonstration."""
    end = (end_at or datetime.now(UTC)).astimezone(UTC)
    end = end.replace(minute=(end.minute // 15) * 15, second=0, microsecond=0)
    periods = settings.synthetic_history_days * 96 + 1
    timestamps = pd.date_range(end=end, periods=periods, freq="15min", tz="UTC")
    rng = np.random.default_rng(settings.synthetic_seed)
    frames: list[pd.DataFrame] = []

    for index in range(settings.synthetic_station_count):
        capacity = 18 + index * 4
        station_bias = (index % 4 - 1.5) * 0.035
        bikes: list[int] = []
        current = int(capacity * (0.45 + station_bias))
        for timestamp in timestamps:
            hour = timestamp.hour + timestamp.minute / 60
            weekday = timestamp.weekday()
            commute = math.sin(2 * math.pi * (hour - 7) / 24)
            weekend = 0.08 if weekday >= 5 else 0.0
            desired = capacity * (0.52 - 0.23 * commute + station_bias + weekend)
            current = int(
                np.clip(round(0.62 * current + 0.38 * desired + rng.normal(0, 0.65)), 0, capacity)
            )
            bikes.append(current)

        frame = pd.DataFrame(
            {
                "station_id": f"demo-station-{index + 1:03d}",
                "event_timestamp": timestamps,
                "hour_utc": timestamps.hour.astype("int64"),
                "day_of_week": timestamps.dayofweek.astype("int64"),
                "available_bikes": bikes,
                "capacity": capacity,
            }
        )
        frame["available_docks"] = frame["capacity"] - frame["available_bikes"]
        frame["availability_ratio"] = frame["available_bikes"] / frame["capacity"]
        frame[TARGET_COLUMN] = frame["available_bikes"].shift(-1)
        frames.append(frame)

    history = pd.concat(frames, ignore_index=True).sort_values(
        ["event_timestamp", "station_id"], ignore_index=True
    )
    path = settings.feast_history_path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    history.to_parquet(temporary, index=False)
    temporary.replace(path)
    labeled_rows = int(history[TARGET_COLUMN].notna().sum())
    return {
        "path": str(path),
        "rows": len(history),
        "labeled_rows": labeled_rows,
        "stations": settings.synthetic_station_count,
        "start": timestamps[0].isoformat(),
        "end": timestamps[-1].isoformat(),
        "source": "synthetic",
    }


def _configure_feast_environment(settings: Settings) -> None:
    settings.feast_registry_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["BIKE_FEAST_HISTORY_PATH"] = str(settings.feast_history_path.resolve())
    os.environ["BIKE_FEAST_REGISTRY_PATH"] = str(settings.feast_registry_path.resolve())
    os.environ["BIKE_FEAST_REDIS_CONNECTION"] = settings.feast_redis_connection


def _load_feature_definitions(settings: Settings) -> ModuleType:
    definition_path = settings.feast_repo_path / "features.py"
    spec = importlib.util.spec_from_file_location("bike_feature_definitions", definition_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Feast definitions from {definition_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_and_materialize_features(settings: Settings) -> dict[str, Any]:
    """Apply Feast objects, load history into Redis, and verify an online read."""
    _configure_feast_environment(settings)
    definitions = _load_feature_definitions(settings)
    store = FeatureStore(repo_path=str(settings.feast_repo_path))
    store.apply(
        [
            definitions.station,
            definitions.station_features,
            definitions.station_feature_service,
        ]
    )
    history = pd.read_parquet(settings.feast_history_path)
    start = history["event_timestamp"].min().to_pydatetime() - timedelta(minutes=1)
    end = history["event_timestamp"].max().to_pydatetime() + timedelta(minutes=1)
    store.materialize(start_date=start, end_date=end)
    station_id = str(history.iloc[-1]["station_id"])
    response = store.get_online_features(
        features=[f"station_features:{name}" for name in FEATURE_COLUMNS],
        entity_rows=[{"station_id": station_id}],
    ).to_dict()
    if response.get("available_bikes", [None])[0] is None:
        raise RuntimeError("Feast materialization completed without an online feature value")
    return {
        "station_id": station_id,
        "materialized_through": end.isoformat(),
        "online_features": {key: values[0] for key, values in response.items()},
    }


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split complete timestamps so no station at a timestamp crosses a boundary."""
    timestamps = np.array(sorted(frame["event_timestamp"].unique()))
    if len(timestamps) < 10:
        raise ValueError("At least 10 distinct timestamps are required for training")
    train_end = max(1, int(len(timestamps) * 0.70))
    validation_end = max(train_end + 1, int(len(timestamps) * 0.85))
    train_times = set(timestamps[:train_end])
    validation_times = set(timestamps[train_end:validation_end])
    test_times = set(timestamps[validation_end:])
    return (
        frame[frame["event_timestamp"].isin(train_times)].copy(),
        frame[frame["event_timestamp"].isin(validation_times)].copy(),
        frame[frame["event_timestamp"].isin(test_times)].copy(),
    )


def should_promote(model_mae: float, baseline_mae: float) -> bool:
    """The candidate must strictly improve on persistence."""
    return model_mae < baseline_mae


def _registered_version(client: MlflowClient, model_name: str, run_id: str) -> str:
    versions = client.search_model_versions(f"name = '{model_name}'")
    matching = [version for version in versions if version.run_id == run_id]
    if not matching:
        raise RuntimeError(f"No registered model version found for MLflow run {run_id}")
    return str(max(matching, key=lambda version: int(version.version)).version)


def train_and_register_model(settings: Settings) -> dict[str, Any]:
    """Train through Feast historical retrieval and gate MLflow champion promotion."""
    _configure_feast_environment(settings)
    history = pd.read_parquet(settings.feast_history_path)
    entities = history.dropna(subset=[TARGET_COLUMN])[
        ["station_id", "event_timestamp", TARGET_COLUMN]
    ]
    store = FeatureStore(repo_path=str(settings.feast_repo_path))
    training_set = store.get_historical_features(
        entity_df=entities,
        features=[f"station_features:{name}" for name in FEATURE_COLUMNS],
    ).to_df()
    if training_set[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feast returned missing training features")

    train, validation, test = chronological_split(training_set)
    model = HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=settings.synthetic_seed,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    validation_prediction = model.predict(validation[FEATURE_COLUMNS])
    test_prediction = model.predict(test[FEATURE_COLUMNS])
    validation_mae = float(mean_absolute_error(validation[TARGET_COLUMN], validation_prediction))
    baseline_mae = float(
        mean_absolute_error(validation[TARGET_COLUMN], validation["available_bikes"])
    )
    promoted = should_promote(validation_mae, baseline_mae)
    metrics = {
        "validation_mae": validation_mae,
        "baseline_validation_mae": baseline_mae,
        "test_mae": float(mean_absolute_error(test[TARGET_COLUMN], test_prediction)),
        "test_rmse": float(mean_squared_error(test[TARGET_COLUMN], test_prediction) ** 0.5),
        "test_r2": float(r2_score(test[TARGET_COLUMN], test_prediction)),
    }

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name=f"bike-availability-{datetime.now(UTC):%Y%m%d-%H%M%S}") as run:
        mlflow.log_params(
            {
                "algorithm": "HistGradientBoostingRegressor",
                "target": TARGET_COLUMN,
                "horizon_minutes": 15,
                "features": ",".join(FEATURE_COLUMNS),
                "synthetic_seed": settings.synthetic_seed,
                "training_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {
                "source": "synthetic",
                "history_path": str(settings.feast_history_path),
                "promotion_gate": "validation_mae < baseline_validation_mae",
                "promoted": promoted,
            },
            "training_summary.json",
        )
        signature = infer_signature(train[FEATURE_COLUMNS], model.predict(train[FEATURE_COLUMNS]))
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=settings.mlflow_model_name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=signature,
            input_example=train[FEATURE_COLUMNS].head(5),
        )
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = _registered_version(client, settings.mlflow_model_name, run_id)
    status = "passed" if promoted else "failed"
    client.set_model_version_tag(settings.mlflow_model_name, version, "validation_status", status)
    client.set_registered_model_alias(settings.mlflow_model_name, "candidate", version)
    if promoted:
        client.set_registered_model_alias(settings.mlflow_model_name, "champion", version)
    result = {
        "run_id": run_id,
        "model_name": settings.mlflow_model_name,
        "model_version": version,
        "candidate_alias": version,
        "champion_alias": version if promoted else None,
        "promoted": promoted,
        "metrics": metrics,
    }
    report_path = settings.data_dir / "training" / f"{run_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
