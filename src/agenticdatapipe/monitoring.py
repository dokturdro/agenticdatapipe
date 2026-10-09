from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from deltalake import DeltaTable
from deltalake.exceptions import TableNotFoundError
from mlflow import MlflowClient

from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import MonitoringReport
from agenticdatapipe.storage import delta_uri
from agenticdatapipe.training import FEATURE_COLUMNS


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    *,
    bins: int = 10,
) -> float:
    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]
    if reference_values.size == 0 or current_values.size == 0:
        return 0.0
    if np.allclose(reference_values, reference_values[0]):
        return 0.0 if np.allclose(current_values, reference_values[0]) else 1.0

    edges = np.unique(np.quantile(reference_values, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)
    epsilon = 1e-6
    reference_ratio = np.maximum(reference_counts / reference_values.size, epsilon)
    current_ratio = np.maximum(current_counts / current_values.size, epsilon)
    return float(np.sum((current_ratio - reference_ratio) * np.log(current_ratio / reference_ratio)))


def join_predictions_to_actuals(
    audits: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    tolerance_seconds: int,
) -> pd.DataFrame:
    joined_parts: list[pd.DataFrame] = []
    audits = audits.copy()
    audits["target_at"] = pd.to_datetime(audits["target_at"], utc=True)
    observations = observations.copy()
    observations["actual_at"] = pd.to_datetime(
        observations["last_reported"], unit="s", utc=True
    )
    actual_columns = ["actual_at", "num_bikes_available"]

    for station_id, station_audits in audits.groupby("station_id", sort=False):
        station_observations = observations[observations["station_id"] == station_id]
        station_audits = station_audits.sort_values("target_at")
        if station_observations.empty:
            station_audits["actual_at"] = pd.NaT
            station_audits["actual_available_bikes"] = np.nan
            joined_parts.append(station_audits)
            continue
        station_observations = station_observations[actual_columns].sort_values("actual_at")
        part = pd.merge_asof(
            station_audits,
            station_observations,
            left_on="target_at",
            right_on="actual_at",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=tolerance_seconds),
        ).rename(columns={"num_bikes_available": "actual_available_bikes"})
        joined_parts.append(part)

    if not joined_parts:
        return audits.assign(actual_at=pd.NaT, actual_available_bikes=np.nan)
    return pd.concat(joined_parts, ignore_index=True)


def evaluate_monitoring_window(
    audits: pd.DataFrame,
    observations: pd.DataFrame,
    reference: pd.DataFrame,
    reference_predictions: np.ndarray,
    settings: Settings,
    *,
    model_version: str,
    window_start: datetime,
    window_end: datetime,
    training_test_mae: float | None = None,
) -> tuple[MonitoringReport, pd.DataFrame, dict[str, Any]]:
    audits = audits.copy()
    audits["served_at"] = pd.to_datetime(audits["served_at"], utc=True)
    audits["feature_observed_at"] = pd.to_datetime(audits["feature_observed_at"], utc=True)
    audits["target_at"] = pd.to_datetime(audits["target_at"], utc=True)
    selected = audits[
        (audits["served_at"] >= pd.Timestamp(window_start))
        & (audits["served_at"] < pd.Timestamp(window_end))
        & (audits["target_at"] <= pd.Timestamp(window_end))
        & (audits["model_version"].astype(str) == str(model_version))
    ].copy()
    joined = join_predictions_to_actuals(
        selected,
        observations,
        tolerance_seconds=settings.monitoring_label_tolerance_seconds,
    )

    sample_count = len(joined)
    labeled = joined.dropna(subset=["actual_available_bikes"]).copy()
    labeled_count = len(labeled)
    label_coverage = labeled_count / sample_count if sample_count else 0.0
    invalid = pd.Series(False, index=joined.index)
    if sample_count:
        invalid = (
            joined[FEATURE_COLUMNS].isna().any(axis=1)
            | (joined["capacity"] <= 0)
            | (joined["available_bikes"] < 0)
            | (joined["available_bikes"] > joined["capacity"])
            | (joined["available_docks"] < 0)
            | (joined["available_docks"] > joined["capacity"])
            | ~joined["availability_ratio"].between(0, 1)
        )
    stale = (
        joined["served_at"] - joined["feature_observed_at"]
        > pd.Timedelta(seconds=settings.stale_seconds)
    ) if sample_count else pd.Series(dtype=bool)

    metrics: dict[str, float] = {
        "sample_count": float(sample_count),
        "labeled_count": float(labeled_count),
        "label_coverage": float(label_coverage),
        "invalid_feature_rate": float(invalid.mean()) if sample_count else 0.0,
        "stale_feature_rate": float(stale.mean()) if sample_count else 0.0,
    }
    for column in FEATURE_COLUMNS:
        metrics[f"psi_{column}"] = population_stability_index(
            reference[column], joined[column] if sample_count else np.array([])
        )
    metrics["psi_prediction"] = population_stability_index(
        reference_predictions,
        joined["raw_prediction"] if sample_count else np.array([]),
    )

    station_breakdown: dict[str, Any] = {}
    if labeled_count:
        errors = (
            labeled["predicted_available_bikes_15m"] - labeled["actual_available_bikes"]
        )
        baseline_errors = labeled["available_bikes"] - labeled["actual_available_bikes"]
        metrics["production_mae"] = float(errors.abs().mean())
        metrics["production_rmse"] = float(math.sqrt((errors**2).mean()))
        metrics["persistence_baseline_mae"] = float(baseline_errors.abs().mean())
        for station_id, station_rows in labeled.groupby("station_id"):
            station_errors = (
                station_rows["predicted_available_bikes_15m"]
                - station_rows["actual_available_bikes"]
            )
            station_breakdown[str(station_id)] = {
                "samples": len(station_rows),
                "mae": float(station_errors.abs().mean()),
            }
    if training_test_mae is not None:
        metrics["training_test_mae"] = float(training_test_mae)
        if "production_mae" in metrics and training_test_mae > 0:
            metrics["production_to_training_mae_ratio"] = (
                metrics["production_mae"] / training_test_mae
            )

    warnings: list[str] = []
    failures: list[str] = []
    if sample_count < settings.monitoring_min_samples:
        status = "insufficient_data"
    else:
        if metrics["invalid_feature_rate"] > 0:
            failures.append("Production feature rows violated model input constraints")
        if label_coverage < settings.monitoring_min_label_coverage:
            warnings.append("Delayed-label coverage is below the configured minimum")
        if metrics["stale_feature_rate"] > 0:
            warnings.append("Some predictions used stale online features")
        for name, value in metrics.items():
            if not name.startswith("psi_"):
                continue
            if value >= settings.monitoring_psi_failure:
                failures.append(f"{name} exceeded the failure threshold")
            elif value >= settings.monitoring_psi_warning:
                warnings.append(f"{name} exceeded the warning threshold")
        if labeled_count >= settings.monitoring_min_samples:
            if metrics["production_mae"] >= metrics["persistence_baseline_mae"]:
                failures.append("Production MAE no longer beats the persistence baseline")
        else:
            warnings.append("Not enough delayed labels to evaluate production model quality")
        status = "failed" if failures else "warning" if warnings else "passed"

    report = MonitoringReport(
        status=status,
        model_name=settings.mlflow_model_name,
        model_version=str(model_version),
        window_start=window_start,
        window_end=window_end,
        metrics=metrics,
        warnings=warnings,
        failures=failures,
    )
    return report, joined, station_breakdown


def _read_delta(settings: Settings, location: str) -> pd.DataFrame:
    uri = delta_uri(settings, location)
    options = settings.delta_storage_options() if settings.storage_backend == "delta" else None
    try:
        return DeltaTable(uri, storage_options=options).to_pandas()
    except TableNotFoundError:
        return pd.DataFrame()


def run_monitoring(settings: Settings, *, now: datetime | None = None) -> dict[str, Any]:
    window_end = (now or datetime.now(UTC)).astimezone(UTC)
    window_start = window_end - timedelta(hours=settings.monitoring_window_hours)
    audits = _read_delta(settings, settings.prediction_audits_table)
    observations = _read_delta(settings, settings.observations_table)
    if audits.empty:
        audits = pd.DataFrame(
            columns=[
                "served_at",
                "feature_observed_at",
                "target_at",
                "model_version",
                "station_id",
                "raw_prediction",
                "predicted_available_bikes_15m",
                *FEATURE_COLUMNS,
            ]
        )
    if observations.empty:
        observations = pd.DataFrame(
            columns=["station_id", "last_reported", "num_bikes_available"]
        )

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    version = client.get_model_version_by_alias(
        settings.mlflow_model_name, settings.mlflow_model_alias
    )
    model_version = str(version.version)
    training_run = client.get_run(version.run_id)
    training_test_mae = training_run.data.metrics.get("test_mae")
    reference = pd.read_parquet(settings.feast_history_path)
    model = mlflow.sklearn.load_model(
        f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
    )
    reference_predictions = np.asarray(model.predict(reference[FEATURE_COLUMNS]), dtype=float)
    report, joined, station_breakdown = evaluate_monitoring_window(
        audits,
        observations,
        reference,
        reference_predictions,
        settings,
        model_version=model_version,
        window_start=window_start,
        window_end=window_end,
        training_test_mae=training_test_mae,
    )

    mlflow.set_experiment(settings.mlflow_monitoring_experiment)
    with mlflow.start_run(run_name=f"production-monitor-{window_end:%Y%m%d-%H%M%S}") as run:
        mlflow.set_tags(
            {
                "monitoring_status": report.status,
                "model_name": report.model_name,
                "model_version": report.model_version,
                "window_start": report.window_start.isoformat(),
                "window_end": report.window_end.isoformat(),
            }
        )
        mlflow.log_metrics(report.metrics)
        mlflow.log_dict(report.model_dump(mode="json"), "monitoring_report.json")
        mlflow.log_dict(station_breakdown, "station_breakdown.json")
        records = json.loads(joined.to_json(orient="records", date_format="iso"))
        mlflow.log_dict({"records": records}, "labeled_predictions.json")
        run_id = run.info.run_id

    result = report.model_dump(mode="json")
    result["run_id"] = run_id
    return result
