from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from agenticdatapipe.audit import AuditPublisher, PredictionAuditPublisher
from agenticdatapipe.config import Settings
from agenticdatapipe.contracts import PredictionAudit
from agenticdatapipe.training import FEATURE_COLUMNS


class PredictionRequest(BaseModel):
    station_id: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    request_id: str
    station_id: str
    predicted_available_bikes_15m: int
    raw_prediction: float
    current_available_bikes: int
    capacity: int
    horizon_minutes: int = 15
    model_name: str
    model_version: str
    model_alias: str
    feature_observed_at: datetime
    served_at: datetime


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    model_alias: str
    audit_status: str
    audit_failed_events: int


@dataclass(frozen=True)
class PredictionResult:
    response: PredictionResponse
    audit: PredictionAudit


class StationFeaturesNotFound(LookupError):
    pass


class PredictionRuntime:
    def __init__(
        self,
        settings: Settings,
        feature_store: FeatureStore,
        feature_service: Any,
        model: Any,
        model_version: str,
    ) -> None:
        self.settings = settings
        self.feature_store = feature_store
        self.feature_service = feature_service
        self.model = model
        self.model_version = model_version

    @classmethod
    def load(cls, settings: Settings) -> PredictionRuntime:
        settings.feast_registry_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["BIKE_FEAST_HISTORY_PATH"] = str(settings.feast_history_path.resolve())
        os.environ["BIKE_FEAST_REGISTRY_PATH"] = str(settings.feast_registry_path.resolve())
        os.environ["BIKE_FEAST_REDIS_CONNECTION"] = settings.feast_redis_connection

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        version = client.get_model_version_by_alias(
            settings.mlflow_model_name, settings.mlflow_model_alias
        )
        model_uri = (
            f"models:/{settings.mlflow_model_name}@{settings.mlflow_model_alias}"
        )
        model = mlflow.sklearn.load_model(model_uri)
        store = FeatureStore(repo_path=str(settings.feast_repo_path))
        service = store.get_feature_service(settings.feast_feature_service)
        return cls(settings, store, service, model, str(version.version))

    def predict(self, station_id: str) -> PredictionResult:
        values = self.feature_store.get_online_features(
            features=self.feature_service,
            entity_rows=[{"station_id": station_id}],
        ).to_dict()
        missing = [
            name
            for name in FEATURE_COLUMNS
            if name not in values or not values[name] or values[name][0] is None
        ]
        if missing:
            raise StationFeaturesNotFound(
                f"No complete online feature row for station {station_id!r}"
            )
        observed_values = values.get("feature_observed_at")
        if not observed_values or observed_values[0] is None:
            raise StationFeaturesNotFound(
                f"No feature observation timestamp for station {station_id!r}"
            )

        features = {name: values[name][0] for name in FEATURE_COLUMNS}
        model_input = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        raw_prediction = float(self.model.predict(model_input)[0])
        capacity = int(features["capacity"])
        bounded_prediction = max(0, min(capacity, round(raw_prediction)))
        served_at = datetime.now(UTC)
        feature_observed_at = datetime.fromtimestamp(int(observed_values[0]), UTC)
        request_id = uuid4().hex
        response = PredictionResponse(
            request_id=request_id,
            station_id=station_id,
            predicted_available_bikes_15m=bounded_prediction,
            raw_prediction=raw_prediction,
            current_available_bikes=int(features["available_bikes"]),
            capacity=capacity,
            model_name=self.settings.mlflow_model_name,
            model_version=self.model_version,
            model_alias=self.settings.mlflow_model_alias,
            feature_observed_at=feature_observed_at,
            served_at=served_at,
        )
        audit = PredictionAudit(
            event_id=request_id,
            request_id=request_id,
            station_id=station_id,
            served_at=served_at,
            feature_observed_at=feature_observed_at,
            target_at=feature_observed_at + timedelta(minutes=15),
            model_name=self.settings.mlflow_model_name,
            model_version=self.model_version,
            model_alias=self.settings.mlflow_model_alias,
            raw_prediction=raw_prediction,
            predicted_available_bikes_15m=bounded_prediction,
            **features,
        )
        return PredictionResult(response=response, audit=audit)


RuntimeFactory = Callable[[Settings], PredictionRuntime]
AuditPublisherFactory = Callable[[Settings], AuditPublisher]


def create_app(
    settings: Settings | None = None,
    runtime_factory: RuntimeFactory = PredictionRuntime.load,
    audit_publisher_factory: AuditPublisherFactory = PredictionAuditPublisher,
) -> FastAPI:
    app_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.runtime = runtime_factory(app_settings)
            application.state.startup_error = None
        except Exception as exc:  # noqa: BLE001
            application.state.runtime = None
            application.state.startup_error = f"{type(exc).__name__}: {exc}"
        try:
            application.state.audit_publisher = audit_publisher_factory(app_settings)
            application.state.audit_startup_error = None
        except Exception as exc:  # noqa: BLE001
            application.state.audit_publisher = None
            application.state.audit_startup_error = f"{type(exc).__name__}: {exc}"
        try:
            yield
        finally:
            publisher = application.state.audit_publisher
            if publisher is not None:
                publisher.close()

    application = FastAPI(
        title="Bike availability prediction API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        runtime = application.state.runtime
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail=f"Serving dependencies are unavailable: {application.state.startup_error}",
            )
        publisher = application.state.audit_publisher
        audit_status = (
            "unavailable"
            if publisher is None
            else "degraded"
            if publisher.failed_events
            else "ready"
        )
        return HealthResponse(
            status="ready",
            model_name=runtime.settings.mlflow_model_name,
            model_version=runtime.model_version,
            model_alias=runtime.settings.mlflow_model_alias,
            audit_status=audit_status,
            audit_failed_events=publisher.failed_events if publisher is not None else 0,
        )

    @application.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        runtime = application.state.runtime
        if runtime is None:
            raise HTTPException(
                status_code=503,
                detail=f"Serving dependencies are unavailable: {application.state.startup_error}",
            )
        try:
            result = runtime.predict(request.station_id)
            publisher = application.state.audit_publisher
            if publisher is not None:
                try:
                    publisher.publish(result.audit)
                except Exception as exc:  # noqa: BLE001
                    application.state.audit_last_error = f"{type(exc).__name__}: {exc}"
            return result.response
        except StationFeaturesNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Prediction dependencies are unavailable: {exc}"
            ) from exc

    return application


app = create_app()
