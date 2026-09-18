"""FastAPI inference service backed by Feast online features and MLflow."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from feast import FeatureStore
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from agenticdatapipe.config import Settings
from agenticdatapipe.training import FEATURE_COLUMNS


class PredictionRequest(BaseModel):
    station_id: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    station_id: str
    predicted_available_bikes_15m: int
    raw_prediction: float
    current_available_bikes: int
    capacity: int
    horizon_minutes: int = 15
    model_name: str
    model_version: str
    model_alias: str
    served_at: datetime


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    model_alias: str


class StationFeaturesNotFound(LookupError):
    """Raised when Feast has no complete online row for a station."""


class PredictionRuntime:
    """Immutable serving dependencies loaded once when the API starts."""

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

    def predict(self, station_id: str) -> PredictionResponse:
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

        features = {name: values[name][0] for name in FEATURE_COLUMNS}
        model_input = pd.DataFrame([features], columns=FEATURE_COLUMNS)
        raw_prediction = float(self.model.predict(model_input)[0])
        capacity = int(features["capacity"])
        bounded_prediction = max(0, min(capacity, round(raw_prediction)))
        return PredictionResponse(
            station_id=station_id,
            predicted_available_bikes_15m=bounded_prediction,
            raw_prediction=raw_prediction,
            current_available_bikes=int(features["available_bikes"]),
            capacity=capacity,
            model_name=self.settings.mlflow_model_name,
            model_version=self.model_version,
            model_alias=self.settings.mlflow_model_alias,
            served_at=datetime.now(UTC),
        )


RuntimeFactory = Callable[[Settings], PredictionRuntime]


def create_app(
    settings: Settings | None = None,
    runtime_factory: RuntimeFactory = PredictionRuntime.load,
) -> FastAPI:
    app_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            application.state.runtime = runtime_factory(app_settings)
            application.state.startup_error = None
        except Exception as exc:  # noqa: BLE001 -- expose dependency failure as readiness.
            application.state.runtime = None
            application.state.startup_error = f"{type(exc).__name__}: {exc}"
        yield

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
        return HealthResponse(
            status="ready",
            model_name=runtime.settings.mlflow_model_name,
            model_version=runtime.model_version,
            model_alias=runtime.settings.mlflow_model_alias,
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
            return runtime.predict(request.station_id)
        except StationFeaturesNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"Prediction dependencies are unavailable: {exc}"
            ) from exc

    return application


app = create_app()
