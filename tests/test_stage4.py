from datetime import UTC
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from agenticdatapipe.api import PredictionRuntime, create_app
from agenticdatapipe.config import Settings
from agenticdatapipe.training import FEATURE_COLUMNS


class FeatureResponse:
    def __init__(self, values):
        self.values = values

    def to_dict(self):
        return self.values


def feature_values(**overrides):
    values = {
        "feature_observed_at": [1_767_225_600],
        "hour_utc": [8],
        "day_of_week": [2],
        "available_bikes": [12],
        "available_docks": [8],
        "capacity": [20],
        "availability_ratio": [0.6],
        "station_id": ["demo-station-001"],
    }
    values.update(overrides)
    return values


def runtime_with_prediction(prediction: float = 13.4):
    settings = Settings()
    store = Mock()
    store.get_online_features.return_value = FeatureResponse(feature_values())
    model = Mock()
    model.predict.return_value = [prediction]
    runtime = PredictionRuntime(settings, store, object(), model, "7")
    return runtime, store, model


def test_prediction_uses_ordered_feast_features_and_bounds_result():
    runtime, store, model = runtime_with_prediction(25.2)
    result = runtime.predict("demo-station-001")
    response = result.response

    assert response.predicted_available_bikes_15m == 20
    assert response.raw_prediction == 25.2
    assert response.current_available_bikes == 12
    assert response.served_at.tzinfo == UTC
    assert result.audit.model_version == "7"
    assert result.audit.capacity == 20
    frame = model.predict.call_args.args[0]
    assert list(frame.columns) == FEATURE_COLUMNS
    store.get_online_features.assert_called_once()


def test_api_health_and_prediction_contract():
    runtime, _, _ = runtime_with_prediction(13.4)
    publisher = Mock()
    publisher.failed_events = 0
    app = create_app(
        runtime.settings,
        runtime_factory=lambda _: runtime,
        audit_publisher_factory=lambda _: publisher,
    )

    with TestClient(app) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json={"station_id": "demo-station-001"})

    assert health.json() == {
        "status": "ready",
        "model_name": "bike-availability-15m",
        "model_version": "7",
        "model_alias": "champion",
        "audit_status": "ready",
        "audit_failed_events": 0,
    }
    assert prediction.status_code == 200
    assert prediction.json()["predicted_available_bikes_15m"] == 13
    assert prediction.json()["horizon_minutes"] == 15
    publisher.publish.assert_called_once()
    assert publisher.publish.call_args.args[0].station_id == "demo-station-001"
    publisher.close.assert_called_once()


def test_unknown_station_returns_404():
    runtime, store, _ = runtime_with_prediction()
    store.get_online_features.return_value = FeatureResponse(
        feature_values(available_bikes=[None])
    )
    publisher = Mock(failed_events=0)
    app = create_app(
        runtime.settings,
        runtime_factory=lambda _: runtime,
        audit_publisher_factory=lambda _: publisher,
    )

    with TestClient(app) as client:
        response = client.post("/predict", json={"station_id": "unknown"})

    assert response.status_code == 404


def test_unavailable_startup_returns_503():
    def fail(_: Settings):
        raise ConnectionError("MLflow unavailable")

    publisher = Mock(failed_events=0)
    app = create_app(
        Settings(), runtime_factory=fail, audit_publisher_factory=lambda _: publisher
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"station_id": "one"}).status_code == 503


def test_invalid_station_id_returns_422():
    runtime = SimpleNamespace(
        settings=Settings(), model_version="1", predict=Mock()
    )
    publisher = Mock(failed_events=0)
    app = create_app(
        runtime.settings,
        runtime_factory=lambda _: runtime,
        audit_publisher_factory=lambda _: publisher,
    )
    with TestClient(app) as client:
        response = client.post("/predict", json={"station_id": ""})
    assert response.status_code == 422
    runtime.predict.assert_not_called()
