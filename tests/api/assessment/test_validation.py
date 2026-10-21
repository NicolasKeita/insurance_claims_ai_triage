"""Test validation."""

import pytest
from fastapi.testclient import TestClient

from tests.api.assessment._constants import app
from tests.api.assessment._support import make_consistency


@pytest.mark.parametrize(
    "path, payload",
    [
        ("/v1/triage/predict", {"vehicle_age": -5, "repair_amount": -100}),
        ("/v1/anomaly/assess", {"vehicle_age": -5, "repair_amount": -100}),
        (
            "/v1/investigation/assess",
            {
                "features": {"vehicle_age": -5, "repair_amount": -100},
                "consistency": make_consistency().model_dump(mode="json"),
                "anomaly": {
                    "is_anomalous": False,
                    "anomaly_score": 0.4,
                    "threshold": 0.6,
                    "reference_percentile": 0.5,
                },
            },
        ),
        (
            "/v1/claims/assess",
            {
                "features": {"vehicle_age": -5, "repair_amount": -100},
                "consistency": make_consistency().model_dump(mode="json"),
            },
        ),
    ],
)
def test_invalid_features_are_rejected(path, payload):
    with TestClient(app) as client:
        response = client.post(path, json=payload)
        assert response.status_code == 422
