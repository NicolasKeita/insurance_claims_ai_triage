"""Shared configuration and fixed dataset contracts."""

from api.main import AppConfig, create_app

from ._support import fake_anomaly_predictor_loader, fake_predictor_loader

app = create_app(
    AppConfig(
        predictor_loader=fake_predictor_loader,
        anomaly_predictor_loader=fake_anomaly_predictor_loader,
    )
)
