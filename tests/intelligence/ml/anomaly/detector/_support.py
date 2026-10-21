"""Support."""

import pandas as pd
import pytest

from ml.anomaly_dataset import generate_anomaly_datasets
from ml.anomaly_training import (
    train_anomaly_detector,
)


@pytest.fixture(scope="module")
def anomaly_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    return generate_anomaly_datasets(
        reference_count=300,
        evaluation_count=160,
        anomaly_fraction=0.25,
        seed=42,
    )


@pytest.fixture(scope="module")
def anomaly_detector(anomaly_datasets):
    reference, _ = anomaly_datasets
    return train_anomaly_detector(reference, seed=42)


def extreme_cases(reference: pd.DataFrame) -> pd.DataFrame:
    cases = reference.iloc[:12].copy()
    cases["collision_type"] = "PARKING_DAMAGE"
    cases["repair_amount"] = 29_000.0
    cases["declared_damage_count"] = 8
    cases["additional_damage_count"] = 4
    cases["document_count"] = 6
    cases["missing_document_count"] = 5
    cases["document_completeness_ratio"] = 1 / 6
    cases["consistency_issue_count"] = 8
    return cases
