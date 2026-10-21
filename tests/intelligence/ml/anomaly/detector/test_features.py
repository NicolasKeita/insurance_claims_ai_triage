"""Test features."""

import pandas as pd

from ml.anomaly_training import (
    ANOMALY_FEATURE_COLUMNS,
    select_anomaly_features,
)
from tests.intelligence.ml.anomaly.detector._support import anomaly_datasets, anomaly_detector
from tests.intelligence.ml.anomaly.detector._support import anomaly_datasets as anomaly_datasets
from tests.intelligence.ml.anomaly.detector._support import anomaly_detector as anomaly_detector


def test_training_features_exclude_outcomes_and_synthetic_labels(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
    anomaly_detector,
) -> None:
    reference, evaluation = anomaly_datasets
    with_outcomes = evaluation.assign(workflow="STANDARD")
    selected = select_anomaly_features(with_outcomes)

    assert list(selected.columns) == list(ANOMALY_FEATURE_COLUMNS)
    assert set(selected.columns).isdisjoint({"workflow", "is_synthetic_anomaly", "anomaly_kind"})
    assert list(anomaly_detector.pipeline.feature_names_in_) == list(ANOMALY_FEATURE_COLUMNS)
    assert list(reference.columns) == list(ANOMALY_FEATURE_COLUMNS)
