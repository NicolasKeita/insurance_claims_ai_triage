"""Test metadata."""

import json

import pytest

from ml.anomaly_evaluation import AnomalyMetrics
from ml.anomaly_tracking import (
    mlflow_anomaly_metrics,
    read_dataset_metadata,
)


def test_dataset_metadata_role_and_version(tmp_path):
    path = tmp_path / "reference.csv"
    path.with_suffix(".metadata.json").write_text(
        json.dumps({"role": "reference", "dataset_version": "v1"}), encoding="utf-8"
    )
    assert read_dataset_metadata(path, expected_role="reference")["dataset_version"] == "v1"
    with pytest.raises(ValueError, match="role"):
        read_dataset_metadata(path, expected_role="evaluation")


def test_metrics_include_measured_errors_and_rates():
    metrics = AnomalyMetrics(
        sample_count=10,
        synthetic_anomaly_count=3,
        flagged_count=3,
        flagged_rate=0.3,
        precision=2 / 3,
        recall=2 / 3,
        f1=2 / 3,
        confusion_matrix=[[6, 1], [1, 2]],
        average_precision=0.8,
        roc_auc=0.9,
        threshold=0.6,
        threshold_percentile=0.98,
        by_anomaly_kind={},
    )
    values = mlflow_anomaly_metrics(metrics)
    assert values["false_positive_count"] == 1
    assert values["false_negative_count"] == 1
    assert values["false_positive_rate"] == pytest.approx(1 / 7)
    assert values["false_negative_rate"] == pytest.approx(1 / 3)
    assert values["average_precision"] == 0.8
