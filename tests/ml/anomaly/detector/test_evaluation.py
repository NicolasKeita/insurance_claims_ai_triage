"""Test evaluation."""

import pandas as pd
import pytest

from ml.anomaly_dataset import ANOMALY_KINDS
from ml.anomaly_evaluation import evaluate_anomaly_detector
from tests.ml.anomaly.detector._support import anomaly_datasets, anomaly_detector
from tests.ml.anomaly.detector._support import anomaly_datasets as anomaly_datasets
from tests.ml.anomaly.detector._support import anomaly_detector as anomaly_detector


def test_anomaly_evaluation_reports_bounded_metrics_and_errors(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
    anomaly_detector,
) -> None:
    _, evaluation = anomaly_datasets
    result = evaluate_anomaly_detector(anomaly_detector, evaluation)
    metrics = result.metrics

    for value in (
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.flagged_rate,
        metrics.average_precision,
        metrics.roc_auc,
    ):
        assert 0 <= value <= 1

    assert len(metrics.confusion_matrix) == 2
    assert all(len(row) == 2 for row in metrics.confusion_matrix)
    assert sum(map(sum, metrics.confusion_matrix)) == len(evaluation)
    assert 0 <= metrics.flagged_count <= len(evaluation)
    assert metrics.flagged_rate == pytest.approx(metrics.flagged_count / len(evaluation))
    assert set(ANOMALY_KINDS) <= set(metrics.by_anomaly_kind)
    assert {"error_type", "anomaly_kind"} <= set(result.errors.columns)
    assert set(result.errors["error_type"]) <= {
        "false_positive",
        "false_negative",
    }
