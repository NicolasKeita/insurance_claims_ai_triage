"""Pure tracking helpers for the novelty detection experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.anomaly_evaluation import AnomalyMetrics
from ml.anomaly_training import AnomalyDetector, select_anomaly_features


def read_dataset_metadata(path: Path, *, expected_role: str) -> dict:
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("role") != expected_role:
        raise ValueError(
            f"{metadata_path} must have role={expected_role!r}"
        )
    if not metadata.get("dataset_version"):
        raise ValueError(f"{metadata_path} has no dataset_version")
    return metadata


def mlflow_anomaly_metrics(metrics: AnomalyMetrics) -> dict[str, float]:
    """Flatten measured benchmark results, deriving error rates from its matrix."""
    true_normal, true_anomaly = metrics.confusion_matrix
    false_positive_count = true_normal[1]
    false_negative_count = true_anomaly[0]
    normal_count = sum(true_normal)
    anomaly_count = sum(true_anomaly)
    values = {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "flagged_count": metrics.flagged_count,
        "flagged_rate": metrics.flagged_rate,
        "false_positive_count": false_positive_count,
        "false_negative_count": false_negative_count,
        "threshold": metrics.threshold,
    }
    if normal_count:
        values["false_positive_rate"] = false_positive_count / normal_count
    if anomaly_count:
        values["false_negative_rate"] = false_negative_count / anomaly_count
    if metrics.average_precision is not None:
        values["average_precision"] = metrics.average_precision
    if metrics.roc_auc is not None:
        values["roc_auc"] = metrics.roc_auc
    return {key: float(value) for key, value in values.items()}


def verify_anomaly_model_round_trip(
    detector: AnomalyDetector,
    loaded_model,
    examples: pd.DataFrame,
) -> pd.DataFrame:
    """Require the reloaded PyFunc to agree with the original detector."""
    features = select_anomaly_features(examples)
    actual = loaded_model.predict(features)
    required = {
        "is_anomalous",
        "anomaly_score",
        "threshold",
        "reference_percentile",
        "statistical_signals",
    }
    if not isinstance(actual, pd.DataFrame) or len(actual) != len(features):
        raise RuntimeError("Reloaded anomaly model returned an invalid row count")
    if not required.issubset(actual.columns):
        raise RuntimeError("Reloaded anomaly model returned an invalid schema")

    for position, (_, row) in enumerate(features.iterrows()):
        expected = detector.assess(row)
        received = actual.iloc[position]
        if bool(received["is_anomalous"]) != expected.is_anomalous:
            raise RuntimeError(f"Anomaly classification changed at row {position}")
        for field in ("anomaly_score", "threshold", "reference_percentile"):
            if not np.isclose(
                float(received[field]),
                getattr(expected, field),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(f"Anomaly {field} changed at row {position}")
        if json.loads(received["statistical_signals"]) != expected.statistical_signals:
            raise RuntimeError(f"Anomaly signals changed at row {position}")
    return actual
