"""Evaluation of a novelty detector against synthetic benchmark labels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.anomaly_training import AnomalyDetector


class AnomalyKindMetrics(BaseModel):
    sample_count: int = Field(ge=0)
    synthetic_anomaly_count: int = Field(ge=0)
    flagged_count: int = Field(ge=0)
    flagged_rate: float = Field(ge=0, le=1)
    recall: float | None = Field(default=None, ge=0, le=1)
    mean_anomaly_score: float


class AnomalyMetrics(BaseModel):
    sample_count: int = Field(ge=0)
    synthetic_anomaly_count: int = Field(ge=0)
    flagged_count: int = Field(ge=0)
    flagged_rate: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)
    confusion_matrix: list[list[int]]
    average_precision: float | None = Field(default=None, ge=0, le=1)
    roc_auc: float | None = Field(default=None, ge=0, le=1)
    threshold: float
    threshold_percentile: float = Field(ge=0, le=1)
    by_anomaly_kind: dict[str, AnomalyKindMetrics]


@dataclass
class AnomalyEvaluation:
    metrics: AnomalyMetrics
    errors: pd.DataFrame


def _read_synthetic_labels(evaluation: pd.DataFrame) -> np.ndarray:
    labels = evaluation["is_synthetic_anomaly"]
    if labels.isna().any():
        raise ValueError("is_synthetic_anomaly must contain boolean values")
    if pd.api.types.is_bool_dtype(labels):
        return labels.to_numpy(dtype=bool)

    parsed = labels.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError("is_synthetic_anomaly must contain boolean values")
    return parsed.to_numpy(dtype=bool)


def evaluate_anomaly_detector(
    detector: AnomalyDetector,
    evaluation: pd.DataFrame,
) -> AnomalyEvaluation:
    """Measure benchmark performance without using labels as model inputs."""
    required = {"is_synthetic_anomaly", "anomaly_kind"}
    missing = required - set(evaluation.columns)
    if missing:
        raise ValueError("Evaluation is missing columns: " + ", ".join(sorted(missing)))
    if evaluation.empty:
        raise ValueError("Evaluation dataset is empty")
    if evaluation["anomaly_kind"].isna().any():
        raise ValueError("anomaly_kind must not contain missing values")

    y_true = _read_synthetic_labels(evaluation)
    scores = detector.score_samples(evaluation)
    y_pred = scores >= detector.threshold
    percentiles = detector.reference_percentiles(scores)
    has_both_classes = bool(np.any(y_true) and np.any(~y_true))

    kinds = evaluation["anomaly_kind"].astype(str).to_numpy()
    by_kind = {}
    for kind in sorted(set(kinds)):
        mask = kinds == kind
        count = int(np.sum(mask))
        synthetic_count = int(np.sum(y_true[mask]))
        flagged_count = int(np.sum(y_pred[mask]))
        by_kind[kind] = AnomalyKindMetrics(
            sample_count=count,
            synthetic_anomaly_count=synthetic_count,
            flagged_count=flagged_count,
            flagged_rate=flagged_count / count,
            recall=(
                int(np.sum(y_pred[mask] & y_true[mask])) / synthetic_count
                if synthetic_count
                else None
            ),
            mean_anomaly_score=float(np.mean(scores[mask])),
        )

    metrics = AnomalyMetrics(
        sample_count=len(evaluation),
        synthetic_anomaly_count=int(np.sum(y_true)),
        flagged_count=int(np.sum(y_pred)),
        flagged_rate=float(np.mean(y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=confusion_matrix(
            y_true, y_pred, labels=[False, True]
        ).tolist(),
        average_precision=(
            float(average_precision_score(y_true, scores))
            if has_both_classes
            else None
        ),
        roc_auc=(
            float(roc_auc_score(y_true, scores)) if has_both_classes else None
        ),
        threshold=detector.threshold,
        threshold_percentile=detector.threshold_percentile,
        by_anomaly_kind=by_kind,
    )

    false_positives = y_pred & ~y_true
    false_negatives = ~y_pred & y_true
    error_mask = false_positives | false_negatives
    errors = evaluation.iloc[np.flatnonzero(error_mask)].copy()
    errors["anomaly_score"] = scores[error_mask]
    errors["reference_percentile"] = percentiles[error_mask]
    errors["is_anomalous"] = y_pred[error_mask]
    errors["error_type"] = np.where(
        false_positives[error_mask], "false_positive", "false_negative"
    )

    return AnomalyEvaluation(metrics=metrics, errors=errors)
