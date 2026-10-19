"""Fit an IsolationForest novelty detector on a reference population."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from ml.anomaly_training.detector import (
    ANOMALY_NUMERICAL_FEATURES,
    AnomalyDetector,
    select_anomaly_features,
)
from ml.training import build_preprocessor


def train_anomaly_detector(
    reference: pd.DataFrame,
    *,
    seed: int = 42,
    threshold_percentile: float = 0.98,
    n_estimators: int = 200,
) -> AnomalyDetector:
    """Fit solely on reference claims and calibrate a reference score quantile."""
    _validate_reference(reference, threshold_percentile, n_estimators)
    features = select_anomaly_features(reference)
    pipeline = _build_pipeline(seed, n_estimators)
    pipeline.fit(features)
    reference_scores = np.sort(-np.asarray(pipeline.score_samples(features), dtype=float))
    return AnomalyDetector(
        pipeline=pipeline,
        threshold=float(np.quantile(reference_scores, threshold_percentile)),
        threshold_percentile=threshold_percentile,
        reference_scores=reference_scores,
        numerical_reference_bounds=_reference_bounds(features),
        metadata={
            "seed": seed,
            "reference_count": len(reference),
            "n_estimators": n_estimators,
            "score_convention": "higher_is_more_anomalous",
        },
    )


def _validate_reference(reference, threshold_percentile, n_estimators):
    if not 0 < threshold_percentile < 1:
        raise ValueError("threshold_percentile must be between 0 and 1")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if "is_synthetic_anomaly" in reference.columns:
        labels = reference["is_synthetic_anomaly"].astype(str).str.strip().str.lower()
        if not labels.isin(("false", "0")).all():
            raise ValueError("Reference population must not contain injected anomalies")


def _build_pipeline(seed, n_estimators):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "isolation_forest",
                IsolationForest(n_estimators=n_estimators, contamination="auto", random_state=seed),
            ),
        ]
    )


def _reference_bounds(features):
    return {
        name: (float(features[name].quantile(0.01)), float(features[name].quantile(0.99)))
        for name in ANOMALY_NUMERICAL_FEATURES
    }
