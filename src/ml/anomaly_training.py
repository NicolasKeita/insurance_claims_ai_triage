"""Novelty detection against an unmodified reference population."""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from ml.features import TriageFeatures
from ml.training import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
)


# This is the complete model input. Workflow and synthetic benchmark labels are
# intentionally absent, even when they are present in a source DataFrame.
ANOMALY_FEATURE_COLUMNS = tuple(CATEGORICAL_FEATURES + NUMERICAL_FEATURES)
ANOMALY_NUMERICAL_FEATURES = tuple(
    column for column in NUMERICAL_FEATURES if column != "injuries_declared"
)


class AnomalyAssessment(BaseModel):
    is_anomalous: bool
    anomaly_score: float
    threshold: float
    reference_percentile: float = Field(ge=0, le=1)
    statistical_signals: list[str] = Field(default_factory=list)


def select_anomaly_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Select only the eleven observed claim features used for novelty detection."""
    missing = set(ANOMALY_FEATURE_COLUMNS) - set(dataset.columns)
    if missing:
        raise ValueError(
            "Dataset is missing anomaly features: " + ", ".join(sorted(missing))
        )
    if dataset.empty:
        raise ValueError("Dataset is empty")

    features = dataset.loc[:, list(ANOMALY_FEATURE_COLUMNS)].copy()
    if features.isna().any().any():
        raise ValueError("Dataset contains missing anomaly feature values")
    return features


@dataclass
class AnomalyDetector:
    """Persistable detector and the reference statistics needed for scoring."""

    pipeline: Pipeline
    threshold: float
    threshold_percentile: float
    reference_scores: np.ndarray
    numerical_reference_bounds: dict[str, tuple[float, float]]
    feature_names: tuple[str, ...] = ANOMALY_FEATURE_COLUMNS
    metadata: dict[str, int | float | str] = field(default_factory=dict)

    def score_samples(self, dataset: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores; larger values mean more atypical claims."""
        features = select_anomaly_features(dataset)
        return -np.asarray(self.pipeline.score_samples(features), dtype=float)

    def reference_percentiles(self, scores: np.ndarray) -> np.ndarray:
        """Fraction of reference claims scoring no higher than each new claim."""
        values = np.asarray(scores, dtype=float)
        ranks = np.searchsorted(self.reference_scores, values, side="right")
        return ranks / len(self.reference_scores)

    def _statistical_signals(self, row: pd.Series) -> list[str]:
        """Describe marginally rare values, not IsolationForest attributions."""
        signals = []
        for name, (low, high) in self.numerical_reference_bounds.items():
            value = float(row[name])
            if value < low:
                signals.append(f"{name}: very_low")
            elif value > high:
                signals.append(f"{name}: very_high")
        return signals

    def assess(
        self,
        claim: TriageFeatures | Mapping[str, object] | pd.Series | pd.DataFrame,
    ) -> AnomalyAssessment:
        """Score one claim and attach independent reference rarity signals."""
        if isinstance(claim, pd.DataFrame):
            if len(claim) != 1:
                raise ValueError("assess expects exactly one claim")
            frame = select_anomaly_features(claim)
        else:
            if isinstance(claim, TriageFeatures):
                payload = claim.model_dump(mode="json")
            elif isinstance(claim, pd.Series):
                payload = claim.to_dict()
            else:
                payload = dict(claim)
            frame = select_anomaly_features(pd.DataFrame([payload]))

        score = float(self.score_samples(frame)[0])
        is_anomalous = bool(score >= self.threshold)
        percentile = float(self.reference_percentiles(np.array([score]))[0])
        return AnomalyAssessment(
            is_anomalous=is_anomalous,
            anomaly_score=score,
            threshold=self.threshold,
            reference_percentile=percentile,
            statistical_signals=(
                self._statistical_signals(frame.iloc[0]) if is_anomalous else []
            ),
        )


def train_anomaly_detector(
    reference: pd.DataFrame,
    *,
    seed: int = 42,
    threshold_percentile: float = 0.98,
    n_estimators: int = 200,
) -> AnomalyDetector:
    """Fit solely on reference claims and calibrate a reference score quantile."""
    if not 0 < threshold_percentile < 1:
        raise ValueError("threshold_percentile must be between 0 and 1")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if "is_synthetic_anomaly" in reference.columns:
        labels = reference["is_synthetic_anomaly"].astype(str).str.strip().str.lower()
        if not labels.isin(("false", "0")).all():
            raise ValueError("Reference population must not contain injected anomalies")

    features = select_anomaly_features(reference)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination="auto",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(features)
    reference_scores = np.sort(
        -np.asarray(pipeline.score_samples(features), dtype=float)
    )
    threshold = float(np.quantile(reference_scores, threshold_percentile))
    numerical_reference_bounds = {
        name: (
            float(features[name].quantile(0.01)),
            float(features[name].quantile(0.99)),
        )
        for name in ANOMALY_NUMERICAL_FEATURES
    }
    return AnomalyDetector(
        pipeline=pipeline,
        threshold=threshold,
        threshold_percentile=threshold_percentile,
        reference_scores=reference_scores,
        numerical_reference_bounds=numerical_reference_bounds,
        metadata={
            "seed": seed,
            "reference_count": len(reference),
            "n_estimators": n_estimators,
            "score_convention": "higher_is_more_anomalous",
        },
    )
