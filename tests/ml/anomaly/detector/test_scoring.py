"""Test scoring."""

import numpy as np
import pandas as pd
import pytest

from ml.anomaly_training import (
    AnomalyAssessment,
)
from ml.features import TriageFeatures
from tests.ml.anomaly.detector._support import (
    anomaly_datasets,
    anomaly_detector,
    extreme_cases,
)
from tests.ml.anomaly.detector._support import anomaly_datasets as anomaly_datasets
from tests.ml.anomaly.detector._support import anomaly_detector as anomaly_detector


def test_higher_anomaly_score_means_more_atypical(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
    anomaly_detector,
) -> None:
    reference, _ = anomaly_datasets
    ordinary = reference.iloc[:12]
    extreme = extreme_cases(reference)

    for features in extreme.to_dict(orient="records"):
        TriageFeatures.model_validate(features)

    assert anomaly_detector.score_samples(extreme).mean() > (anomaly_detector.score_samples(ordinary).mean())


def test_threshold_and_assessment_follow_score_convention(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
    anomaly_detector,
) -> None:
    reference, _ = anomaly_datasets
    reference_scores = anomaly_detector.score_samples(reference)
    extreme = extreme_cases(reference).iloc[0]
    assessment = anomaly_detector.assess(extreme)

    assert np.isfinite(anomaly_detector.threshold)
    assert anomaly_detector.threshold == pytest.approx(float(np.quantile(reference_scores, 0.98)))
    assert isinstance(assessment, AnomalyAssessment)
    assert assessment.anomaly_score == pytest.approx(float(anomaly_detector.score_samples(extreme.to_frame().T)[0]))
    assert assessment.threshold == pytest.approx(anomaly_detector.threshold)
    assert assessment.is_anomalous == (assessment.anomaly_score >= assessment.threshold)
    assert 0 <= assessment.reference_percentile <= 1
    assert assessment.is_anomalous
    assert "repair_amount: very_high" in assessment.statistical_signals

    ordinary = reference.iloc[int(np.argmin(reference_scores))]
    ordinary_assessment = anomaly_detector.assess(ordinary)
    assert not ordinary_assessment.is_anomalous
    assert ordinary_assessment.statistical_signals == []
