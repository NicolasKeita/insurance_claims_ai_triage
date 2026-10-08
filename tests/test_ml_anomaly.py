import numpy as np
import pandas as pd
import pytest

from ml.anomaly_dataset import ANOMALY_KINDS, generate_anomaly_datasets
from ml.anomaly_evaluation import evaluate_anomaly_detector
from ml.anomaly_training import (
    ANOMALY_FEATURE_COLUMNS,
    AnomalyAssessment,
    select_anomaly_features,
    train_anomaly_detector,
)
from ml.features import TriageFeatures


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


def test_anomaly_dataset_is_reproducible(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    expected_reference, expected_evaluation = anomaly_datasets
    reference, evaluation = generate_anomaly_datasets(
        reference_count=300,
        evaluation_count=160,
        anomaly_fraction=0.25,
        seed=42,
    )

    pd.testing.assert_frame_equal(reference, expected_reference)
    pd.testing.assert_frame_equal(evaluation, expected_evaluation)


def test_anomaly_labels_are_evaluation_only(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, evaluation = anomaly_datasets

    assert "is_synthetic_anomaly" not in reference.columns
    assert "anomaly_kind" not in reference.columns
    assert set(evaluation["is_synthetic_anomaly"].unique()) == {False, True}
    assert set(evaluation.loc[~evaluation["is_synthetic_anomaly"], "anomaly_kind"]) == {
        "NONE"
    }


def test_all_synthetic_anomaly_families_are_present(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    _, evaluation = anomaly_datasets
    injected = evaluation.loc[evaluation["is_synthetic_anomaly"]]

    assert set(injected["anomaly_kind"]) == set(ANOMALY_KINDS)


def test_anomaly_examples_respect_feature_bounds(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    reference, evaluation = anomaly_datasets
    feature_columns = list(TriageFeatures.model_fields)

    for dataset in (reference, evaluation):
        assert np.isfinite(dataset["repair_amount"]).all()
        assert (
            dataset["missing_document_count"] <= dataset["document_count"]
        ).all()
        assert np.allclose(
            dataset["document_completeness_ratio"],
            (dataset["document_count"] - dataset["missing_document_count"])
            / dataset["document_count"],
        )
        for features in dataset[feature_columns].to_dict(orient="records"):
            TriageFeatures.model_validate(features)


def test_training_features_exclude_outcomes_and_synthetic_labels(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
    anomaly_detector,
) -> None:
    reference, evaluation = anomaly_datasets
    with_outcomes = evaluation.assign(workflow="STANDARD")
    selected = select_anomaly_features(with_outcomes)

    assert list(selected.columns) == list(ANOMALY_FEATURE_COLUMNS)
    assert set(selected.columns).isdisjoint(
        {"workflow", "is_synthetic_anomaly", "anomaly_kind"}
    )
    assert list(anomaly_detector.pipeline.feature_names_in_) == list(
        ANOMALY_FEATURE_COLUMNS
    )
    assert list(reference.columns) == list(ANOMALY_FEATURE_COLUMNS)


def test_higher_anomaly_score_means_more_atypical(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
    anomaly_detector,
) -> None:
    reference, _ = anomaly_datasets
    ordinary = reference.iloc[:12]
    extreme = extreme_cases(reference)

    for features in extreme.to_dict(orient="records"):
        TriageFeatures.model_validate(features)

    assert anomaly_detector.score_samples(extreme).mean() > (
        anomaly_detector.score_samples(ordinary).mean()
    )


def test_threshold_and_assessment_follow_score_convention(
    anomaly_datasets: tuple[pd.DataFrame, pd.DataFrame],
    anomaly_detector,
) -> None:
    reference, _ = anomaly_datasets
    reference_scores = anomaly_detector.score_samples(reference)
    extreme = extreme_cases(reference).iloc[0]
    assessment = anomaly_detector.assess(extreme)

    assert np.isfinite(anomaly_detector.threshold)
    assert anomaly_detector.threshold == pytest.approx(
        float(np.quantile(reference_scores, 0.98))
    )
    assert isinstance(assessment, AnomalyAssessment)
    assert assessment.anomaly_score == pytest.approx(
        float(anomaly_detector.score_samples(extreme.to_frame().T)[0])
    )
    assert assessment.threshold == pytest.approx(anomaly_detector.threshold)
    assert assessment.is_anomalous == (
        assessment.anomaly_score >= assessment.threshold
    )
    assert 0 <= assessment.reference_percentile <= 1
    assert assessment.is_anomalous
    assert "repair_amount: very_high" in assessment.statistical_signals

    ordinary = reference.iloc[int(np.argmin(reference_scores))]
    ordinary_assessment = anomaly_detector.assess(ordinary)
    assert not ordinary_assessment.is_anomalous
    assert ordinary_assessment.statistical_signals == []


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
    assert metrics.flagged_rate == pytest.approx(
        metrics.flagged_count / len(evaluation)
    )
    assert set(ANOMALY_KINDS) <= set(metrics.by_anomaly_kind)
    assert {"error_type", "anomaly_kind"} <= set(result.errors.columns)
    assert set(result.errors["error_type"]) <= {
        "false_positive",
        "false_negative",
    }
