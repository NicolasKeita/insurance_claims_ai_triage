"""Test dataset."""

import numpy as np
import pandas as pd

from ml.anomaly_dataset import ANOMALY_KINDS, generate_anomaly_datasets
from ml.features import TriageFeatures
from tests.intelligence.ml.anomaly.detector._support import anomaly_datasets
from tests.intelligence.ml.anomaly.detector._support import anomaly_datasets as anomaly_datasets
from tests.intelligence.ml.anomaly.detector._support import anomaly_detector as anomaly_detector


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
    assert set(evaluation.loc[~evaluation["is_synthetic_anomaly"], "anomaly_kind"]) == {"NONE"}


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
        assert (dataset["missing_document_count"] <= dataset["document_count"]).all()
        assert np.allclose(
            dataset["document_completeness_ratio"],
            (dataset["document_count"] - dataset["missing_document_count"]) / dataset["document_count"],
        )
        for features in dataset[feature_columns].to_dict(orient="records"):
            TriageFeatures.model_validate(features)
