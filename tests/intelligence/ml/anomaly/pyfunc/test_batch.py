"""Test batch."""

import json

import pytest

from ml.anomaly_dataset import generate_anomaly_datasets
from ml.anomaly_pyfunc import AnomalyPyfuncModel
from ml.anomaly_training import train_anomaly_detector


def test_pyfunc_uses_complete_detector_for_every_row() -> None:
    reference, evaluation = generate_anomaly_datasets(
        reference_count=100,
        evaluation_count=12,
        anomaly_fraction=0.25,
        seed=42,
    )
    detector = train_anomaly_detector(reference, n_estimators=20)
    model = AnomalyPyfuncModel(detector)
    inputs = evaluation.iloc[[1, 4, 7]].copy()
    result = model.predict(None, inputs)

    assert result.index.equals(inputs.index)
    assert list(result.columns) == [
        "is_anomalous",
        "anomaly_score",
        "threshold",
        "reference_percentile",
        "statistical_signals",
    ]
    for position, (_, claim) in enumerate(inputs.iterrows()):
        expected = detector.assess(claim)
        actual = result.iloc[position]
        assert bool(actual["is_anomalous"]) == expected.is_anomalous
        assert actual["anomaly_score"] == pytest.approx(expected.anomaly_score)
        assert actual["threshold"] == pytest.approx(expected.threshold)
        assert actual["reference_percentile"] == pytest.approx(expected.reference_percentile)
        assert json.loads(actual["statistical_signals"]) == expected.statistical_signals
