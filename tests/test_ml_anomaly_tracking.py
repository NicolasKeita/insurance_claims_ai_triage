import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ml.anomaly_evaluation import AnomalyMetrics
from ml.anomaly_registry import (
    ANOMALY_EXPERIMENT_NAME,
    anomaly_candidate_source_from_run,
    find_latest_anomaly_candidate_source,
)
from ml.anomaly_tracking import (
    mlflow_anomaly_metrics,
    read_dataset_metadata,
    verify_anomaly_model_round_trip,
)


def make_run(*, status="FINISHED", task="novelty_detection", model_uri="models:/m-1"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id="run-1", status=status),
        data=SimpleNamespace(
            tags={"task": task, "candidate.model_uri": model_uri},
            params={
                "dataset.reference.version": "anomaly_reference_v1",
                "dataset.reference.sha256": "a" * 64,
                "dataset.evaluation.version": "anomaly_evaluation_v1",
                "dataset.evaluation.sha256": "b" * 64,
                "threshold.percentile": "0.98",
                "threshold.value": "0.616",
            },
        ),
    )


def test_candidate_lineage_requires_a_finished_novelty_run():
    source = anomaly_candidate_source_from_run(make_run())
    assert source.model_uri == "models:/m-1"
    assert source.reference_version == "anomaly_reference_v1"
    assert source.threshold_value == "0.616"

    with pytest.raises(ValueError, match="not finished"):
        anomaly_candidate_source_from_run(make_run(status="FAILED"))
    with pytest.raises(ValueError, match="not a novelty"):
        anomaly_candidate_source_from_run(make_run(task="classification"))


def test_candidate_search_pages_past_ineligible_runs():
    class Page(list):
        def __init__(self, runs, token):
            super().__init__(runs)
            self.token = token

    class Client:
        def get_experiment_by_name(self, name):
            assert name == ANOMALY_EXPERIMENT_NAME
            return SimpleNamespace(experiment_id="4")

        def search_runs(self, *, page_token, **kwargs):
            assert kwargs["experiment_ids"] == ["4"]
            if page_token is None:
                return Page([make_run(status="FAILED")], "next")
            assert page_token == "next"
            return Page([make_run()], None)

    assert find_latest_anomaly_candidate_source(Client()).run_id == "run-1"


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


def test_round_trip_detects_score_change():
    class Detector:
        def assess(self, row):
            return SimpleNamespace(
                is_anomalous=False,
                anomaly_score=0.4,
                threshold=0.6,
                reference_percentile=0.5,
                statistical_signals=[],
            )

    class Model:
        def __init__(self, score):
            self.score = score

        def predict(self, frame):
            return pd.DataFrame(
                {
                    "is_anomalous": [False],
                    "anomaly_score": [self.score],
                    "threshold": [0.6],
                    "reference_percentile": [0.5],
                    "statistical_signals": ["[]"],
                }
            )

    from ml.anomaly_training import ANOMALY_FEATURE_COLUMNS

    examples = pd.DataFrame([dict.fromkeys(ANOMALY_FEATURE_COLUMNS, 0)])
    assert np.isclose(
        verify_anomaly_model_round_trip(Detector(), Model(0.4), examples)[
            "anomaly_score"
        ].iloc[0],
        0.4,
    )
    with pytest.raises(RuntimeError, match="anomaly_score changed"):
        verify_anomaly_model_round_trip(Detector(), Model(0.5), examples)
