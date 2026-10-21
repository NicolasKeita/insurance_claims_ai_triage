"""Test candidates."""

from types import SimpleNamespace

import pytest

from ml.anomaly_registry import (
    ANOMALY_EXPERIMENT_NAME,
    anomaly_candidate_source_from_run,
    find_latest_anomaly_candidate_source,
)
from tests.ml.anomaly.tracking._support import make_run


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
