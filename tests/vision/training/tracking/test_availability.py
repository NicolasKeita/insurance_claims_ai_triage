"""Test availability."""

import sys
from dataclasses import replace
from types import ModuleType

import pytest

from vision.tracking import vision_tracking_run
from vision.training import load_training_config


def test_optional_tracking_without_uri_does_not_block_cv(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    config = replace(load_training_config(), track_mlflow=True)
    with pytest.warns(RuntimeWarning, match="not configured"):
        with vision_tracking_run(config) as tracker:
            assert tracker is None


def test_optional_server_failure_does_not_swallow_training_failure(monkeypatch):
    class StubMlflowException(Exception):
        pass

    def unavailable(_):
        raise StubMlflowException("offline stub server")

    fake = ModuleType("mlflow")
    fake.set_experiment = unavailable
    exceptions = ModuleType("mlflow.exceptions")
    exceptions.MlflowException = StubMlflowException
    configuration = ModuleType("ml.tracking")
    configuration.configure_mlflow_tracking = lambda: "http://offline.invalid"
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.exceptions", exceptions)
    monkeypatch.setitem(sys.modules, "ml.tracking", configuration)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://offline.invalid")
    with pytest.warns(RuntimeWarning, match="initialization failed"):
        with pytest.raises(RuntimeError, match="actual training failure"):
            with vision_tracking_run(load_training_config()) as tracker:
                assert tracker is None
                raise RuntimeError("actual training failure")
