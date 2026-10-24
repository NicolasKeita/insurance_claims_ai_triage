"""Check preflight lifecycle and recording guards without inference frameworks."""

import sys
from types import SimpleNamespace

import pytest

import evaluation.real.safety as safety
from evaluation.models import PipelineObservation
from evaluation.preflight.config import RealEvaluationConfig
from evaluation.preflight.session import PreflightSession
from evaluation.real.context import ComponentRecorder
from evaluation.real.safety.managed import ManagedDetector
from evaluation.real.safety.telemetry import VisionSafetyStop
from vision.damage.detector import UltralyticsDamageDetector
from vision.parts.detector import UltralyticsVehiclePartDetector


@pytest.fixture
def preflight_doubles(monkeypatch):
    events, models = [], []
    monitor = SimpleNamespace(
        start=lambda: events.append("monitor_start"),
        check=lambda: None,
        event=lambda name, **details: events.append(name),
        close=lambda: events.append("monitor_close"),
    )

    def load_model(adapter):
        model = SimpleNamespace(task="segment", names={0: "mock"}, overrides={}, predictor=None)
        model.to = lambda device: events.append("model_to:" + device)
        adapter._model = model
        adapter._metadata = SimpleNamespace(version="mock-v1", identifier=str(adapter.settings.model_path))
        models.append(model)
        return model

    monkeypatch.setattr(safety, "SafetyMonitor", lambda *args, **kwargs: monitor)
    for adapter in (UltralyticsDamageDetector, UltralyticsVehiclePartDetector):
        monkeypatch.setattr(adapter, "_load_model", load_model)
    for framework in ("torch", "ultralytics"):
        monkeypatch.setitem(sys.modules, framework, None)
    return events, models


@pytest.mark.parametrize("kind", ["damage", "part"])
def test_detector_preflight_uses_managed_cpu_settings_and_releases_weights(tmp_path, preflight_doubles, kind):
    events, models = preflight_doubles
    config = RealEvaluationConfig(vision_device="cpu", fixture_dir=tmp_path / "fixtures")
    session = PreflightSession(config)
    try:
        metadata = session.detector(kind)
        runtime = session.report.resources["vision_runtime"]
        managed = session.report.resources[kind + "_detector"]
        assert isinstance(managed, ManagedDetector)
        assert managed.runtime is runtime
        assert managed.adapter.settings.device == runtime.device == "cpu"
        assert managed.adapter.settings.inference_image_size == runtime.imgsz == 640
        assert managed.adapter._model is None
        assert runtime._active is None
        assert not runtime.memory.enabled
        assert models[0].overrides == {"device": "cpu", "imgsz": 640, "batch": 1, "workers": 0, "augment": False}
        assert metadata["task"] == "segment" and metadata["class_count"] == 1
        assert events.index("monitor_start") < events.index("model_to:cpu")
        assert events.index("model_load_end") < events.index("model_unload_start")
        assert events[-1] == "model_unload_end"
    finally:
        session.report.close()
    assert events[-1] == "monitor_close"
    assert "vision_runtime" not in session.report.resources


@pytest.mark.parametrize("stopped_before_operation", [True, False])
def test_component_record_checks_guard_before_and_after_operation(stopped_before_operation):
    calls = []
    state = {"stopped": stopped_before_operation}

    def check():
        calls.append("check")
        if state["stopped"]:
            raise VisionSafetyStop("Latched safety stop")

    def operation():
        calls.append("operation")
        state["stopped"] = True
        return {"finished": True}

    recorder = ComponentRecorder()
    recorder.resources = {"vision_runtime": SimpleNamespace(monitor=SimpleNamespace(check=check))}
    observation = PipelineObservation()
    with pytest.raises(VisionSafetyStop, match="Latched safety stop"):
        recorder._record(observation, "vision", operation)
    assert calls == (["check"] if stopped_before_operation else ["check", "operation", "check"])
    assert observation.components == {}
    assert observation.trace == []
    assert observation.errors == []
