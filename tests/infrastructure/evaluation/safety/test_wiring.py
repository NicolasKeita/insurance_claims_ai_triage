"""Validate safety settings and stop propagation without models or scenarios."""

from types import SimpleNamespace

import pytest

from evaluation import E2E_BENCHMARK_VERSION
from evaluation.models import PipelineObservation
from evaluation.preflight.config import RealEvaluationConfig
from evaluation.preflight.report import PreflightReport
from evaluation.preflight.session import PreflightSession
from evaluation.real.context import ComponentRecorder
from evaluation.real.safety.telemetry import VisionSafetyStop


@pytest.mark.parametrize("image_size", [32, 320, 640])
def test_image_size_accepts_bounded_multiples_of_32(image_size):
    assert RealEvaluationConfig(vision_image_size=image_size).vision_image_size == image_size


@pytest.mark.parametrize("image_size", [0, 31, 33, 641, 672, True, 320.0])
def test_image_size_rejects_values_outside_the_safe_contract(image_size):
    with pytest.raises(ValueError, match="vision_image_size"):
        RealEvaluationConfig(vision_image_size=image_size)


@pytest.mark.parametrize("temperature", [0.5, 80, 80.5])
def test_temperature_accepts_finite_positive_values(temperature):
    assert RealEvaluationConfig(vision_max_temperature_c=temperature).vision_max_temperature_c == temperature


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf"), -float("inf"), True])
def test_temperature_rejects_nonpositive_or_nonfinite_values(temperature):
    with pytest.raises(ValueError, match="vision_max_temperature_c"):
        RealEvaluationConfig(vision_max_temperature_c=temperature)


def test_toml_loads_vision_safety_and_explicit_overrides_take_precedence(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(
        f'benchmark_version = "{E2E_BENCHMARK_VERSION}"\n'
        "[vision_safety]\nvision_image_size = 320\nvision_max_temperature_c = 75.5\n",
        encoding="utf-8",
    )
    configured = RealEvaluationConfig.from_toml(path)
    assert configured.vision_image_size == 320
    assert configured.vision_max_temperature_c == 75.5
    overridden = RealEvaluationConfig.from_toml(path, vision_image_size=512, vision_max_temperature_c=78)
    assert overridden.vision_image_size == 512
    assert overridden.vision_max_temperature_c == 78
    retained = RealEvaluationConfig.from_toml(path, vision_image_size=None, vision_max_temperature_c=None)
    assert retained.vision_image_size == configured.vision_image_size
    assert retained.vision_max_temperature_c == configured.vision_max_temperature_c


def test_toml_without_safety_section_retains_prudent_defaults(tmp_path):
    path = tmp_path / "settings.toml"
    path.write_text(f'benchmark_version = "{E2E_BENCHMARK_VERSION}"\n', encoding="utf-8")
    config = RealEvaluationConfig.from_toml(path)
    assert config.vision_image_size == 640
    assert config.vision_max_temperature_c == 80


@pytest.mark.parametrize("dependency_policy", ["SKIP", "FAIL"])
def test_preflight_safety_stop_is_never_converted_to_a_dependency_result(dependency_policy):
    progress = []
    session = PreflightSession(RealEvaluationConfig(dependency_unavailable=dependency_policy), progress.append)
    stop = VisionSafetyStop("Temperature guard requested an immediate stop")

    def guarded_operation():
        raise stop

    with pytest.raises(VisionSafetyStop) as caught:
        session.probe("damage_vision", guarded_operation)
    assert caught.value is stop
    assert session.report.rows == []
    assert progress == []


def test_component_recorder_does_not_swallow_a_safety_stop_or_report_completion():
    recorder = ComponentRecorder()
    progress = []
    recorder.progress = lambda component, status, observation, **details: progress.append((component, status))
    observation = PipelineObservation()
    stop = VisionSafetyStop("GPU memory guard stopped inference")

    def guarded_operation():
        raise stop

    with pytest.raises(VisionSafetyStop) as caught:
        recorder._record(observation, "vision", guarded_operation)
    assert caught.value is stop
    assert progress == [("vision", "RUNNING")]
    assert observation.components == {}
    assert observation.trace == []
    assert observation.errors == []


def test_preflight_report_closes_vision_runtime_before_service_resources():
    closed = []
    report = PreflightReport(
        resources={
            "vision_runtime": SimpleNamespace(close=lambda: closed.append("vision")),
            "database_engine": SimpleNamespace(dispose=lambda: closed.append("database")),
            "qdrant_client": SimpleNamespace(close=lambda: closed.append("qdrant")),
        }
    )
    report.close()
    assert closed == ["vision", "database", "qdrant"]
    assert "vision_runtime" not in report.resources
