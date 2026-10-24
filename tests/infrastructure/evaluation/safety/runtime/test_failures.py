"""CUDA failures and temperature guards stop work without retries."""

from pathlib import Path

import pytest

from evaluation.real.safety.telemetry import VisionSafetyStop
from tests.infrastructure.evaluation.safety.runtime._support import Adapter, Model, runtime


@pytest.mark.parametrize("operation", ["load", "infer"])
def test_wrapped_cuda_failure_latches_abort_and_never_retries(operation):
    instance, monitor, _ = runtime()
    adapter, following = Adapter(), Adapter("parts")

    def fail(*args):
        if operation == "load":
            adapter._model = Model()
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError as error:
            raise ValueError("Adapter could not execute") from error

    if operation == "load":
        adapter._load_model = fail
    else:
        adapter.assess_image = fail
    managed = instance.manage(adapter)
    with pytest.raises(VisionSafetyStop, match="Vision stopped"):
        managed.assess_image(Path("one.jpg"))
    if operation == "load":
        assert adapter._model is None
    with pytest.raises(VisionSafetyStop):
        instance.manage(following).assess_image(Path("two.jpg"))
    assert following.load_count == 0
    assert [event for event, _ in monitor.events].count("vision_abort") == 1
    instance.close()
    assert adapter._model is None


def test_failed_device_transfer_cleans_the_partially_cached_model():
    instance, _, _ = runtime("cuda:0")
    adapter = Adapter()
    original_load = adapter._load_model

    def load():
        model = original_load()
        model.to = lambda device: (_ for _ in ()).throw(RuntimeError("CUDA device unavailable"))
        return model

    adapter._load_model = load
    with pytest.raises(VisionSafetyStop):
        instance.manage(adapter).load()
    assert adapter._model is None


def test_thermal_stop_prevents_loading_and_cleanup_still_works():
    instance, monitor, _ = runtime()
    adapter = Adapter()
    managed = instance.manage(adapter)
    managed.load()
    monitor.reason = "GPU temperature reached the configured threshold"
    with pytest.raises(VisionSafetyStop, match="temperature"):
        managed.assess_image(Path("one.jpg"))
    assert adapter.infer_count == 0
    instance.close()
    assert adapter._model is None
    assert monitor.closed == 1


def test_checks_temperature_after_an_inference_before_returning_results():
    instance, monitor, _ = runtime()
    adapter = Adapter()
    original_assess = adapter.assess_image

    def hot_inference(path):
        result = original_assess(path)
        monitor.reason = "Temperature threshold reached"
        return result

    adapter.assess_image = hot_inference
    with pytest.raises(VisionSafetyStop, match="Temperature"):
        instance.manage(adapter).assess_image(Path("one.jpg"))
    assert adapter.infer_count == 1
    assert [event for event, _ in monitor.events][-1] == "inference_error"


def test_unrelated_adapter_failure_is_logged_and_propagated_without_retry():
    instance, monitor, _ = runtime()
    adapter = Adapter()

    def invalid_image(path):
        adapter.infer_count += 1
        raise ValueError("Image is invalid")

    adapter.assess_image = invalid_image
    with pytest.raises(ValueError, match="Image is invalid"):
        instance.manage(adapter).assess_image(Path("one.jpg"))
    assert adapter.infer_count == 1
    assert "vision_abort" not in [event for event, _ in monitor.events]
    assert monitor.events[-1][0] == "inference_error"
