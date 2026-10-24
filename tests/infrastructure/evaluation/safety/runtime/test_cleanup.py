"""Model disposal remains effective after framework or disk faults."""

import weakref
from pathlib import Path

from evaluation.real.safety.telemetry import VisionSafetyStop
from tests.infrastructure.evaluation.safety.runtime._support import Adapter, runtime


def test_switch_frees_previous_model_and_predictor_before_loading_the_next():
    instance, monitor, memory = runtime()
    previous, following = Adapter("damage"), Adapter("parts")
    managed = instance.manage(previous)
    managed.load()
    model_ref = weakref.ref(previous._model)
    predictor = previous._model.predictor
    original_load = following._load_model

    def load_next():
        assert previous._model is None
        assert model_ref() is None
        assert predictor.model is None
        assert predictor.results is None
        assert predictor.batch is None
        return original_load()

    following._load_model = load_next
    instance.manage(following).assess_image(Path("one.jpg"))
    assert memory.operations.count("empty") == 1
    events = [event for event, _ in monitor.events]
    assert events.index("model_unload_end") < events.index("model_load_start", 1)
    instance.close()
    instance.close()
    assert following._model is None
    assert monitor.closed == 1
    assert memory.operations.count("empty") == 2


def test_close_preserves_the_first_fault_if_cuda_cleanup_also_fails():
    instance, monitor, memory = runtime("cuda:0")
    adapter = Adapter()
    instance.manage(adapter).load()

    def broken_cache():
        raise RuntimeError("CUDA context is no longer available")

    memory.empty_cache = broken_cache
    instance.close()
    assert adapter._model is None
    assert monitor.closed == 1
    assert "model_unload_error" in [event for event, _ in monitor.events]


def test_cleanup_releases_model_even_if_the_event_log_disk_fails():
    instance, monitor, memory = runtime()
    adapter = Adapter()
    instance.manage(adapter).load()
    model_ref = weakref.ref(adapter._model)

    def unavailable_log(*args, **kwargs):
        raise VisionSafetyStop("Event log disk is unavailable")

    monitor.event = unavailable_log
    instance.close()
    assert adapter._model is None
    assert model_ref() is None
    assert "empty" in memory.operations
    assert monitor.closed == 1
