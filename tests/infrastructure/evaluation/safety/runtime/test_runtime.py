"""Model reuse, memory measurements and serialized inference."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from evaluation.real.safety.telemetry import VisionSafetyStop
from tests.infrastructure.evaluation.safety.runtime._support import Adapter, runtime


def test_reuses_the_same_model_and_records_load_and_first_inference_peaks():
    instance, monitor, memory = runtime("0")
    adapter = Adapter()
    managed = instance.manage(adapter, name="damage")

    first = managed.assess_image(Path("one.jpg"))
    managed.assess_image(Path("two.jpg"))

    assert first.filename == "one.jpg"
    assert adapter.load_count == 1
    assert adapter.infer_count == 2
    assert adapter._model.device == "cuda:0"
    assert adapter._model.overrides == {"device": "cuda:0", "imgsz": 640, "batch": 1, "workers": 0, "augment": False}
    assert managed.confidence_threshold == 0.25
    assert managed.model_metadata is adapter.model_metadata
    assert adapter._model.predictor.results is None
    assert adapter._model.predictor.batch is None
    assert [event for event, _ in monitor.events] == [
        "model_load_start",
        "model_load_end",
        "inference_start",
        "inference_end",
        "inference_start",
        "inference_end",
    ]
    load_operations = ["sync", "snapshot"] * 2
    inference_operations = ["sync", "reset", "sync", "snapshot"] * 2
    assert memory.operations == load_operations + inference_operations
    details = monitor.events[3][1]
    assert details["memory"]["max_allocated_bytes"] == 30
    assert (details["width"], details["height"]) == (640, 480)
    assert "empty" not in memory.operations


def test_release_after_preflight_unloads_and_next_use_loads_again():
    instance, _, _ = runtime()
    adapter = Adapter()
    managed = instance.manage(adapter)
    assert managed.load() is adapter._model
    instance.release()
    assert adapter._model is None
    managed.assess_image(Path("one.jpg"))
    assert adapter.load_count == 2
    instance.close()
    with pytest.raises(VisionSafetyStop, match="already closed"):
        managed.load()


def test_serializes_inference_and_model_changes_between_threads():
    instance, _, _ = runtime()
    first, second = Adapter("damage"), Adapter("parts")
    entered, finish, next_requested = Event(), Event(), Event()
    original_assess = first.assess_image

    def blocking_inference(path):
        entered.set()
        assert finish.wait(timeout=5)
        return original_assess(path)

    first.assess_image = blocking_inference

    def next_inference():
        next_requested.set()
        return instance.manage(second).assess_image(Path("two.jpg"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(instance.manage(first).assess_image, Path("one.jpg"))
        assert entered.wait(timeout=5)
        two = pool.submit(next_inference)
        try:
            assert next_requested.wait(timeout=5)
            assert second.load_count == 0
            assert first._model is not None
        finally:
            finish.set()
        assert one.result(timeout=5).filename == "one.jpg"
        assert two.result(timeout=5).filename == "two.jpg"
    assert first._model is None
    assert second.load_count == 1
