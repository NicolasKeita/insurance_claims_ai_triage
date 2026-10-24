"""Temperature, unavailable telemetry and disk failures stop further GPU work."""

from threading import Event

import pytest

from evaluation.real.safety.telemetry import SafetyMonitor, VisionSafetyStop
from tests.infrastructure.evaluation.safety.monitoring._support import broken_disk, events, gpu_sample


@pytest.mark.parametrize(
    "sample",
    [
        {"available": False, "reason": "nvidia-smi not found"},
        {"available": True, "devices": []},
        gpu_sample(None),
        gpu_sample(float("nan")),
        gpu_sample(80),
    ],
)
def test_unavailable_or_hot_initial_gpu_stops_before_work_and_remains_latched(tmp_path, sample):
    monitor = SafetyMonitor(tmp_path, reader=lambda: sample)
    with pytest.raises(VisionSafetyStop):
        monitor.start()
    with pytest.raises(VisionSafetyStop):
        monitor.check()
    assert events(tmp_path)[-1]["event"] == "safety_stop"
    monitor.close()


@pytest.mark.parametrize("failure", [gpu_sample(81), {"available": False, "reason": "probe failed"}])
def test_worker_latches_temperature_or_lost_telemetry_until_caller_checks(tmp_path, failure):
    second_read = Event()
    calls = []

    def reader():
        calls.append(True)
        if len(calls) == 1:
            return gpu_sample()
        second_read.set()
        return failure

    monitor = SafetyMonitor(tmp_path, interval_seconds=0.001, reader=reader).start()
    try:
        assert second_read.wait(1)
        assert monitor._stop.wait(1)
        monitor._thread.join(timeout=1)
        for _ in range(2):
            with pytest.raises(VisionSafetyStop):
                monitor.check()
        assert calls == [True, True]
        assert events(tmp_path)[-1]["event"] == "safety_stop"
    finally:
        monitor.close()


def test_event_disk_failure_stops_before_the_next_operation(tmp_path, monkeypatch):
    monitor = SafetyMonitor(tmp_path, gpu_enabled=False).start()
    monkeypatch.setattr(monitor, "_flush", broken_disk)
    try:
        with pytest.raises(VisionSafetyStop, match="disk full"):
            monitor.event("model_load_started", model="fake")
        with pytest.raises(VisionSafetyStop, match="disk full"):
            monitor.check()
    finally:
        monitor.close()


def test_worker_disk_failure_is_latched_and_closes_cleanly(tmp_path, monkeypatch):
    release = Event()
    calls = []

    def reader():
        calls.append(True)
        if len(calls) > 1:
            assert release.wait(1)
        return gpu_sample()

    monitor = SafetyMonitor(tmp_path, interval_seconds=0.001, reader=reader).start()
    monkeypatch.setattr(monitor, "_flush", broken_disk)
    release.set()
    try:
        assert monitor._stop.wait(1)
        with pytest.raises(VisionSafetyStop, match="disk full"):
            monitor.check()
    finally:
        monitor.close()
    assert not monitor._thread.is_alive()


def test_close_attempts_both_streams_even_when_one_disk_flush_fails(tmp_path):
    monitor = SafetyMonitor(tmp_path, gpu_enabled=False).start()

    class BrokenStream:
        def close(self):
            raise OSError("disk full")

    monitor._gpu = BrokenStream()
    with pytest.raises(VisionSafetyStop, match="disk full"):
        monitor.close()
    assert monitor._events.closed
    monitor.close()
