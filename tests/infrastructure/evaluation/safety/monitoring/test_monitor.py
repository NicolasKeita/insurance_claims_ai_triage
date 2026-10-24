"""Durability, synchronous samples and stale telemetry without NVIDIA or CUDA."""

import csv
from datetime import datetime, timezone

import pytest

from evaluation.real.safety import journal, telemetry
from evaluation.real.safety.telemetry import SafetyMonitor, VisionSafetyStop
from tests.infrastructure.evaluation.safety.monitoring._support import events, gpu_sample


def test_cpu_never_reads_nvidia_and_events_are_durable_append_only(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(journal.os, "fsync", lambda descriptor: synced.append(descriptor))

    def forbidden():
        raise AssertionError("CPU execution must not query a GPU")

    for index in range(2):
        monitor = SafetyMonitor(tmp_path, gpu_enabled=False, reader=forbidden).start()
        monitor.event("inference_started", image=f"image-{index}.png", width=640, height=480)
        monitor.check()
        assert events(tmp_path)[-1]["image"] == f"image-{index}.png"
        monitor.close()
        monitor.close()
    saved = events(tmp_path)
    assert [event["event"] for event in saved] == ["monitor_started", "inference_started"] * 2
    assert saved[0]["run_id"] != saved[2]["run_id"]
    assert all(datetime.fromisoformat(event["timestamp"]).tzinfo == timezone.utc for event in saved)
    assert len(synced) == 4
    assert not (tmp_path / "gpu.csv").exists()


def test_first_sample_is_written_before_start_returns_and_cache_is_a_copy(tmp_path):
    calls = []

    def reader():
        calls.append(True)
        return gpu_sample()

    monitor = SafetyMonitor(tmp_path, interval_seconds=60, reader=reader).start()
    try:
        assert calls == [True]
        with (tmp_path / "gpu.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert rows[0]["run_id"] == monitor.run_id
        assert rows[0]["temperature_c"] == "55"
        assert rows[0]["power_w"] == ""
        cached = monitor.last_sample()
        cached["devices"][0]["temperature_c"] = 999
        assert monitor.last_sample()["devices"][0]["temperature_c"] == 55
    finally:
        monitor.close()


def test_stale_gpu_sample_stops_without_waiting_in_real_time(tmp_path, monkeypatch):
    now = [0]
    monkeypatch.setattr(telemetry, "monotonic", lambda: now[0])
    monitor = SafetyMonitor(tmp_path, interval_seconds=60, reader=gpu_sample).start()
    try:
        now[0] = 181
        with pytest.raises(VisionSafetyStop, match="stopped updating"):
            monitor.check()
        assert events(tmp_path)[-1]["event"] == "safety_stop"
    finally:
        monitor.close()


@pytest.mark.parametrize("kwargs", [{"max_temperature_c": 0}, {"interval_seconds": 0}])
def test_invalid_guard_configuration_is_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        SafetyMonitor(tmp_path, **kwargs)
