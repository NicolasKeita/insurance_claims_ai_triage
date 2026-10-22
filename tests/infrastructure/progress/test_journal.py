"""Crash evidence is written before operations and retained across attempts."""

import json
from types import SimpleNamespace

import pytest

from evaluation.models import PipelineObservation
from evaluation.real.context import ComponentRecorder
from evaluation.real.progress.journal import ProgressJournal, atomic_json
from evaluation.real.progress.monitor import ScenarioProgress


def test_stage_start_is_durable_before_operation_and_checkpoint_records_results(tmp_path, monkeypatch):
    syncs = []
    monkeypatch.setattr("evaluation.real.progress.journal.os.fsync", lambda descriptor: syncs.append(descriptor))
    observation = PipelineObservation()
    recorder = ComponentRecorder()
    with ScenarioProgress(tmp_path, "unit", interval=60) as monitor:
        recorder.progress = monitor

        def operation():
            state = json.loads((monitor.journal.directory / "last_state.json").read_text())
            assert state["component"] == "document" and state["status"] == "RUNNING"
            assert len(syncs) >= 5
            return {"extracted": True}

        assert recorder._record(observation, "document", operation) == {"extracted": True}
        checkpoint = json.loads((monitor.journal.directory / "checkpoint.json").read_text())
        assert checkpoint["components"]["document"]["status"] == "PASS"
        assert checkpoint["trace"][-1]["outputs"] == {"extracted": True}
        monitor.complete(observation)
    events = [json.loads(line) for line in (monitor.journal.directory / "events.jsonl").read_text().splitlines()]
    assert [item["event"] for item in events] == [
        "scenario_started",
        "stage_started",
        "stage_finished",
        "scenario_finished",
    ]
    assert all(item["timestamp"] and item["pid"] and item["scenario_id"] == "unit" for item in events)
    assert not monitor.thread.is_alive()


def test_interrupt_records_active_stage_and_keeps_previous_completed_steps(tmp_path):
    observation = PipelineObservation()
    recorder = ComponentRecorder()
    with pytest.raises(KeyboardInterrupt):
        with ScenarioProgress(tmp_path, "unit", interval=60) as monitor:
            recorder.progress = monitor
            recorder._record(observation, "document", lambda: {"done": True})

            def interrupted():
                raise KeyboardInterrupt

            recorder._record(observation, "agent", interrupted)
    state = json.loads((monitor.journal.directory / "last_state.json").read_text())
    assert state["component"] == "agent" and state["status"] == "INTERRUPTED"
    checkpoint = json.loads((monitor.journal.directory / "checkpoint.json").read_text())
    assert checkpoint["components"]["document"]["status"] == "PASS"
    assert "agent" not in checkpoint["components"]
    assert not monitor.thread.is_alive()


def test_retry_preserves_previous_progress_and_failed_checkpoint_preserves_old_json(tmp_path):
    first = ProgressJournal(tmp_path, "unit")
    first.record({"event": "stage_started", "component": "agent", "status": "RUNNING"})
    first.close()
    second = ProgressJournal(tmp_path, "unit")
    second.close()
    assert first.directory != second.directory and (first.directory / "events.jsonl").is_file()
    target = tmp_path / "checkpoint.json"
    atomic_json(target, {"status": "PASS"})
    with pytest.raises(ValueError):
        atomic_json(target, {"invalid": float("nan")})
    assert json.loads(target.read_text()) == {"status": "PASS"}
    assert not target.with_suffix(".tmp").exists()


def test_failed_and_skipped_operations_keep_their_reasons(tmp_path):
    recorder, observation = ComponentRecorder(), PipelineObservation()
    recorder.config = SimpleNamespace(dependency_unavailable="SKIP")
    with ScenarioProgress(tmp_path, "unit", interval=60) as monitor:
        recorder.progress = monitor
        recorder._record(observation, "agent", lambda: 1 / 0)
        recorder._skip(observation, "vision", "Disabled by configuration")
    events = [json.loads(line) for line in (monitor.journal.directory / "events.jsonl").read_text().splitlines()]
    assert any(event.get("status") == "FAIL" and "ZeroDivisionError" in event["reason"] for event in events)
    assert events[-1]["status"] == "SKIP" and events[-1]["reason"] == "Disabled by configuration"
