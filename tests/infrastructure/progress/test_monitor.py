"""Heartbeats and scoped callback attachment using trivial offline operations."""

import json
from threading import Event
from types import SimpleNamespace

import pytest

from evaluation.models import PipelineObservation
from evaluation.real.progress import run_with_progress
from evaluation.real.progress.monitor import ScenarioProgress


def test_heartbeat_reports_current_stage_and_saves_gpu_sample(tmp_path, monkeypatch, capsys):
    saved = Event()
    sample = {"available": True, "devices": [{"index": 0, "utilization_percent": 42}]}
    monkeypatch.setattr("evaluation.real.progress.monitor.read_gpu", lambda: sample)
    with ScenarioProgress(tmp_path, "unit", reference={"total": 62.5}, interval=0.01) as monitor:
        record = monitor.journal.record

        def remember(event):
            record(event)
            if event["event"] == "heartbeat":
                saved.set()

        monkeypatch.setattr(monitor.journal, "record", remember)
        monitor("rag_generation", "RUNNING", PipelineObservation())
        assert saved.wait(2), "A long operation must produce heartbeat output"
        monitor.complete(PipelineObservation())
        assert monitor.current is None
    events = [json.loads(line) for line in (monitor.journal.directory / "events.jsonl").read_text().splitlines()]
    heartbeat = next(item for item in events if item["event"] == "heartbeat")
    assert events[-1]["event"] == "scenario_finished"
    assert heartbeat["component"] == "rag_generation" and heartbeat["gpu"] == sample
    assert heartbeat["stage_elapsed_seconds"] >= 0
    assert heartbeat["estimated_remaining_seconds"] == pytest.approx(62.5 - heartbeat["elapsed_seconds"])
    assert heartbeat["eta_reason"] is None and heartbeat["reference_total_seconds"] == 62.5
    assert events[-1]["estimated_remaining_seconds"] == 0 and events[-1]["eta_reason"] is None
    output = capsys.readouterr().out
    assert "Generate grounded answer with Ollama" in output and "GPU0 42%" in output
    assert "Previous recorded run: 62.5s" in output
    assert f"ETA ~{heartbeat['estimated_remaining_seconds']:.1f}s remaining (previous run)" in output
    assert "step " in output and "total " in output and not monitor.thread.is_alive()


def test_terminal_and_journal_keep_the_same_eta_as_stages_finish_or_exceed_reference(tmp_path, monkeypatch, capsys):
    clock = [100.0]
    monkeypatch.setattr("evaluation.real.progress.monitor.perf_counter", lambda: clock[0])
    monitor = ScenarioProgress(
        tmp_path, "unit", reference={"total": 100, "document": 20, "rag_generation": 30, "agent": 40}
    )
    monkeypatch.setattr(monitor.thread, "start", lambda: None)
    monkeypatch.setattr(monitor.thread, "join", lambda **kwargs: None)
    observation = PipelineObservation()

    def heartbeat_at(now):
        clock[0] = now
        waits = iter((False, True))
        monkeypatch.setattr(monitor.stop, "wait", lambda interval: next(waits))
        monitor._heartbeat()

    with monitor:
        clock[0] = 110
        monitor("document", "RUNNING", observation)
        heartbeat_at(115)
        clock[0] = 135
        monitor("document", "PASS", observation, duration_seconds=25)
        clock[0] = 140
        monitor("rag_generation", "SKIP", observation, duration_seconds=0)
        clock[0] = 150
        monitor("agent", "RUNNING", observation)
        heartbeat_at(195)
        clock[0] = 200
        monitor("agent", "PASS", observation, duration_seconds=50)
        monitor.complete(observation)

    _assert_eta_updates(monitor, capsys.readouterr().out.splitlines())


def _assert_eta_updates(monitor, output):
    events = [json.loads(line) for line in (monitor.journal.directory / "events.jsonl").read_text().splitlines()]
    assert [item["estimated_remaining_seconds"] for item in events] == [100, 100, 95, 80, 50, 50, None, 10, 0]
    assert all(item["reference_total_seconds"] == 100 for item in events)
    assert [item["eta_reason"] for item in events] == [None] * 6 + ["reference_exceeded", None, None]
    heartbeats = [item for item in events if item["event"] == "heartbeat"]
    assert [item["stage_elapsed_seconds"] for item in heartbeats] == [5, 45]
    assert [item["elapsed_seconds"] for item in heartbeats] == [15, 95]

    document_heartbeat = next(line for line in output if "[RUNNING] document | step" in line)
    assert "step 5.0s | total 15.0s | ETA ~95.0s remaining (previous run)" in document_heartbeat
    document_finished = next(line for line in output if "[PASS] document:" in line)
    assert "ETA ~80.0s remaining (previous run)" in document_finished
    skipped = next(line for line in output if "[SKIP] rag_generation:" in line)
    assert "ETA ~50.0s remaining (previous run)" in skipped
    exceeded = next(line for line in output if "[RUNNING] agent | step" in line)
    assert "step 45.0s | total 95.0s | ETA unknown" in exceeded
    assert "~0.0s" not in exceeded


@pytest.mark.parametrize("interrupted", [False, True])
def test_wrapper_restores_callback_and_keeps_log_on_success_or_interrupt(tmp_path, interrupted):
    original = object()

    def run(scenario):
        pipeline.progress("agent", "RUNNING", PipelineObservation())
        if interrupted:
            raise KeyboardInterrupt
        return PipelineObservation()

    pipeline = SimpleNamespace(progress=original, resources={}, run=run)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):
            run_with_progress(pipeline, SimpleNamespace(scenario_id="unit"), tmp_path)
    else:
        assert isinstance(
            run_with_progress(pipeline, SimpleNamespace(scenario_id="unit"), tmp_path), PipelineObservation
        )
    assert pipeline.progress is original
    path = next((tmp_path / "progress/unit").glob("*/last_state.json"))
    assert json.loads(path.read_text())["status"] == ("INTERRUPTED" if interrupted else "FINISHED")
