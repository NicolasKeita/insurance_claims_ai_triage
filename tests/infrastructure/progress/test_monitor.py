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
    output = capsys.readouterr().out
    assert "Generate grounded answer with Ollama" in output and "GPU0 42%" in output
    assert "Previous recorded run: 62.5s (reference only; ETA unknown)" in output
    assert "step " in output and "total " in output and not monitor.thread.is_alive()


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
