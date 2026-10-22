"""Returned agent failures must agree across live progress and durable artifacts."""

import json
from types import SimpleNamespace

import pytest

from evaluation.models import PipelineObservation
from evaluation.real.context import ComponentRecorder
from evaluation.real.progress.monitor import ScenarioProgress
from evaluation.real.stages.agent import AgentSteps
from tests.agent.persistence._support import make_agent_run


class AgentRecorder(AgentSteps, ComponentRecorder):
    def similar_service(self):
        raise AssertionError("Unit tests must not load services")


@pytest.mark.parametrize("failed", [False, True])
def test_agent_terminal_result_is_classified_before_progress(tmp_path, monkeypatch, capsys, failed):
    result = make_agent_run(failed=failed)
    closed = []
    service = SimpleNamespace(tools=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr("agent.factory.create_claims_agent_service", lambda *args, **kwargs: service)
    recorder, observation = AgentRecorder(), PipelineObservation()
    recorder.resources = {"llm_model": "offline-model", "session_factory": object()}
    monkeypatch.setattr(recorder, "review", lambda *args: result)
    run = SimpleNamespace(case=SimpleNamespace(claim=SimpleNamespace(claim_id=result.claim_id)))
    with ScenarioProgress(tmp_path, "unit", interval=60) as monitor:
        recorder.progress = monitor
        recorder._run_agent(run, observation)
        monitor.complete(observation)
    expected = "FAIL" if failed else "PASS"
    assert observation.components["agent"].status == observation.trace[-1].status == expected
    assert observation.components["agent"].reason == result.error
    assert observation.trace[-1].outputs == observation.agent == result.model_dump(mode="json")
    assert observation.trace[-1].inputs == {"claim_id": result.claim_id, "source": "existing PostgreSQL claim"}
    assert observation.errors == ([f"agent: {result.error}"] if failed else [])
    _assert_progress(monitor, expected, result.error, capsys.readouterr().out)
    assert closed == [True]


def _assert_progress(monitor, expected, reason, output):
    events = [json.loads(line) for line in (monitor.journal.directory / "events.jsonl").read_text().splitlines()]
    finished = next(event for event in events if event["event"] == "stage_finished")
    assert finished["status"] == expected and finished["reason"] == reason
    assert events[-1]["component_statuses"]["agent"] == expected
    checkpoint = json.loads((monitor.journal.directory / "checkpoint.json").read_text())
    assert checkpoint["components"]["agent"]["status"] == checkpoint["trace"][-1]["status"] == expected
    assert checkpoint["components"]["agent"]["reason"] == reason
    assert f"[{expected}] agent:" in output
    assert f"[{'PASS' if expected == 'FAIL' else 'FAIL'}] agent:" not in output
    assert (f"Reason: {reason}" in output) == (expected == "FAIL")


def test_monitor_does_not_print_unexpected_agent_exception_text(tmp_path, capsys):
    recorder, observation = ComponentRecorder(), PipelineObservation()
    with ScenarioProgress(tmp_path, "unit", interval=60) as monitor:
        recorder.progress = monitor

        def broken_operation():
            raise RuntimeError("private provider details")

        recorder._record(observation, "agent", broken_operation)
    output = capsys.readouterr().out
    assert "[FAIL] agent:" in output and "private provider details" not in output
