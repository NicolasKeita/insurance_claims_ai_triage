"""A child process exits abruptly: no E2E, model, service or GPU workload."""

import json
import subprocess
import sys

SCRIPT = """
import os, sys
from evaluation.models import PipelineObservation
from evaluation.real.context import ComponentRecorder
from evaluation.real.progress.monitor import ScenarioProgress
observation = PipelineObservation()
recorder = ComponentRecorder()
with ScenarioProgress(sys.argv[1], 'abrupt_exit', interval=60) as monitor:
    recorder.progress = monitor
    recorder._record(observation, 'document', lambda: {'done': True})
    recorder._record(observation, 'agent', lambda: os._exit(7))
"""


def test_abrupt_exit_preserves_running_stage_without_context_cleanup(tmp_path):
    result = subprocess.run([sys.executable, "-c", SCRIPT, str(tmp_path)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 7, result.stderr
    directory = next((tmp_path / "progress/abrupt_exit").iterdir())
    state = json.loads((directory / "last_state.json").read_text())
    assert state["component"] == "agent" and state["status"] == "RUNNING"
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "stage_started"
    assert not any(item["event"] == "scenario_interrupted" for item in events)
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    assert checkpoint["components"]["document"]["status"] == "PASS"
