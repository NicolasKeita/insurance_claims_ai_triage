"""Exercise suite orchestration with fake child reports, never real E2E runs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.scenarios import load_scenarios
from scripts import evaluate_end_to_end_suite as suite

SCENARIOS = load_scenarios()
IDS = [scenario.scenario_id for scenario in SCENARIOS]


def option(command, name):
    return command[command.index(name) + 1]


@pytest.fixture
def children(monkeypatch):
    """Replace every child launch and cooldown before invoking the suite CLI."""
    harness = SimpleNamespace(commands=[], events=[], outcomes={}, before_run=None)

    def run(command):
        scenario_id = option(command, "--scenario")
        output_dir = Path(option(command, "--output-dir"))
        if harness.before_run is not None:
            harness.before_run(scenario_id, output_dir.parent)
        harness.commands.append(command)
        harness.events.append(("run", scenario_id))
        outcome = harness.outcomes.get(scenario_id, {})
        if "error" in outcome:
            raise outcome["error"]
        output_dir.mkdir(parents=True, exist_ok=True)
        if outcome.get("report", True):
            summary = {
                "benchmark_version": "e2e_claims_v2",
                "mode": "real",
                "full_stack_status": outcome.get("full_stack_status", "COMPLETE"),
                "scenarios": [
                    {
                        "scenario_id": outcome.get("reported_id", scenario_id),
                        "status": outcome.get("status", "PASS"),
                    }
                ],
            }
            (output_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        if outcome.get("safety_stop"):
            (output_dir / "safety_stop.json").write_text(
                json.dumps({"status": "STOPPED", "reason": "GPU temperature reached cutoff"}),
                encoding="utf-8",
            )
        harness.events.append(("exit", scenario_id))
        return outcome.get("returncode", 0)

    monkeypatch.setattr(suite, "_run_child", run)
    monkeypatch.setattr(suite.time, "sleep", lambda seconds: harness.events.append(("sleep", seconds)))
    return harness


def read_summary(output_dir):
    summaries = list(output_dir.glob("run_*/suite_summary.json"))
    assert len(summaries) == 1
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def test_all_ten_run_serially_with_cleanup_pause_and_vision_only_when_needed(tmp_path, children):
    assert suite.main(["--output-dir", str(tmp_path)]) == 0
    assert [option(command, "--scenario") for command in children.commands] == IDS
    assert len(IDS) == 10
    expected_events = []
    for index, scenario in enumerate(SCENARIOS):
        if index:
            expected_events.append(("sleep", 30))
        expected_events.extend([("run", scenario.scenario_id), ("exit", scenario.scenario_id)])
        command = children.commands[index]
        assert command[:2] == [suite.sys.executable, "-u"]
        assert Path(command[2]).name == "evaluate_end_to_end.py"
        assert command.count("--scenario") == 1
        assert "--no-include-diagnostics" in command
        expected_vision = "--include-vision" if scenario.vision_required else "--no-include-vision"
        assert expected_vision in command
    assert children.events == expected_events
    assert [scenario.scenario_id for scenario in SCENARIOS if scenario.vision_required] == IDS[-2:]
    summary = read_summary(tmp_path)
    assert summary["full_stack_status"] == "COMPLETE"
    assert summary["counts"]["PASS"] == 10
    assert [result["scenario_id"] for result in summary["results"]] == IDS


def test_effective_options_are_forwarded_without_changing_thresholds(tmp_path, children):
    values = {
        "--config": str(suite.ROOT / "configs/evaluation/e2e_v2_damage960_val640.toml"),
        "--llm-model": "claims-e2e-partiel",
        "--vision-device": "0",
        "--vision-image-size": "640",
        "--vision-max-temperature-c": "77.0",
        "--damage-confidence-threshold": "0.35",
        "--part-confidence-threshold": "0.25",
        "--database-url": "postgresql+psycopg://claims_test:claims_test@127.0.0.1:5434/insurance_claims_test",
        "--qdrant-url": "http://127.0.0.1:6335",
    }
    args = [
        "--output-dir",
        str(tmp_path),
        "--scenario",
        IDS[-1],
        "--isolated-demo-stack",
        "--include-vision",
        "--no-include-diagnostics",
        "--cooldown-seconds",
        "45",
    ]
    for name, value in values.items():
        args.extend([name, value])
    assert suite.main(args) == 0
    assert len(children.commands) == 1
    command = children.commands[0]
    for name, value in values.items():
        assert option(command, name) == value
    assert "--isolated-demo-stack" in command
    assert "--include-vision" in command
    assert not any(event[0] == "sleep" for event in children.events)


def test_selected_scenarios_keep_manifest_order_and_use_custom_cooldown(tmp_path, children):
    assert (
        suite.main(
            [
                "--output-dir",
                str(tmp_path),
                "--scenario",
                IDS[-1],
                "--scenario",
                IDS[0],
                "--cooldown-seconds",
                "2.5",
            ]
        )
        == 0
    )
    assert [option(command, "--scenario") for command in children.commands] == [IDS[0], IDS[-1]]
    assert [event for event in children.events if event[0] == "sleep"] == [("sleep", 2.5)]


def test_default_fail_fast_stops_before_cooldown_or_next_process(tmp_path, children):
    children.outcomes[IDS[0]] = {"status": "FAIL", "returncode": 1}
    assert suite.main(["--output-dir", str(tmp_path)]) != 0
    assert len(children.commands) == 1
    assert not any(event[0] == "sleep" for event in children.events)
    summary = read_summary(tmp_path)
    assert summary["counts"]["FAIL"] == 1
    assert summary["counts"]["NOT_RUN"] == 9
    assert summary["full_stack_status"] == "PARTIAL"


def test_continue_after_ordinary_failure_still_returns_unsuccessful_suite(tmp_path, children):
    children.outcomes[IDS[0]] = {"status": "FAIL", "returncode": 1}
    assert suite.main(["--output-dir", str(tmp_path), "--no-fail-fast"]) != 0
    assert len(children.commands) == 10
    summary = read_summary(tmp_path)
    assert summary["counts"]["FAIL"] == 1
    assert summary["counts"]["PASS"] == 9
    assert summary["full_stack_status"] == "PARTIAL"


@pytest.mark.parametrize(
    "outcome",
    [
        {"status": "SKIP", "full_stack_status": "PARTIAL"},
        {"report": False},
        {"reported_id": "another_scenario"},
        {"returncode": 2},
        {"full_stack_status": "PARTIAL"},
    ],
)
def test_exit_zero_or_partial_report_cannot_disguise_missing_success(tmp_path, children, outcome):
    children.outcomes[IDS[0]] = outcome
    assert suite.main(["--output-dir", str(tmp_path)]) != 0
    assert len(children.commands) == 1
    summary = read_summary(tmp_path)
    assert summary["full_stack_status"] == "PARTIAL"
    if outcome.get("status") == "SKIP":
        assert summary["counts"]["SKIP"] == 1


def test_safety_stop_aborts_even_when_continuation_was_requested(tmp_path, children):
    children.outcomes[IDS[0]] = {"safety_stop": True, "returncode": 1}
    assert suite.main(["--output-dir", str(tmp_path), "--no-fail-fast"]) != 0
    assert len(children.commands) == 1
    summary = read_summary(tmp_path)
    assert summary["counts"]["STOPPED"] == 1
    assert summary["counts"]["NOT_RUN"] == 9
    assert "temperature" in summary["results"][0]["reason"].lower()
    assert not any(event[0] == "sleep" for event in children.events)


def test_summary_is_saved_before_starting_next_child(tmp_path, children):
    def before_run(scenario_id, run_dir):
        if scenario_id != IDS[0]:
            report = json.loads((run_dir / "suite_summary.json").read_text(encoding="utf-8"))
            assert report["results"][0]["scenario_id"] == IDS[0]
            assert report["results"][0]["status"] == "PASS"

    children.before_run = before_run
    assert suite.main(["--output-dir", str(tmp_path), "--scenario", IDS[0], "--scenario", IDS[1]]) == 0


def test_keyboard_interrupt_saves_partial_result_and_launches_no_more_children(tmp_path, children):
    children.outcomes[IDS[1]] = {"error": KeyboardInterrupt()}
    assert suite.main(["--output-dir", str(tmp_path)]) != 0
    assert len(children.commands) == 2
    summary = read_summary(tmp_path)
    assert summary["results"][0]["status"] == "PASS"
    assert summary["full_stack_status"] == "PARTIAL"


def test_interrupt_during_cooldown_leaves_next_scenario_unstarted(tmp_path, children, monkeypatch):
    def interrupted_sleep(seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr(suite.time, "sleep", interrupted_sleep)
    assert suite.main(["--output-dir", str(tmp_path)]) == 130
    assert len(children.commands) == 1
    summary = read_summary(tmp_path)
    assert summary["results"][0]["status"] == "PASS"
    assert summary["counts"]["NOT_RUN"] == 9


def test_repeated_invocation_does_not_reuse_a_previous_success_report(tmp_path, children):
    args = ["--output-dir", str(tmp_path), "--scenario", IDS[0]]
    assert suite.main(args) == 0
    first_dir = Path(option(children.commands[0], "--output-dir"))
    children.outcomes[IDS[0]] = {"report": False}
    assert suite.main(args) != 0
    second_dir = Path(option(children.commands[1], "--output-dir"))
    assert first_dir != second_dir
    assert (first_dir / "summary.json").is_file()
    assert len(list(tmp_path.glob("run_*/suite_summary.json"))) == 2


def test_child_cannot_be_started_is_recorded_as_failure(tmp_path, children):
    children.outcomes[IDS[0]] = {"error": OSError("cannot start child")}
    assert suite.main(["--output-dir", str(tmp_path)]) != 0
    summary = read_summary(tmp_path)
    assert summary["counts"]["FAIL"] == 1
    assert "cannot start child" in summary["results"][0]["reason"]
    assert len(children.commands) == 1


def test_zero_cooldown_has_no_sleep_calls(tmp_path, children):
    assert (
        suite.main(
            [
                "--output-dir",
                str(tmp_path),
                "--scenario",
                IDS[0],
                "--scenario",
                IDS[1],
                "--cooldown-seconds",
                "0",
            ]
        )
        == 0
    )
    assert len(children.commands) == 2
    assert not any(event[0] == "sleep" for event in children.events)


def test_dry_run_has_no_process_cooldown_or_artifact_side_effects(tmp_path, children):
    output_dir = tmp_path / "not-created"
    assert suite.main(["--output-dir", str(output_dir), "--dry-run"]) == 0
    assert children.events == []
    assert not output_dir.exists()


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "-inf", "invalid"])
def test_invalid_cooldown_is_rejected_before_any_child_launch(tmp_path, children, value):
    with pytest.raises(SystemExit) as error:
        suite.main(["--output-dir", str(tmp_path), "--cooldown-seconds", value])
    assert error.value.code != 0
    assert children.events == []
    assert not list(tmp_path.iterdir())


def test_unknown_scenario_fails_without_starting_children(tmp_path, children):
    with pytest.raises((SystemExit, ValueError)):
        suite.main(["--output-dir", str(tmp_path), "--scenario", "misspelled_scenario"])
    assert children.events == []


def test_explicitly_disabled_vision_is_preserved_for_visual_scenarios(tmp_path, children):
    children.outcomes[IDS[-1]] = {"status": "SKIP", "full_stack_status": "PARTIAL"}
    assert (
        suite.main(
            [
                "--output-dir",
                str(tmp_path),
                "--scenario",
                IDS[-1],
                "--no-include-vision",
            ]
        )
        != 0
    )
    assert "--no-include-vision" in children.commands[0]
    assert "--include-vision" not in children.commands[0]


def fake_popen(monkeypatch, replies):
    """Return a fake process with scripted waits; no OS child can be started."""
    replies = iter(replies)
    events = []

    def wait(timeout=None):
        events.append(("wait", timeout))
        result = next(replies)
        if isinstance(result, BaseException):
            raise result
        return result

    process = SimpleNamespace(
        wait=wait,
        terminate=lambda: events.append(("terminate",)),
        kill=lambda: events.append(("kill",)),
    )

    def popen(command, *, cwd):
        assert command == ["fake-python", "fake-evaluator"]
        assert cwd == suite.ROOT
        return process

    monkeypatch.setattr(suite.subprocess, "Popen", popen)
    return events


def test_child_wait_finishes_before_returning_to_orchestrator(monkeypatch):
    events = fake_popen(monkeypatch, [3])
    assert suite._run_child(["fake-python", "fake-evaluator"]) == 3
    assert events == [("wait", None)]


@pytest.mark.parametrize("cleanup", ["graceful", "terminate", "kill"])
def test_interrupted_child_gets_cleanup_time_before_forced_exit(monkeypatch, cleanup):
    timeout = suite.subprocess.TimeoutExpired("fake-evaluator", 30)
    replies = [KeyboardInterrupt()]
    expected = [("wait", None), ("wait", 30)]
    if cleanup in {"terminate", "kill"}:
        replies.append(timeout)
        expected.extend([("terminate",), ("wait", 10)])
    if cleanup == "kill":
        replies.append(timeout)
        expected.extend([("kill",), ("wait", None)])
    replies.append(130)
    events = fake_popen(monkeypatch, replies)
    with pytest.raises(KeyboardInterrupt):
        suite._run_child(["fake-python", "fake-evaluator"])
    assert events == expected
