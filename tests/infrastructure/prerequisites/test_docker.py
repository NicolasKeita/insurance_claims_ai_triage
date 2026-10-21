"""Test docker."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_missing_docker_has_a_different_diagnostic_from_a_stopped_engine(monkeypatch):
    import tests.infrastructure.integration_prerequisites._compose as support

    monkeypatch.setattr(support.shutil, "which", lambda command: None)
    with pytest.raises(support.PrerequisiteError, match="Docker CLI is not installed"):
        support.require_compose_services(["postgres-test"], Path.cwd())
    monkeypatch.setattr(support.shutil, "which", lambda command: "docker")
    monkeypatch.setattr(
        support.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="cannot connect to engine"),
    )
    with pytest.raises(support.PrerequisiteError, match="Docker Engine is not available.*") as error:
        support.require_compose_services(["postgres-test"], Path.cwd())
    assert "--wait postgres-test" in str(error.value)
    assert "Start Docker Desktop" in str(error.value)


@pytest.mark.parametrize(
    "state,message",
    [
        ({"Running": False, "Status": "exited"}, "is stopped"),
        ({"Running": True, "Health": {"Status": "unhealthy"}}, "not healthy.*unhealthy"),
        ({"Running": True, "Health": {"Status": "starting"}}, "not healthy.*starting"),
    ],
)
def test_container_state_diagnostics_are_specific_and_commands_are_read_only(monkeypatch, state, message):
    import json

    import tests.infrastructure.integration_prerequisites._compose as support

    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "info" in command:
            output = "29.8.2"
        elif "ps" in command:
            output = "test-container-id"
        else:
            assert command[1] == "inspect"
            output = json.dumps(state)
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(support.shutil, "which", lambda command: "docker")
    monkeypatch.setattr(support.subprocess, "run", run)
    with pytest.raises(support.PrerequisiteError, match=message):
        support.require_compose_services(["qdrant-test"], Path.cwd())
    assert len(calls) == 3
    assert all(not {"start", "up", "restart", "install"}.intersection(command) for command in calls)


def test_container_not_created_and_inspection_timeout_are_explicit(monkeypatch):
    import tests.infrastructure.integration_prerequisites._compose as support

    monkeypatch.setattr(support.shutil, "which", lambda command: "docker")
    monkeypatch.setattr(
        support.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr="")
    )
    with pytest.raises(support.PrerequisiteError, match="has not been created"):
        support.require_compose_services(["postgres-test"], Path.cwd())

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker", 10)

    monkeypatch.setattr(support.subprocess, "run", timeout)
    with pytest.raises(support.PrerequisiteError, match="Docker inspection failed"):
        support.require_compose_services(["postgres-test"], Path.cwd())
