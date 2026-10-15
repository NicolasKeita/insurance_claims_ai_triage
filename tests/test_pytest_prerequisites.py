"""Prerequisite failures are explicit and diagnostic probes never start services."""

from pathlib import Path
from types import SimpleNamespace
import subprocess
from urllib.error import HTTPError, URLError

import pytest

import integration_prerequisites as support


def test_unit_and_vision_selections_do_not_probe_docker_or_services(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("This selection must not inspect Docker or external services")
    monkeypatch.setattr(support, "require_compose_services", forbidden)
    monkeypatch.setattr(support, "require_postgres_connection", forbidden)
    monkeypatch.setattr(support, "require_qdrant_connection", forbidden)
    for markers in ({"unit"}, {"vision"}, set()):
        support.check_selected_prerequisites(markers, Path.cwd())


@pytest.mark.parametrize("markers,services,probes", [
    ({"postgres"}, ["postgres-test"], ["postgres"]),
    ({"qdrant"}, ["qdrant-test"], ["qdrant"]),
    ({"rag"}, ["qdrant-test"], ["qdrant", "cache"]),
    ({"postgres", "qdrant"}, ["postgres-test", "qdrant-test"], ["postgres", "qdrant"]),
])
def test_only_selected_marker_dependencies_are_probed(monkeypatch, markers, services, probes):
    for name in ("TEST_DATABASE_URL", "TEST_QDRANT_URL", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    seen = []
    monkeypatch.setattr(support, "require_compose_services", lambda names, root: seen.append(list(names)))
    monkeypatch.setattr(support, "require_postgres_connection", lambda config: seen.append("postgres"))
    monkeypatch.setattr(support, "require_qdrant_connection", lambda url: seen.append("qdrant"))
    monkeypatch.setattr(support, "require_cached_rag_model", lambda: seen.append("cache"))
    support.check_selected_prerequisites(markers, Path.cwd())
    assert seen == [services, *probes]


def test_external_ci_services_do_not_require_docker(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://ci:secret@ci-db:5432/pipeline_test")
    monkeypatch.setenv("TEST_QDRANT_URL", "http://ci-qdrant:6333")
    seen = []
    monkeypatch.setattr(support, "require_compose_services", lambda names, root: seen.extend(names))
    monkeypatch.setattr(support, "require_postgres_connection", lambda config: None)
    monkeypatch.setattr(support, "require_qdrant_connection", lambda url: None)
    support.check_selected_prerequisites({"postgres", "qdrant"}, Path.cwd())
    assert seen == []


def test_missing_docker_has_a_different_diagnostic_from_a_stopped_engine(monkeypatch):
    monkeypatch.setattr(support.shutil, "which", lambda command: None)
    with pytest.raises(support.PrerequisiteError, match="Docker CLI is not installed"):
        support.require_compose_services(["postgres-test"], Path.cwd())
    monkeypatch.setattr(support.shutil, "which", lambda command: "docker")
    monkeypatch.setattr(support.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr="cannot connect to engine"))
    with pytest.raises(support.PrerequisiteError, match="Docker Engine is not available.*") as error:
        support.require_compose_services(["postgres-test"], Path.cwd())
    assert "--wait postgres-test" in str(error.value)
    assert "Start Docker Desktop" in str(error.value)


@pytest.mark.parametrize("state,message", [
    ({"Running": False, "Status": "exited"}, "is stopped"),
    ({"Running": True, "Health": {"Status": "unhealthy"}}, "not healthy.*unhealthy"),
    ({"Running": True, "Health": {"Status": "starting"}}, "not healthy.*starting"),
])
def test_container_state_diagnostics_are_specific_and_commands_are_read_only(monkeypatch, state, message):
    import json
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
    monkeypatch.setattr(support.shutil, "which", lambda command: "docker")
    monkeypatch.setattr(support.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="", stderr=""))
    with pytest.raises(support.PrerequisiteError, match="has not been created"):
        support.require_compose_services(["postgres-test"], Path.cwd())
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker", 10)
    monkeypatch.setattr(support.subprocess, "run", timeout)
    with pytest.raises(support.PrerequisiteError, match="Docker inspection failed"):
        support.require_compose_services(["postgres-test"], Path.cwd())


def test_postgres_refused_connection_is_not_a_skip(monkeypatch):
    from integration_support import get_test_database_config
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    def refused(*args, **kwargs):
        raise ConnectionRefusedError("refused")
    monkeypatch.setattr(support.socket, "create_connection", refused)
    with pytest.raises(support.PrerequisiteError, match="PostgreSQL connection refused") as error:
        support.require_postgres_connection(get_test_database_config())
    assert "--wait postgres-test" in str(error.value)
    assert ":claims_test@" not in str(error.value)


@pytest.mark.parametrize("failure,message", [
    (URLError(ConnectionRefusedError("refused")), "Qdrant connection refused"),
    (HTTPError("http://test", 503, "not ready", {}, None), "HTTP 503"),
    (HTTPError("http://test", 401, "unauthorized", {}, None), "HTTP 401"),
])
def test_qdrant_network_and_readiness_diagnostics(monkeypatch, failure, message):
    def unavailable(*args, **kwargs):
        raise failure
    monkeypatch.setattr(support, "urlopen", unavailable)
    with pytest.raises(support.PrerequisiteError, match=message):
        support.require_qdrant_connection("http://127.0.0.1:6335")


def test_missing_rag_cache_is_checked_without_loading_or_downloading_a_model(monkeypatch):
    import huggingface_hub
    calls = []
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda model, name: calls.append(name))
    with pytest.raises(support.PrerequisiteError, match="not present in the local cache"):
        support.require_cached_rag_model()
    assert calls == ["modules.json"]
