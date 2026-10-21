"""Test selection."""

from pathlib import Path

import pytest


def test_unit_and_vision_selections_do_not_probe_docker_or_services(monkeypatch):
    import tests.integration_prerequisites._selection as support

    def forbidden(*args, **kwargs):
        pytest.fail("This selection must not inspect Docker or external services")

    monkeypatch.setattr(support, "require_compose_services", forbidden)
    monkeypatch.setattr(support, "require_postgres_connection", forbidden)
    monkeypatch.setattr(support, "require_qdrant_connection", forbidden)
    for markers in ({"unit"}, {"vision"}, set()):
        support.check_selected_prerequisites(markers, Path.cwd())


@pytest.mark.parametrize(
    "markers,services,probes",
    [
        ({"postgres"}, ["postgres-test"], ["postgres"]),
        ({"qdrant"}, ["qdrant-test"], ["qdrant"]),
        ({"rag"}, ["qdrant-test"], ["qdrant", "cache"]),
        ({"postgres", "qdrant"}, ["postgres-test", "qdrant-test"], ["postgres", "qdrant"]),
    ],
)
def test_only_selected_marker_dependencies_are_probed(monkeypatch, markers, services, probes):
    import tests.integration_prerequisites._selection as support

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
    import tests.integration_prerequisites._selection as support

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://ci:secret@ci-db:5432/pipeline_test")
    monkeypatch.setenv("TEST_QDRANT_URL", "http://ci-qdrant:6333")
    seen = []
    monkeypatch.setattr(support, "require_compose_services", lambda names, root: seen.extend(names))
    monkeypatch.setattr(support, "require_postgres_connection", lambda config: None)
    monkeypatch.setattr(support, "require_qdrant_connection", lambda url: None)
    support.check_selected_prerequisites({"postgres", "qdrant"}, Path.cwd())
    assert seen == []
