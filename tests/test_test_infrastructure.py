"""Data-safety regressions for test setup; no external services are contacted."""

from contextlib import contextmanager
import re

import pytest
from sqlalchemy.exc import OperationalError

import integration_support
from integration_support import (
    DEFAULT_TEST_DATABASE_URL, DEFAULT_TEST_QDRANT_URL,
    get_test_database_config, get_test_qdrant_url, isolated_postgres_schema,
)
from persistence.database import DatabaseConfigurationError, require_test_database

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    for variable in ("DATABASE_URL", "TEST_DATABASE_URL", "QDRANT_URL", "TEST_QDRANT_URL"):
        monkeypatch.delenv(variable, raising=False)


def test_integration_defaults_use_dedicated_services(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://claims:claims@localhost:5433/insurance_claims")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    config = get_test_database_config()
    assert config.url.render_as_string(hide_password=False) == DEFAULT_TEST_DATABASE_URL
    assert config.url.database == "insurance_claims_test"
    assert config.url.host == "127.0.0.1" and config.url.port == 5434
    assert get_test_qdrant_url() == DEFAULT_TEST_QDRANT_URL


def test_ci_can_override_dedicated_test_services(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://ci:secret@ci-db:5544/pipeline_test?sslmode=require")
    monkeypatch.setenv("TEST_QDRANT_URL", "http://ci-qdrant:6333")
    config = get_test_database_config()
    assert config.url.database == "pipeline_test" and config.url.port == 5544
    assert get_test_qdrant_url() == "http://ci-qdrant:6333"


@pytest.mark.parametrize("database", [
    "insurance_claims", "postgres", "template0", "template1", "production",
    "test", "insurance_claims_test_backup", "public/claims_test", "claims%255Ftest",
    "claims_test;dbname=insurance_claims", "claims_test#fragment", "Claims_test",
    "x" * 60 + "_test",
])
def test_database_guard_rejects_unsafe_or_ambiguous_names(database):
    with pytest.raises(DatabaseConfigurationError):
        require_test_database(f"postgresql+psycopg://claims:secret@localhost:5434/{database}")


@pytest.mark.parametrize("query", [
    "dbname=insurance_claims", "database=insurance_claims", "host=production",
    "hostaddr=10.0.0.1", "port=5432", "user=postgres", "service=production",
    "options=-csearch_path%3Dpublic", "options=-crole%3Dpostgres", "search_path=public",
    "connect_timeout=600", "driver=psycopg2", "schema=public",
    "sslmode=require&sslmode=disable", "sslmode=anything", "SSLMODE=require",
])
def test_database_guard_blocks_libpq_connection_and_schema_overrides(query):
    with pytest.raises(DatabaseConfigurationError):
        require_test_database(DEFAULT_TEST_DATABASE_URL + "?" + query)


@pytest.mark.parametrize("url", [
    "postgresql://localhost/claims_test", "postgresql+psycopg2://localhost/claims_test",
    "sqlite:///claims_test", "postgresql+psycopg:///claims_test",
    "postgresql+psycopg://host1,host2/claims_test",
    "postgresql+psycopg://local%68ost/claims_test",
    "postgresql+psycopg://localhost:0/claims_test",
    "postgresql+psycopg://localhost:65536/claims_test",
    "postgresql+psycopg://localhost:invalid/claims_test",
])
def test_database_guard_rejects_implicit_or_ambiguous_connection_targets(url):
    with pytest.raises(DatabaseConfigurationError):
        require_test_database(url)


def test_database_guard_rejects_development_name_even_through_another_hostname(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost:5432/shared_test")
    with pytest.raises(DatabaseConfigurationError, match="differ"):
        require_test_database("postgresql+psycopg://127.0.0.1:5434/shared_test")


def test_explicit_default_port_cannot_inherit_pgport(monkeypatch):
    monkeypatch.setenv("PGPORT", "5433")
    assert require_test_database("postgresql+psycopg://localhost/claims_test").url.port == 5432


@pytest.mark.parametrize("variable", ["TEST_DATABASE_URL", "TEST_QDRANT_URL"])
def test_empty_explicit_override_is_an_error(variable, monkeypatch):
    monkeypatch.setenv(variable, "")
    with pytest.raises(ValueError):
        (get_test_database_config if variable == "TEST_DATABASE_URL" else get_test_qdrant_url)()


class RecordingConnection:
    def __init__(self, database):
        self.database = database
        self.statements = []

    def scalar(self, statement):
        self.statements.append(str(statement))
        return self.database

    def execute(self, statement):
        self.statements.append(str(statement))


class RecordingEngine:
    def __init__(self, connection):
        self.connection = connection
        self.disposed = False

    @contextmanager
    def begin(self):
        yield self.connection

    def dispose(self):
        self.disposed = True


def test_server_database_is_verified_before_any_schema_write(monkeypatch):
    connection = RecordingConnection("insurance_claims")
    admin = RecordingEngine(connection)
    monkeypatch.setattr(integration_support, "create_engine", lambda *args, **kwargs: admin)
    with pytest.raises(RuntimeError, match="server database does not match"):
        with isolated_postgres_schema():
            pytest.fail("Unsafe server must never be yielded to a test")
    assert connection.statements == ["SELECT current_database()"]
    assert admin.disposed


def test_schema_cleanup_targets_only_owned_schema_even_after_test_failure(monkeypatch):
    connection = RecordingConnection("insurance_claims_test")
    admin, scoped = RecordingEngine(connection), RecordingEngine(connection)
    calls = []

    def create_recording_engine(*args, **kwargs):
        calls.append(kwargs["connect_args"])
        return admin if len(calls) == 1 else scoped

    monkeypatch.setattr(integration_support, "create_engine", create_recording_engine)
    with pytest.raises(RuntimeError, match="test body failed"):
        with isolated_postgres_schema() as engine:
            assert engine is scoped
            raise RuntimeError("test body failed")
    _, create, drop = connection.statements
    match = re.fullmatch(r'CREATE SCHEMA "(persistence_test_[a-f0-9]{32})"', create)
    assert match is not None
    schema = match.group(1)
    assert drop == f'DROP SCHEMA "{schema}" CASCADE'
    assert calls == [
        {"connect_timeout": 5, "options": "-csearch_path=pg_catalog"},
        {"connect_timeout": 5, "options": f"-csearch_path={schema}"},
    ]
    assert admin.disposed and scoped.disposed


def test_unavailable_postgres_has_actionable_failure_and_redacted_password(monkeypatch):
    @contextmanager
    def unavailable():
        raise OperationalError("connect", {}, RuntimeError("connection refused"))
        yield  # pragma: no cover

    admin = RecordingEngine(RecordingConnection("insurance_claims_test"))
    admin.begin = unavailable
    monkeypatch.setattr(integration_support, "create_engine", lambda *args, **kwargs: admin)
    with pytest.raises(RuntimeError, match="docker compose --profile test") as caught:
        with isolated_postgres_schema():
            pytest.fail("Unavailable service must fail setup")
    assert "claims_test:claims_test" not in str(caught.value)
    assert "5434/insurance_claims_test" in str(caught.value)
    assert admin.disposed


def test_owned_schema_and_admin_are_cleaned_even_when_scoped_engine_disposal_fails(monkeypatch):
    connection = RecordingConnection("insurance_claims_test")
    admin = RecordingEngine(connection)

    class FailingDisposeEngine(RecordingEngine):
        def dispose(self):
            self.disposed = True
            raise RuntimeError("scoped engine disposal failed")

    scoped = FailingDisposeEngine(connection)
    engines = iter((admin, scoped))
    monkeypatch.setattr(integration_support, "create_engine", lambda *args, **kwargs: next(engines))
    with pytest.raises(RuntimeError, match="scoped engine disposal failed"):
        with isolated_postgres_schema():
            pass
    _, create, drop = connection.statements
    schema = re.fullmatch(r'CREATE SCHEMA "(persistence_test_[a-f0-9]{32})"', create).group(1)
    assert drop == f'DROP SCHEMA "{schema}" CASCADE'
    assert scoped.disposed and admin.disposed
