"""Test schema."""

import re
from contextlib import contextmanager

import pytest
from sqlalchemy.exc import OperationalError

import tests.infrastructure.integration_support._postgres as integration_support
from tests.infrastructure.integration_support import (
    isolated_postgres_schema,
)
from tests.infrastructure.isolation._constants import pytestmark as pytestmark
from tests.infrastructure.isolation._support import RecordingConnection, RecordingEngine
from tests.infrastructure.isolation._support import (
    isolated_configuration as isolated_configuration,
)


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
