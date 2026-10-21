"""Support."""

from contextlib import contextmanager

import pytest


@pytest.fixture(autouse=True)
def isolated_configuration(monkeypatch):
    for variable in ("DATABASE_URL", "TEST_DATABASE_URL", "QDRANT_URL", "TEST_QDRANT_URL"):
        monkeypatch.delenv(variable, raising=False)


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
