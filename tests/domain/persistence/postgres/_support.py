"""Support."""

import pytest
from sqlalchemy.orm import sessionmaker

from tests.infrastructure.integration_support import isolated_postgres_schema, migrate_test_schema


@pytest.fixture(scope="module")
def postgres_engine():
    with isolated_postgres_schema() as engine:
        migrate_test_schema(engine, verify_downgrade=True)
        yield engine


@pytest.fixture
def factory(postgres_engine):
    """
    Application commits become savepoints; an outer rollback isolates tests.
    """
    with postgres_engine.connect() as connection:
        outer = connection.begin()
        factory = sessionmaker(connection, expire_on_commit=False, join_transaction_mode="create_savepoint")
        try:
            yield factory
        finally:
            outer.rollback()
