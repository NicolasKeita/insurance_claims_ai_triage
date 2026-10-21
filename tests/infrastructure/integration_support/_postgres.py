"""Postgres."""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from persistence.database import DatabaseConfig, require_test_database
from tests.infrastructure.integration_support._config import get_test_database_config


@contextmanager
def isolated_postgres_schema(config: DatabaseConfig | None = None) -> Iterator[Engine]:
    """Own exactly one random schema, and clean only that schema on exit.

    A validated URL and the server's current_database() must agree before any
    DDL. Explicit libpq options override inherited PGOPTIONS and omit public
    from the schema search path, including during migration downgrade checks.
    """
    config = get_test_database_config() if config is None else require_test_database(config.url)
    schema = f"persistence_test_{uuid4().hex}"
    admin = create_engine(
        config.url,
        hide_parameters=True,
        connect_args={"connect_timeout": 5, "options": "-csearch_path=pg_catalog"},
    )
    engine = None
    created = False
    try:
        _create_test_schema(admin, config, schema)
        created = True
        engine = create_engine(
            config.url,
            hide_parameters=True,
            connect_args={"connect_timeout": 5, "options": f"-csearch_path={schema}"},
        )
        yield engine
    finally:
        try:
            if engine is not None:
                engine.dispose()
        finally:
            try:
                if created:
                    with admin.begin() as connection:
                        connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            finally:
                admin.dispose()


def _create_test_schema(admin, config, schema):
    try:
        with admin.begin() as connection:
            actual_database = connection.scalar(text("SELECT current_database()"))
            if actual_database != config.url.database:
                raise RuntimeError(
                    "Refusing PostgreSQL test setup: server database does not match the validated TEST_DATABASE_URL"
                )
            connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    except SQLAlchemyError as error:
        endpoint = config.url.render_as_string(hide_password=True)
        raise RuntimeError(
            f"PostgreSQL test setup failed at {endpoint} ({type(error).__name__}). "
            "Start the dedicated service with docker compose --profile test up -d "
            "--wait postgres-test, or set a safe TEST_DATABASE_URL in CI. "
            "The selected PostgreSQL suite requires a reachable, writable test database."
        ) from error
