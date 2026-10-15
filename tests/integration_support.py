"""Shared, test-only defaults and isolated PostgreSQL setup (no import-time I/O)."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from persistence.database import DatabaseConfig, require_test_database
from persistence.models import Base

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DATABASE_URL = (
    "postgresql+psycopg://claims_test:claims_test@127.0.0.1:5434/insurance_claims_test"
)
DEFAULT_TEST_QDRANT_URL = "http://127.0.0.1:6335"


def get_test_database_config() -> DatabaseConfig:
    """Use Compose's isolated test database unless CI explicitly overrides it."""
    return require_test_database(os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL))


def get_test_qdrant_url() -> str:
    """Never inherit the application's QDRANT_URL for integration tests."""
    value = os.environ.get("TEST_QDRANT_URL", DEFAULT_TEST_QDRANT_URL)
    if not value:
        raise ValueError("TEST_QDRANT_URL must not be empty")
    return value


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
        config.url, hide_parameters=True,
        connect_args={"connect_timeout": 5, "options": "-csearch_path=pg_catalog"},
    )
    engine = None
    created = False
    try:
        try:
            with admin.begin() as connection:
                actual_database = connection.scalar(text("SELECT current_database()"))
                if actual_database != config.url.database:
                    raise RuntimeError(
                        "Refusing PostgreSQL test setup: server database does not match "
                        "the validated TEST_DATABASE_URL"
                    )
                connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            created = True
        except SQLAlchemyError as error:
            endpoint = config.url.render_as_string(hide_password=True)
            raise RuntimeError(
                f"PostgreSQL test setup failed at {endpoint} ({type(error).__name__}). "
                "Start the dedicated service with docker compose --profile test up -d "
                "--wait postgres-test, or set a safe TEST_DATABASE_URL in CI. "
                "The selected PostgreSQL suite requires a reachable, writable test database."
            ) from error
        engine = create_engine(
            config.url, hide_parameters=True,
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


def migrate_test_schema(engine: Engine, *, verify_downgrade: bool = False) -> None:
    """Apply the existing Alembic history to this run's fresh isolated schema."""
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    with engine.begin() as connection:
        if inspect(connection).get_table_names():
            raise RuntimeError("PostgreSQL test migrations require a fresh isolated schema")
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")
        if verify_downgrade:
            command.downgrade(cfg, "base")
            if inspect(connection).get_table_names() != ["alembic_version"]:
                raise RuntimeError("Alembic downgrade left unexpected tables in the test schema")
            command.upgrade(cfg, "head")
        tables = set(inspect(connection).get_table_names())
        if not set(Base.metadata.tables) <= tables:
            raise RuntimeError("Alembic did not create all expected PostgreSQL test tables")
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
        if differences:
            raise RuntimeError(f"PostgreSQL test migrations differ from ORM metadata: {differences}")
