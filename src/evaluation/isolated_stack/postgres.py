"""Explicit opt-in real PostgreSQL demo isolation; never targets application data.

The caller supplies the dedicated test URL. Each context owns one random schema,
applies the application's migrations, and drops only that schema at completion.
No fake model or test module participates in this real-stack setup.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from persistence.database import create_session_factory, require_test_database
from persistence.models import Base

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class IsolatedPostgresStack:
    engine: Engine
    factory: sessionmaker
    schema: str


def _create_schema(admin, config, schema):
    with admin.begin() as connection:
        actual = connection.scalar(text("SELECT current_database()"))
        if actual != config.url.database:
            raise RuntimeError("Evaluation setup refused: PostgreSQL database differs from the validated test URL")
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))


def _upgrade_schema(engine):
    """Apply the application's migrations to the empty isolated schema."""
    migration = Config(str(ROOT / "alembic.ini"))
    migration.set_main_option("script_location", str(ROOT / "migrations"))
    with engine.begin() as connection:
        if inspect(connection).get_table_names():
            raise RuntimeError("Evaluation migrations require an empty isolated schema")
        migration.attributes["connection"] = connection
        command.upgrade(migration, "head")
        if not set(Base.metadata.tables).issubset(inspect(connection).get_table_names()):
            raise RuntimeError("Evaluation migrations did not create all application tables")


@contextmanager
def isolated_postgres(database_url) -> Iterator[IsolatedPostgresStack]:
    """Migrate a fresh schema in an explicitly validated ``*_test`` database."""
    config = require_test_database(database_url)
    schema = f"evaluation_e2e_{uuid4().hex}"
    admin = create_engine(
        config.url, hide_parameters=True, connect_args={"connect_timeout": 5, "options": "-csearch_path=pg_catalog"}
    )
    engine, created = None, False
    try:
        _create_schema(admin, config, schema)
        created = True
        engine = create_engine(
            config.url, hide_parameters=True, connect_args={"connect_timeout": 5, "options": f"-csearch_path={schema}"}
        )
        _upgrade_schema(engine)
        yield IsolatedPostgresStack(engine, create_session_factory(engine), schema)
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