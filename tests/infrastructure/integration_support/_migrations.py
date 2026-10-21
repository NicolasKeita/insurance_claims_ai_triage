"""Migrations."""

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect

from persistence.models import Base
from tests.infrastructure.integration_support._constants import PROJECT_ROOT


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
