"""Explicit configuration and short-lived units of work. No import-time I/O."""

import os
import re
from ipaddress import ip_address
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import URL, make_url
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class DatabaseConfig:
    url: URL

    @classmethod
    def from_env(cls, variable: str = "DATABASE_URL") -> "DatabaseConfig":
        value = os.environ.get(variable)
        if not value:
            raise DatabaseConfigurationError(f"{variable} is required")
        return cls.from_url(value)

    @classmethod
    def from_url(cls, value: str | URL) -> "DatabaseConfig":
        try:
            url = make_url(value)
        except Exception as error:
            raise DatabaseConfigurationError("Invalid database URL") from error
        if url.drivername != "postgresql+psycopg" or not url.database:
            raise DatabaseConfigurationError(
                "A postgresql+psycopg URL with an explicit database is required"
            )
        return cls(url)


def create_database_engine(config: DatabaseConfig) -> Engine:
    return create_engine(config.url, pool_pre_ping=True, hide_parameters=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def transaction(factory: sessionmaker[Session]) -> Iterator[Session]:
    """The application owns this atomic commit/rollback and session lifetime."""
    with factory.begin() as session:
        yield session


def require_test_database(value: str | URL) -> DatabaseConfig:
    """Guard destructive setup using the actual connection target, not an env name.

    Libpq connection overrides can redirect an otherwise safe-looking URL to a
    different database or schema. Test URLs therefore allow only ``sslmode`` as
    a query parameter; search paths are set by the isolated-schema fixtures.
    """
    config = DatabaseConfig.from_url(value)
    target = config.url
    if (
        target.database in {"insurance_claims", "postgres", "template0", "template1"}
        or not re.fullmatch(r"[a-z][a-z0-9_]*_test", target.database)
        or len(target.database) > 63
    ):
        raise DatabaseConfigurationError(
            "TEST_DATABASE_URL must name an explicit non-system database ending in _test; "
            "the application database insurance_claims is forbidden"
        )
    host = target.host
    if not host:
        raise DatabaseConfigurationError("TEST_DATABASE_URL requires an explicit host")
    try:
        ip_address(host)
    except ValueError:
        # Reject sockets, multiple hosts, percent escapes and delimiter tricks.
        if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?", host):
            raise DatabaseConfigurationError(
                "TEST_DATABASE_URL requires one unambiguous hostname or IP address"
            ) from None
    if target.port is not None and not 1 <= target.port <= 65535:
        raise DatabaseConfigurationError("TEST_DATABASE_URL port must be between 1 and 65535")
    for key, entry in target.query.items():
        if key != "sslmode" or not isinstance(entry, str) or entry not in {
            "disable", "allow", "prefer", "require", "verify-ca", "verify-full",
        }:
            raise DatabaseConfigurationError(
                "TEST_DATABASE_URL query only permits one sslmode value; "
                "connection, driver, database and search_path overrides are forbidden"
            )
    development = os.environ.get("DATABASE_URL")
    if development:
        dev = DatabaseConfig.from_url(development).url
        if dev.database == target.database:
            raise DatabaseConfigurationError("Test and development databases must differ")
    # An explicit port also prevents a PGPORT environment value from changing it.
    return DatabaseConfig(target.set(port=target.port or 5432))
