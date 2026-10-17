"""Explicit database configuration and short-lived units of work."""

import os
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
            raise DatabaseConfigurationError("A postgresql+psycopg URL with an explicit database is required")
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
