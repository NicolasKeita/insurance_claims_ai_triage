"""Explicit configuration and short-lived units of work. No import-time I/O."""

import os
import re
from ipaddress import ip_address

from sqlalchemy.engine import URL

from persistence.database_config import (
    DatabaseConfig,
    DatabaseConfigurationError,
    create_database_engine,
    create_session_factory,
    transaction,
)

__all__ = [
    "DatabaseConfig",
    "DatabaseConfigurationError",
    "create_database_engine",
    "create_session_factory",
    "require_test_database",
    "transaction",
]


def require_test_database(value: str | URL) -> DatabaseConfig:
    """Guard destructive setup using the actual connection target, not an env name.

    Libpq connection overrides can redirect an otherwise safe-looking URL to a
    different database or schema. Test URLs therefore allow only ``sslmode`` as
    a query parameter; search paths are set by the isolated-schema fixtures.

    Reject sockets, multiple hosts, percent escapes and delimiter tricks.

    An explicit port also prevents a PGPORT environment value from changing it.
    """
    config = DatabaseConfig.from_url(value)
    target = config.url
    _validate_test_database_name(target)
    _validate_test_host(target)
    _validate_test_query(target)
    _validate_distinct_from_development(target)
    return DatabaseConfig(target.set(port=target.port or 5432))


def _validate_test_database_name(target):
    if (
        target.database in {"insurance_claims", "postgres", "template0", "template1"}
        or not re.fullmatch(r"[a-z][a-z0-9_]*_test", target.database)
        or len(target.database) > 63
    ):
        raise DatabaseConfigurationError(
            "TEST_DATABASE_URL must name an explicit non-system database ending in _test; "
            "the application database insurance_claims is forbidden"
        )


def _validate_test_host(target):
    host = target.host
    if not host:
        raise DatabaseConfigurationError("TEST_DATABASE_URL requires an explicit host")
    try:
        ip_address(host)
    except ValueError:
        if not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?", host):
            raise DatabaseConfigurationError(
                "TEST_DATABASE_URL requires one unambiguous hostname or IP address"
            ) from None
    if target.port is not None and not 1 <= target.port <= 65535:
        raise DatabaseConfigurationError("TEST_DATABASE_URL port must be between 1 and 65535")


def _validate_test_query(target):
    for key, entry in target.query.items():
        if (
            key != "sslmode"
            or not isinstance(entry, str)
            or entry
            not in {
                "disable",
                "allow",
                "prefer",
                "require",
                "verify-ca",
                "verify-full",
            }
        ):
            raise DatabaseConfigurationError(
                "TEST_DATABASE_URL query only permits one sslmode value; "
                "connection, driver, database and search_path overrides are forbidden"
            )


def _validate_distinct_from_development(target):
    development = os.environ.get("DATABASE_URL")
    if development:
        dev = DatabaseConfig.from_url(development).url
        if dev.database == target.database:
            raise DatabaseConfigurationError("Test and development databases must differ")
