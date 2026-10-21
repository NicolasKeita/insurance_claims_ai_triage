"""Config."""

import os

from persistence.database import DatabaseConfig, require_test_database
from tests.infrastructure.integration_support._constants import (
    DEFAULT_TEST_DATABASE_URL,
    DEFAULT_TEST_QDRANT_URL,
)


def get_test_database_config() -> DatabaseConfig:
    """Use Compose's isolated test database unless CI explicitly overrides it."""
    return require_test_database(os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL))


def get_test_qdrant_url() -> str:
    """Never inherit the application's QDRANT_URL for integration tests."""
    value = os.environ.get("TEST_QDRANT_URL", DEFAULT_TEST_QDRANT_URL)
    if not value:
        raise ValueError("TEST_QDRANT_URL must not be empty")
    return value
