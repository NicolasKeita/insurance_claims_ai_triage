"""Test configuration."""

import pytest

from persistence.database import (
    DatabaseConfig,
    DatabaseConfigurationError,
    require_test_database,
)
from persistence.models import utc_now
from tests.domain.persistence.mapping._support import reference_claim as reference_claim


def test_database_configuration_is_explicit(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(DatabaseConfigurationError, match="required"):
        DatabaseConfig.from_env()
    with pytest.raises(DatabaseConfigurationError):
        DatabaseConfig.from_url("sqlite://")
    assert utc_now().utcoffset().total_seconds() == 0


@pytest.mark.parametrize("database", ["insurance_claims", "production", "test", "insurance_claims_test_backup"])
def test_destructive_test_guard_rejects_unsafe_names(database, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(DatabaseConfigurationError, match="_test"):
        require_test_database(f"postgresql+psycopg://localhost/{database}")


def test_test_database_must_differ_from_development(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost/insurance_claims_test")
    with pytest.raises(DatabaseConfigurationError, match="differ"):
        require_test_database("postgresql+psycopg://localhost:5432/insurance_claims_test")
