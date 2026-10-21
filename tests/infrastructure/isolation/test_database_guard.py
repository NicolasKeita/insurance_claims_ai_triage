"""Test database guard."""

import pytest

from persistence.database import DatabaseConfigurationError, require_test_database
from tests.infrastructure.isolation._constants import pytestmark as pytestmark
from tests.infrastructure.isolation._support import (
    isolated_configuration as isolated_configuration,
)
from tests.integration_support import (
    DEFAULT_TEST_DATABASE_URL,
    get_test_database_config,
    get_test_qdrant_url,
)


@pytest.mark.parametrize(
    "database",
    [
        "insurance_claims",
        "postgres",
        "template0",
        "template1",
        "production",
        "test",
        "insurance_claims_test_backup",
        "public/claims_test",
        "claims%255Ftest",
        "claims_test;dbname=insurance_claims",
        "claims_test#fragment",
        "Claims_test",
        "x" * 60 + "_test",
    ],
)
def test_database_guard_rejects_unsafe_or_ambiguous_names(database):
    with pytest.raises(DatabaseConfigurationError):
        require_test_database(f"postgresql+psycopg://claims:secret@localhost:5434/{database}")


@pytest.mark.parametrize(
    "query",
    [
        "dbname=insurance_claims",
        "database=insurance_claims",
        "host=production",
        "hostaddr=10.0.0.1",
        "port=5432",
        "user=postgres",
        "service=production",
        "options=-csearch_path%3Dpublic",
        "options=-crole%3Dpostgres",
        "search_path=public",
        "connect_timeout=600",
        "driver=psycopg2",
        "schema=public",
        "sslmode=require&sslmode=disable",
        "sslmode=anything",
        "SSLMODE=require",
    ],
)
def test_database_guard_blocks_libpq_connection_and_schema_overrides(query):
    with pytest.raises(DatabaseConfigurationError):
        require_test_database(DEFAULT_TEST_DATABASE_URL + "?" + query)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql://localhost/claims_test",
        "postgresql+psycopg2://localhost/claims_test",
        "sqlite:///claims_test",
        "postgresql+psycopg:///claims_test",
        "postgresql+psycopg://host1,host2/claims_test",
        "postgresql+psycopg://local%68ost/claims_test",
        "postgresql+psycopg://localhost:0/claims_test",
        "postgresql+psycopg://localhost:65536/claims_test",
        "postgresql+psycopg://localhost:invalid/claims_test",
    ],
)
def test_database_guard_rejects_implicit_or_ambiguous_connection_targets(url):
    with pytest.raises(DatabaseConfigurationError):
        require_test_database(url)


def test_database_guard_rejects_development_name_even_through_another_hostname(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://localhost:5432/shared_test")
    with pytest.raises(DatabaseConfigurationError, match="differ"):
        require_test_database("postgresql+psycopg://127.0.0.1:5434/shared_test")


def test_explicit_default_port_cannot_inherit_pgport(monkeypatch):
    monkeypatch.setenv("PGPORT", "5433")
    assert require_test_database("postgresql+psycopg://localhost/claims_test").url.port == 5432


@pytest.mark.parametrize("variable", ["TEST_DATABASE_URL", "TEST_QDRANT_URL"])
def test_empty_explicit_override_is_an_error(variable, monkeypatch):
    monkeypatch.setenv(variable, "")
    with pytest.raises(ValueError):
        (get_test_database_config if variable == "TEST_DATABASE_URL" else get_test_qdrant_url)()
