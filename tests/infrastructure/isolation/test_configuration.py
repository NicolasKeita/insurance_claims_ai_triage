"""Test configuration."""

from tests.infrastructure.isolation._constants import pytestmark as pytestmark
from tests.infrastructure.isolation._support import (
    isolated_configuration as isolated_configuration,
)
from tests.integration_support import (
    DEFAULT_TEST_DATABASE_URL,
    DEFAULT_TEST_QDRANT_URL,
    get_test_database_config,
    get_test_qdrant_url,
)


def test_integration_defaults_use_dedicated_services(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://claims:claims@localhost:5433/insurance_claims")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    config = get_test_database_config()
    assert config.url.render_as_string(hide_password=False) == DEFAULT_TEST_DATABASE_URL
    assert config.url.database == "insurance_claims_test"
    assert config.url.host == "127.0.0.1" and config.url.port == 5434
    assert get_test_qdrant_url() == DEFAULT_TEST_QDRANT_URL


def test_ci_can_override_dedicated_test_services(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", "postgresql+psycopg://ci:secret@ci-db:5544/pipeline_test?sslmode=require")
    monkeypatch.setenv("TEST_QDRANT_URL", "http://ci-qdrant:6333")
    config = get_test_database_config()
    assert config.url.database == "pipeline_test" and config.url.port == 5544
    assert get_test_qdrant_url() == "http://ci-qdrant:6333"
