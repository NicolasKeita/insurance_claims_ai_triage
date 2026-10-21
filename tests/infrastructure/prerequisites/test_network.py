"""Test network."""

from urllib.error import HTTPError, URLError

import pytest


def test_postgres_refused_connection_is_not_a_skip(monkeypatch):
    import tests.integration_prerequisites._network as support
    from tests.integration_support import get_test_database_config

    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    def refused(*args, **kwargs):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(support.socket, "create_connection", refused)
    with pytest.raises(support.PrerequisiteError, match="PostgreSQL connection refused") as error:
        support.require_postgres_connection(get_test_database_config())
    assert "--wait postgres-test" in str(error.value)
    assert ":claims_test@" not in str(error.value)


@pytest.mark.parametrize(
    "failure,message",
    [
        (URLError(ConnectionRefusedError("refused")), "Qdrant connection refused"),
        (HTTPError("http://test", 503, "not ready", {}, None), "HTTP 503"),
        (HTTPError("http://test", 401, "unauthorized", {}, None), "HTTP 401"),
    ],
)
def test_qdrant_network_and_readiness_diagnostics(monkeypatch, failure, message):
    import tests.integration_prerequisites._network as support

    def unavailable(*args, **kwargs):
        raise failure

    monkeypatch.setattr(support, "urlopen", unavailable)
    with pytest.raises(support.PrerequisiteError, match=message):
        support.require_qdrant_connection("http://127.0.0.1:6335")
