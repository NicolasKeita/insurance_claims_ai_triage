"""Selection."""

from urllib.parse import urlsplit

from tests.infrastructure.integration_prerequisites._cache import require_cached_rag_model
from tests.infrastructure.integration_prerequisites._compose import require_compose_services
from tests.infrastructure.integration_prerequisites._errors import PrerequisiteError
from tests.infrastructure.integration_prerequisites._network import (
    require_postgres_connection,
    require_qdrant_connection,
)


def _loopback(host):
    return host in {"127.0.0.1", "localhost", "::1"}


def check_selected_prerequisites(markers, root):
    """
    Unit and vision selections must not even inspect Docker.
    """
    needs_postgres = "postgres" in markers
    needs_qdrant = bool({"qdrant", "rag"}.intersection(markers))
    if not needs_postgres and not needs_qdrant:
        return
    from persistence.database import DatabaseConfig
    from tests.infrastructure.integration_support import (
        DEFAULT_TEST_DATABASE_URL,
        DEFAULT_TEST_QDRANT_URL,
        get_test_database_config,
        get_test_qdrant_url,
    )

    try:
        pg = get_test_database_config() if needs_postgres else None
        qdrant = get_test_qdrant_url() if needs_qdrant else None
        services = []
        if pg:
            default = DatabaseConfig.from_url(DEFAULT_TEST_DATABASE_URL).url
            if _loopback(pg.url.host) and (pg.url.port, pg.url.database) == (default.port, default.database):
                services.append("postgres-test")
        if qdrant:
            url, default = urlsplit(qdrant), urlsplit(DEFAULT_TEST_QDRANT_URL)
            if _loopback(url.hostname) and (url.scheme, url.port) == (default.scheme, default.port):
                services.append("qdrant-test")
    except ValueError as error:
        raise PrerequisiteError(f"Invalid integration test configuration: {error}") from None
    require_compose_services(services, root)
    if pg:
        require_postgres_connection(pg)
    if qdrant:
        require_qdrant_connection(qdrant)
    if "rag" in markers:
        require_cached_rag_model()
