"""Network."""

import os
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from tests.infrastructure.integration_prerequisites._errors import PrerequisiteError, service_hint


def require_postgres_connection(config):
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError

    endpoint = config.url.render_as_string(hide_password=True)
    try:
        with socket.create_connection((config.url.host, config.url.port), timeout=3):
            pass
    except ConnectionRefusedError:
        raise PrerequisiteError(
            f"PostgreSQL connection refused at {endpoint}. Check TEST_DATABASE_URL." + service_hint("postgres-test")
        ) from None
    except OSError as error:
        raise PrerequisiteError(f"PostgreSQL connection unavailable at {endpoint}: {error}") from None
    engine = create_engine(
        config.url, hide_parameters=True, connect_args={"connect_timeout": 3, "options": "-csearch_path=pg_catalog"}
    )
    try:
        with engine.connect() as connection:
            actual = connection.scalar(text("SELECT current_database()"))
            if actual != config.url.database:
                raise PrerequisiteError("PostgreSQL server database differs from the authorized test database.")
    except SQLAlchemyError as error:
        raise PrerequisiteError(
            f"PostgreSQL connection failed at {endpoint}: {getattr(error, 'orig', error)}"
        ) from None
    finally:
        engine.dispose()


def require_qdrant_connection(url):
    """
    Reject malformed ports before making a request.
    """
    try:
        target = urlsplit(url)
        if target.scheme not in {"http", "https"} or not target.hostname or target.username or target.password:
            raise ValueError("expected an HTTP(S) URL without embedded credentials")
        _ = target.port
    except ValueError as error:
        raise PrerequisiteError(f"Invalid TEST_QDRANT_URL: {error}") from None
    headers = {}
    if os.environ.get("TEST_QDRANT_API_KEY"):
        headers["api-key"] = os.environ["TEST_QDRANT_API_KEY"]
    try:
        with urlopen(Request(url.rstrip("/") + "/readyz", headers=headers), timeout=3) as response:
            if response.status != 200:
                raise PrerequisiteError(f"Qdrant is not ready at {url}: HTTP {response.status}")
    except HTTPError as error:
        raise PrerequisiteError(f"Qdrant readiness failed at {url}: HTTP {error.code}") from None
    except URLError as error:
        reason = "connection refused" if isinstance(error.reason, ConnectionRefusedError) else str(error.reason)
        raise PrerequisiteError(
            f"Qdrant {reason} at {url}. Check TEST_QDRANT_URL." + service_hint("qdrant-test")
        ) from None
    except OSError as error:
        raise PrerequisiteError(f"Qdrant connection unavailable at {url}: {error}") from None
