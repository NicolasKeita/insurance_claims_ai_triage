"""Read-only prerequisite diagnostics. Never start services or download assets."""

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class PrerequisiteError(RuntimeError):
    pass


def service_hint(*services):
    return (
        "\n\nStart the test containers manually:\n\n"
        "docker compose --profile test up -d --no-deps --wait " + " ".join(services) + "\n\nRetry your pytest command."
    )


def _docker_read(arguments, root):
    try:
        return subprocess.run(arguments, cwd=root, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PrerequisiteError(f"Docker inspection failed: {error}") from None


def require_compose_services(services, root):
    """Inspect only requested local Compose services; all commands are read-only."""
    if not services:
        return
    docker = shutil.which("docker")
    if docker is None:
        raise PrerequisiteError(
            "Docker CLI is not installed or is not on PATH.\n\n"
            "Docker is required for the selected local service integrations. "
            "Install/configure Docker Desktop on Windows/macOS or Docker on Linux." + service_hint(*services)
        )
    info = _docker_read([docker, "info", "--format", "{{.ServerVersion}}"], root)
    if info.returncode:
        raise PrerequisiteError(
            "Docker Engine is not available.\n\n"
            "Docker is required for the selected local service integrations.\n\n"
            "Start Docker Desktop on Windows/macOS, or start your Docker service on Linux."
            + service_hint(*services)
            + f"\n\nDocker diagnostic: {info.stderr.strip()}"
        )
    for service in services:
        containers = _docker_read(
            [
                docker,
                "compose",
                "--file",
                str(Path(root) / "compose.yaml"),
                "--profile",
                "test",
                "ps",
                "--all",
                "--quiet",
                service,
            ],
            root,
        )
        if containers.returncode:
            raise PrerequisiteError(
                f"Docker Compose could not inspect {service}: {containers.stderr.strip()}" + service_hint(service)
            )
        ids = containers.stdout.split()
        if not ids:
            raise PrerequisiteError(f"Test container {service} has not been created." + service_hint(service))
        for container_id in ids:
            result = _docker_read([docker, "inspect", "--format", "{{json .State}}", container_id], root)
            try:
                if result.returncode:
                    raise ValueError(result.stderr.strip())
                state = json.loads(result.stdout)
                if not isinstance(state, dict):
                    raise ValueError("expected a container state object")
            except (ValueError, TypeError) as error:
                raise PrerequisiteError(f"Cannot inspect test container {service}: {error}") from None
            if not state.get("Running"):
                raise PrerequisiteError(
                    f"Test container {service} is stopped (state: {state.get('Status', 'unknown')})."
                    + service_hint(service)
                )
            health = state.get("Health", {}).get("Status", "missing healthcheck")
            if health != "healthy":
                raise PrerequisiteError(
                    f"Test container {service} is not healthy (health: {health})." + service_hint(service)
                )


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
    try:
        target = urlsplit(url)
        if target.scheme not in {"http", "https"} or not target.hostname or target.username or target.password:
            raise ValueError("expected an HTTP(S) URL without embedded credentials")
        _ = target.port  # Reject malformed ports before making a request.
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


def require_cached_rag_model():
    from huggingface_hub import try_to_load_from_cache

    from retrieval.config import DEFAULT_EMBEDDING_MODEL

    def cached(filename):
        try:
            result = try_to_load_from_cache(DEFAULT_EMBEDDING_MODEL, filename)
        except OSError as error:
            raise PrerequisiteError(f"RAG embedding cache cannot be read: {error}") from None
        return isinstance(result, str) and Path(result).is_file()

    if not all(cached(name) for name in ("modules.json", "config.json")) or not any(
        cached(name) for name in ("model.safetensors", "pytorch_model.bin")
    ):
        raise PrerequisiteError(
            f"RAG embedding model is not present in the local cache: {DEFAULT_EMBEDDING_MODEL}.\n"
            "Populate its cache deliberately before running pytest -m rag. "
            "Pytest never downloads embedding models."
        )


def _loopback(host):
    return host in {"127.0.0.1", "localhost", "::1"}


def check_selected_prerequisites(markers, root):
    needs_postgres = "postgres" in markers
    needs_qdrant = bool({"qdrant", "rag"}.intersection(markers))
    if not needs_postgres and not needs_qdrant:
        return  # Unit and vision selections must not even inspect Docker.
    from integration_support import (
        DEFAULT_TEST_DATABASE_URL,
        DEFAULT_TEST_QDRANT_URL,
        get_test_database_config,
        get_test_qdrant_url,
    )

    from persistence.database import DatabaseConfig

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
