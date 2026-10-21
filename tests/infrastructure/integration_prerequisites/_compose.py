"""Compose."""

import json
import shutil
import subprocess
from pathlib import Path

from tests.infrastructure.integration_prerequisites._errors import PrerequisiteError, service_hint


def _docker_read(arguments, root):
    try:
        return subprocess.run(arguments, cwd=root, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PrerequisiteError(f"Docker inspection failed: {error}") from None


def require_compose_services(services, root):
    """Inspect only requested local Compose services; all commands are read-only."""
    if not services:
        return
    docker = _require_docker_engine(root, services)
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
            _require_container(docker, root, service, container_id)


def _require_docker_engine(root, services):
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
    return docker


def _require_container(docker, root, service, container_id):
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
            f"Test container {service} is stopped (state: {state.get('Status', 'unknown')})." + service_hint(service)
        )
    health = state.get("Health", {}).get("Status", "missing healthcheck")
    if health != "healthy":
        raise PrerequisiteError(f"Test container {service} is not healthy (health: {health})." + service_hint(service))
