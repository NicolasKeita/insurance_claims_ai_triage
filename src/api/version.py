"""Resolve the API version from the application's single packaging source."""

from importlib.metadata import version
from pathlib import Path

import tomllib


def application_version() -> str:
    """Editable checkouts read pyproject; installed wheels use their metadata."""
    source = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if source.is_file():
        with source.open("rb") as stream:
            return tomllib.load(stream)["project"]["version"]
    return version("ai-insurance-claims")
