"""Preflight rows, retained resources and version fingerprints."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PreflightReport:
    rows: list[dict[str, Any]] = field(default_factory=list)
    resources: dict[str, Any] = field(default_factory=dict, repr=False)

    def available(self, name: str) -> bool:
        return any(row["component"] == name and row["status"] == "PASS" for row in self.rows)

    def reason(self, name: str) -> str:
        return next((row.get("reason", "") for row in self.rows if row["component"] == name), "Not preflighted")

    def close(self):
        engine = self.resources.get("database_engine")
        if engine is not None:
            engine.dispose()
        client = self.resources.get("qdrant_client")
        if client is not None:
            client.close()


def checkpoint_version(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def print_preflight_row(row: dict) -> None:
    print(
        f"{row['component']:<27} {row['status']:<5} {row.get('version') or '-'} {row.get('reason') or ''}",
        flush=True,
    )
