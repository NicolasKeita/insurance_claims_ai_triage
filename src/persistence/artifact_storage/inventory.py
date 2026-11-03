"""Confinement and content fingerprints for a lossless local directory relocation."""

import hashlib
import json
from pathlib import Path


def confined(path: Path, root: Path) -> Path:
    """Reject links, junctions and paths escaping the explicitly selected repository."""
    absolute = path.absolute()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"Artifact path leaves the repository: {path}")
    cursor = absolute
    while cursor != root:
        if cursor.is_symlink() or cursor.is_junction():
            raise ValueError(f"Artifact path contains a link or junction: {cursor}")
        if cursor.parent == cursor:
            raise ValueError(f"Artifact path has no repository ancestor: {path}")
        cursor = cursor.parent
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(paths: list[Path], root: Path) -> list[dict]:
    records = []
    pending = list(paths)
    while pending:
        path = confined(pending.pop(), root)
        if path.is_dir():
            pending.extend(path.iterdir())
        elif path.is_file():
            records.append(
                {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
        else:
            raise ValueError(f"Unsupported artifact entry: {path}")
    return sorted(records, key=lambda item: item["path"])


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
