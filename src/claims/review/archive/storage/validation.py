"""Resolve only archive-contained, intact regular files."""

import hashlib
import json
from pathlib import PurePosixPath


def contained(root, relative):
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or ":" in relative
        or PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
    ):
        raise ValueError("Archive path leaves its configured directory")
    candidate = root / relative
    if any(part.is_symlink() for part in (candidate, *candidate.parents)):
        raise ValueError("Archive file symlinks are not allowed")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Archive path leaves its configured directory")
    return resolved


def read_json(path):
    if path.is_symlink():
        raise ValueError("Archive manifest symlinks are not allowed")
    return json.loads(path.read_text(encoding="utf-8"))


def verified_asset(root, entry):
    if entry.get("status") != "COPIED":
        return None
    try:
        path = contained(root, entry.get("path"))
        if not path.is_file() or path.stat().st_size != entry.get("size_bytes"):
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            return None
        return path
    except (ValueError, OSError):
        return None
