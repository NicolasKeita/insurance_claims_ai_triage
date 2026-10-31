"""Portable analysis identities and atomic collection index updates."""

import json
import os
from pathlib import Path

ARCHIVE_FORMAT = "claim_analysis_archive_v1"
COLLECTION_FORMAT = "claim_analysis_collection_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "analysis_results"


def collection(root):
    path = root / "manifest.json"
    if not path.exists():
        return []
    if path.is_symlink():
        raise ValueError("Collection manifest symlinks are not allowed")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format") != COLLECTION_FORMAT or not isinstance(data.get("analyses"), list):
        raise ValueError("Unsupported analysis collection manifest")
    return data["analyses"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def atomic_write(path, text):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(text)
    os.replace(temporary, path)
