"""Relocate operational SQLite storage URIs, preserving parameters and original databases."""

import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from persistence.artifact_storage.inventory import confined, sha256
from persistence.artifact_storage.layout import relocate_path

DATABASES = ("Modeles/vision/mlflow_parts/tracking.db", "Experiences_MLflow/mlflow.db")
URI_COLUMNS = (("experiments", "artifact_location"), ("runs", "artifact_uri"))


def _relocated_uri(value: str, root: Path) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return value
    original = Path(url2pathname(parsed.path)).resolve()
    moved = relocate_path(original, root=root)
    return moved.as_uri() if moved != original else value


def _changes(connection, root):
    changes = []
    for table, column in URI_COLUMNS:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if not exists:
            continue
        for rowid, old in connection.execute(f'SELECT rowid, "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL'):
            new = _relocated_uri(old, root)
            if new != old:
                changes.append((table, column, rowid, old, new))
    return changes


def relocate_mlflow_databases(root: Path, journal: Path) -> list[dict]:
    updated = []
    for relative in DATABASES:
        database = root / "artifacts" / relative
        if not database.is_file():
            continue
        confined(database, root)
        with closing(sqlite3.connect(database, timeout=5)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            changes = _changes(connection, root)
            if not changes:
                connection.rollback()
                continue
            backup = journal / "sqlite_backups" / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(database, backup)
            original_hash = sha256(database)
            if sha256(backup) != original_hash:
                raise ValueError(f"SQLite backup differs from original: {database}")
            for table, column, rowid, _old, new in changes:
                connection.execute(f'UPDATE "{table}" SET "{column}"=? WHERE rowid=?', (new, rowid))
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError(f"SQLite integrity check failed: {database}")
            connection.commit()
            updated.append(
                {
                    "path": database.relative_to(root).as_posix(),
                    "rows_updated": len(changes),
                    "backup": backup.relative_to(root).as_posix(),
                    "original_sha256": original_hash,
                    "changes": [{"table": t, "column": c, "rowid": r, "old": o, "new": n} for t, c, r, o, n in changes],
                }
            )
    return updated
