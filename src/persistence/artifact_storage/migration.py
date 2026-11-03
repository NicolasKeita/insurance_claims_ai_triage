"""Move local artifacts once, with preflight checks and complete content inventories."""

from datetime import UTC, datetime
from pathlib import Path

from persistence.artifact_storage.inventory import confined, inventory, write_json
from persistence.artifact_storage.layout import LEGACY_LOCATIONS, ROOT, relocate_path
from persistence.artifact_storage.mlflow import relocate_mlflow_databases


def planned_moves(root: Path) -> list[tuple[Path, Path]]:
    repository = root.resolve()
    moves = []
    for old, new in LEGACY_LOCATIONS.items():
        source = repository / "artifacts" / old
        if not source.exists() and not source.is_symlink() and not source.is_junction():
            continue
        destination = repository / "artifacts" / new
        confined(source, repository)
        confined(destination, repository)
        if destination.exists() or destination.is_symlink() or destination.is_junction():
            raise FileExistsError(f"Refusing to merge or overwrite artifacts: {destination}")
        moves.append((source, destination))
    return moves


def _verify(before, after, root, allowed_changes=()):
    expected = {relocate_path(item["path"], root=root).relative_to(root).as_posix(): item for item in before}
    actual = {item["path"]: item for item in after}
    if expected.keys() != actual.keys():
        raise ValueError("Artifact file inventory changed during relocation")
    for path, original in expected.items():
        if path not in allowed_changes and any(original[key] != actual[path][key] for key in ("bytes", "sha256")):
            raise ValueError(f"Artifact content changed during relocation: {path}")


def _move_existing(moves, repository, journal, result):
    result["completed_moves"] = []
    for source, destination in moves:
        confined(source, repository)
        confined(destination, repository)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        result["completed_moves"].append(source.relative_to(repository).as_posix())
        write_json(journal / "summary.json", result)


def migrate_artifacts(root: Path = ROOT, *, apply: bool = False) -> dict:
    repository = root.resolve()
    moves = planned_moves(repository)
    result = {
        "applied": False,
        "moves": [
            {"source": s.relative_to(repository).as_posix(), "destination": d.relative_to(repository).as_posix()}
            for s, d in moves
        ],
    }
    before = inventory([source for source, _ in moves], repository)
    result.update(files=len(before), bytes=sum(item["bytes"] for item in before))
    for item in before:
        if item["path"].endswith((".db-wal", ".db-shm", ".db-journal")):
            raise ValueError("Stop SQLite writers before relocating artifact databases")
    if not apply or not moves:
        return result
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    journal = repository / "artifacts/Historique_technique/reorganisations" / stamp
    confined(journal, repository)
    journal.mkdir(parents=True, exist_ok=False)
    write_json(journal / "before.json", before)
    write_json(journal / "summary.json", result)
    _move_existing(moves, repository, journal, result)
    after_move = inventory([destination for _, destination in moves], repository)
    _verify(before, after_move, repository)
    databases = relocate_mlflow_databases(repository, journal)
    after = inventory([destination for _, destination in moves], repository)
    _verify(before, after, repository, {entry["path"] for entry in databases})
    result.update(
        applied=True,
        files_verified=len(after),
        sqlite_updates=databases,
        journal=journal.relative_to(repository).as_posix(),
    )
    write_json(journal / "after.json", after)
    write_json(journal / "summary.json", result)
    return result
