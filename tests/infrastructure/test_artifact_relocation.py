"""Exercise data preservation and MLflow storage references during a real local move."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from persistence.artifact_storage.migration import migrate_artifacts

pytestmark = pytest.mark.unit


def test_migration_preserves_models_and_provenance_and_updates_only_storage_uris(tmp_path):
    original = tmp_path / "artifacts/vision"
    original.mkdir(parents=True)
    checkpoint = original / "best.pt"
    checkpoint.write_bytes(b"opaque trained model\x00\xff")
    metadata = original / "training_summary.json"
    metadata.write_text(json.dumps({"checkpoint": str(checkpoint)}), encoding="utf-8")
    tracking = original / "mlflow_parts"
    tracking.mkdir()
    database = tracking / "tracking.db"
    old_uri = (tracking / "artifacts").as_uri()
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute("CREATE TABLE experiments (artifact_location TEXT)")
        connection.execute("CREATE TABLE runs (artifact_uri TEXT)")
        connection.execute("CREATE TABLE params (value TEXT)")
        connection.execute("INSERT INTO experiments VALUES (?)", (old_uri,))
        connection.execute("INSERT INTO runs VALUES (?)", (old_uri + "/run",))
        connection.execute("INSERT INTO params VALUES (?)", (str(original),))
    frozen = {path.name: path.read_bytes() for path in (checkpoint, metadata, database)}
    preview = migrate_artifacts(tmp_path)
    assert preview["applied"] is False and original.is_dir()
    assert database.read_bytes() == frozen[database.name]
    result = migrate_artifacts(tmp_path, apply=True)
    moved = tmp_path / "artifacts/Modeles/vision"
    assert result["applied"] and result["files_verified"] == 3
    assert not original.exists()
    assert (moved / checkpoint.name).read_bytes() == frozen[checkpoint.name]
    assert (moved / metadata.name).read_bytes() == frozen[metadata.name]
    with closing(sqlite3.connect(moved / "mlflow_parts/tracking.db")) as connection:
        expected_uri = (moved / "mlflow_parts/artifacts").as_uri()
        assert connection.execute("SELECT artifact_location FROM experiments").fetchone()[0] == expected_uri
        assert connection.execute("SELECT value FROM params").fetchone()[0] == str(original)
    backup = tmp_path / result["sqlite_updates"][0]["backup"]
    assert backup.read_bytes() == frozen[database.name]
    assert migrate_artifacts(tmp_path, apply=True)["moves"] == []


def test_conflicting_destination_rejects_the_entire_plan_before_any_move(tmp_path):
    for relative in ("artifacts/vision", "artifacts/Modeles/vision", "artifacts/triage_v1"):
        (tmp_path / relative).mkdir(parents=True)
    with pytest.raises(FileExistsError, match="merge or overwrite"):
        migrate_artifacts(tmp_path, apply=True)
    assert (tmp_path / "artifacts/triage_v1").is_dir()
    assert (tmp_path / "artifacts/vision").is_dir()
    assert not (tmp_path / "artifacts/Historique_technique").exists()


@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("branch", ["damage", "parts"])
def test_previous_model_environment_paths_still_find_the_same_checkpoint(monkeypatch, absolute, branch):
    from persistence.artifact_storage import ROOT
    from vision.config import VisionSettings
    from vision.parts.detector import VehiclePartSettings

    monkeypatch.chdir(ROOT)
    variable, settings = (
        ("DAMAGE_MODEL_PATH", VisionSettings) if branch == "damage" else ("PART_MODEL_PATH", VehiclePartSettings)
    )
    for threshold in ("DAMAGE_CONFIDENCE_THRESHOLD", "PART_CONFIDENCE_THRESHOLD"):
        monkeypatch.delenv(threshold, raising=False)
    previous = Path("artifacts/vision/trained/weights/best.pt")
    monkeypatch.setenv(variable, str(ROOT / previous if absolute else previous))

    assert settings.from_env().model_path.resolve() == ROOT / "artifacts/Modeles/vision/trained/weights/best.pt"


def test_previous_evaluation_configuration_paths_resolve_without_editing_the_config(monkeypatch):
    from evaluation.preflight.config import checkpoint_paths, vision_diagnostic_paths
    from persistence.artifact_storage import ROOT

    for variable in ("DAMAGE_MODEL_PATH", "PART_MODEL_PATH"):
        monkeypatch.delenv(variable, raising=False)
    previous = {"triage_checkpoint": "artifacts/triage_v1/model.joblib"}
    actual = checkpoint_paths(previous)
    assert actual["triage_checkpoint"] == ROOT / "artifacts/Modeles/triage_v1/model.joblib"
    assert previous["triage_checkpoint"] == "artifacts/triage_v1/model.joblib"
    diagnostic = vision_diagnostic_paths({"source_summary": "artifacts/vision/summary.json"})
    assert diagnostic["source_summary"] == ROOT / "artifacts/Modeles/vision/summary.json"
