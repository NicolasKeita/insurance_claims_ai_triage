"""Separate part experiment, reusing the metadata-only step-29 MLflow logger."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vision.tracking import VisionMLflowTracking, vision_tracking_run
from vision.training import REPO_ROOT, write_json


class PartsMLflowTracking(VisionMLflowTracking):
    def __init__(self, tracking: VisionMLflowTracking, config: Any):
        super().__init__(tracking.mlflow, tracking.run_id, log_checkpoint=tracking.log_checkpoint)
        self.config = config

    def log_training(self, summary: dict[str, Any], summary_path: Path, checkpoint: Path) -> None:
        super().log_training(summary, summary_path, checkpoint)
        metadata = self.config.output_dir / "training_config.json"
        write_json(metadata, self.config.payload())
        self.log_metadata_artifacts([metadata, self.config.dataset_yaml.parent / "carparts_dataset_manifest.json"])

    def log_evaluation(self, metrics: dict[str, Any], paths: list[Path]) -> None:
        super().log_evaluation(metrics, paths)
        metadata = self.config.output_dir / "error_examples/metadata.json"
        if metadata.is_file():
            self.log_metadata_artifacts([metadata])


def _temporary_uri_directory(config: Any) -> Path | None:
    """A local SQLite store makes explicit part training reproducible without a server."""
    if not (config.track_mlflow and not os.environ.get("MLFLOW_TRACKING_URI")):
        return None
    directory = REPO_ROOT / "artifacts/vision/mlflow_parts"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + (directory / "tracking.db").as_posix()
    return directory


def _create_local_experiment(config: Any, directory: Path | None) -> None:
    if directory is None:
        return
    try:
        import mlflow
    except ImportError:
        return
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    if mlflow.get_experiment_by_name(config.experiment_name) is None:
        mlflow.create_experiment(config.experiment_name, artifact_location=(directory / "artifacts").as_uri())


def _parts_branch(tracking: VisionMLflowTracking | None, config: Any) -> VisionMLflowTracking | None:
    if tracking is None:
        return None
    tracking.mlflow.set_tags({"perception_branch": "vehicle-parts", "dataset": "Carparts-Seg"})
    return PartsMLflowTracking(tracking, config)


@contextmanager
def parts_tracking_run(config: Any, *, run_id: str | None = None) -> Iterator[VisionMLflowTracking | None]:
    """An existing configured tracking URI remains authoritative."""
    directory = _temporary_uri_directory(config)
    try:
        _create_local_experiment(config, directory)
        with vision_tracking_run(config, run_id=run_id) as tracking:
            yield _parts_branch(tracking, config)
    finally:
        if directory is not None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
