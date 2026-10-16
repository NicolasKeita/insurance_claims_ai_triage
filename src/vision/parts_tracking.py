"""Separate part experiment, reusing the metadata-only step-29 MLflow logger."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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


@contextmanager
def parts_tracking_run(config: Any, *, run_id: str | None = None) -> Iterator[VisionMLflowTracking | None]:
    # A local SQLite store makes explicit part training reproducible without a
    # server. Existing configured tracking URI remains authoritative.
    temporary_uri = bool(config.track_mlflow and not os.environ.get("MLFLOW_TRACKING_URI"))
    if temporary_uri:
        directory = REPO_ROOT / "artifacts/vision/mlflow_parts"
        directory.mkdir(parents=True, exist_ok=True)
        os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + (directory / "tracking.db").as_posix()
    try:
        if temporary_uri:
            try:
                import mlflow
            except ImportError:
                mlflow = None
            if mlflow is not None:
                mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
                if mlflow.get_experiment_by_name(config.experiment_name) is None:
                    mlflow.create_experiment(
                        config.experiment_name, artifact_location=(directory / "artifacts").as_uri()
                    )
        with vision_tracking_run(config, run_id=run_id) as tracking:
            if tracking is not None:
                tracking.mlflow.set_tags({"perception_branch": "vehicle-parts", "dataset": "Carparts-Seg"})
                tracking = PartsMLflowTracking(tracking, config)
            yield tracking
    finally:
        if temporary_uri:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
