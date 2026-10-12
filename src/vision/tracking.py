"""Optional MLflow logging with an explicit metadata-only artifact allowlist."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from vision.training import DamageTrainingConfig


def _flatten_metrics(payload: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}.{group}.{key}": value
        for group in ("box", "mask")
        for key, value in payload.get(group, {}).items()
        if value is not None
    }


class VisionMLflowTracking:
    def __init__(self, mlflow: Any, run_id: str, *, log_checkpoint: bool):
        self.mlflow, self.run_id, self.log_checkpoint = mlflow, run_id, log_checkpoint

    def log_training(self, summary: dict[str, Any], summary_path: Path, checkpoint: Path) -> None:
        params = {f"config.{key}": value for key, value in summary["config"].items()}
        params.update({f"environment.{key}": json.dumps(value) for key, value in summary["environment"].items()})
        params.update({f"dataset.{split}.sha256": value for split, value in summary["dataset"]["annotation_hashes"].items()})
        for split, counts in summary["dataset"]["splits"].items():
            params[f"dataset.{split}.images"] = counts["image_count"]
            params[f"dataset.{split}.annotations"] = counts["annotation_count"]
            for name, count in counts["annotations_per_category"].items():
                params[f"dataset.{split}.{name}.source_annotations"] = count
            for name, count in counts.get("converted_annotations_per_category", {}).items():
                params[f"dataset.{split}.{name}.converted_annotations"] = count
        params["checkpoint.sha256"] = summary["best_checkpoint_sha256"]
        params["confidence_threshold"] = summary["confidence_threshold"]
        self.mlflow.log_params(params)
        self.mlflow.set_tags({"task": "instance-segmentation", "checkpoint_selection_split": "val", "registry_status": "deferred"})
        self.mlflow.log_metrics(_flatten_metrics(summary["validation_metrics"], "validation"))
        self.log_metadata_artifacts([summary_path])
        if self.log_checkpoint:
            self.mlflow.log_artifact(str(checkpoint), artifact_path="checkpoint")
        else:
            self.mlflow.set_tag("checkpoint_reference", summary["best_checkpoint"])

    def log_evaluation(self, metrics: dict[str, Any], paths: list[Path]) -> None:
        self.mlflow.log_metrics(_flatten_metrics(metrics, "test"))
        self.mlflow.set_tag("final_evaluation_split", "test")
        self.log_metadata_artifacts(paths)

    def log_metadata_artifacts(self, paths: list[Path]) -> None:
        for path in paths:
            if path.suffix.lower() not in {".json", ".csv", ".toml"}:
                raise ValueError("Vision MLflow allowlist accepts metadata JSON/CSV/TOML only; no CarDD images")
            self.mlflow.log_artifact(str(path), artifact_path="vision_metadata")


@contextmanager
def vision_tracking_run(
    config: DamageTrainingConfig, *, run_id: str | None = None,
) -> Iterator[VisionMLflowTracking | None]:
    if not (config.track_mlflow or os.environ.get("MLFLOW_TRACKING_URI")):
        yield None
        return
    # Reuse the repository's tracking configuration, without imports when off.
    import mlflow
    from ml.tracking import configure_mlflow_tracking
    configure_mlflow_tracking()
    mlflow.set_experiment(config.experiment_name)
    kwargs = {"run_id": run_id} if run_id else {"run_name": config.version}
    with mlflow.start_run(**kwargs) as run:
        yield VisionMLflowTracking(mlflow, run.info.run_id, log_checkpoint=config.log_checkpoint)
