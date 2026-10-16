"""Optional MLflow logging with an explicit metadata-only artifact allowlist."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from vision.training import DamageTrainingConfig


def _flatten_metrics(payload: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}.{group}.{key}": value
        for group in ("box", "mask")
        for key, value in payload.get(group, {}).items()
        if value is not None
    }


def _per_class_metrics(payload: dict[str, Any], prefix: str) -> dict[str, float]:
    """Keep real absent-class values absent, with stable taxonomy metric keys."""
    metrics = {}
    for row in payload.get("per_class", []):
        name = str(row.get("damage_type") or row["name"].replace(" ", "_"))
        metrics.update(_flatten_metrics(row, f"{prefix}.class.{name}"))
    return metrics


class VisionMLflowTracking:
    def __init__(self, mlflow: Any, run_id: str, *, log_checkpoint: bool):
        self.mlflow, self.run_id, self.log_checkpoint = mlflow, run_id, log_checkpoint

    def log_training(self, summary: dict[str, Any], summary_path: Path, checkpoint: Path) -> None:
        dataset = summary["dataset"]
        params = {f"config.{key}": value for key, value in summary["config"].items()}
        params.update({f"environment.{key}": json.dumps(value) for key, value in summary["environment"].items()})
        params.update({f"dataset.{split}.sha256": value for split, value in dataset["annotation_hashes"].items()})
        for key in (
            "source_name",
            "source_repository",
            "source_revision",
            "actual_sample_count",
            "official_splits_used",
            "split_version",
            "split_manifest_sha256",
            "preparation_version",
            "prepared_manifest_sha256",
            "dataset_yaml_sha256",
        ):
            if dataset.get(key) is not None:
                params[f"dataset.{key}"] = dataset[key]
        for key in ("category_mapping", "annotation_fields"):
            if dataset.get(key) is not None:
                params[f"dataset.{key}"] = json.dumps(dataset[key], sort_keys=True)
        for split, counts in dataset["splits"].items():
            params[f"dataset.{split}.images"] = counts["image_count"]
            params[f"dataset.{split}.annotations"] = counts["annotation_count"]
            for name, count in counts["annotations_per_category"].items():
                params[f"dataset.{split}.{name}.source_annotations"] = count
            for name, count in counts.get("converted_annotations_per_category", {}).items():
                params[f"dataset.{split}.{name}.converted_annotations"] = count
            for name, count in counts.get("images_per_category", {}).items():
                params[f"dataset.{split}.{name}.source_images"] = count
            for name, count in counts.get("converted_images_per_category", {}).items():
                params[f"dataset.{split}.{name}.converted_images"] = count
        params["checkpoint.sha256"] = summary["best_checkpoint_sha256"]
        params["confidence_threshold"] = summary["confidence_threshold"]
        if summary.get("best_epoch") is not None:
            params["training.best_epoch"] = summary["best_epoch"]
        self.mlflow.log_params(params)
        self.mlflow.set_tags(
            {"task": "instance-segmentation", "checkpoint_selection_split": "val", "registry_status": "deferred"}
        )
        if dataset.get("metrics_scope"):
            self.mlflow.set_tag("metrics_scope", dataset["metrics_scope"])
        metrics = _flatten_metrics(summary["validation_metrics"], "validation")
        metrics.update(_per_class_metrics(summary["validation_metrics"], "validation"))
        for key in ("epochs_completed", "elapsed_seconds"):
            if summary.get(key) is not None:
                metrics[f"training.{key}"] = summary[key]
        self.mlflow.log_metrics(metrics)
        paths = [summary_path]
        results_csv = checkpoint.parent.parent / "results.csv"
        if results_csv.is_file():
            paths.append(results_csv)
        self.log_metadata_artifacts(paths)
        if self.log_checkpoint:
            self.mlflow.log_artifact(str(checkpoint), artifact_path="checkpoint")
        else:
            self.mlflow.set_tag("checkpoint_reference", summary["best_checkpoint"])

    def log_evaluation(self, metrics: dict[str, Any], paths: list[Path]) -> None:
        logged = _flatten_metrics(metrics, "test")
        logged.update(_per_class_metrics(metrics, "test"))
        self.mlflow.log_metrics(logged)
        self.mlflow.set_tag("final_evaluation_split", "test")
        self.log_metadata_artifacts(paths)

    def log_metadata_artifacts(self, paths: list[Path]) -> None:
        for path in paths:
            if path.suffix.lower() not in {".json", ".csv", ".toml"}:
                raise ValueError("Vision MLflow allowlist accepts metadata JSON/CSV/TOML only; no CarDD images")
            self.mlflow.log_artifact(str(path), artifact_path="vision_metadata")


@contextmanager
def vision_tracking_run(
    config: DamageTrainingConfig,
    *,
    run_id: str | None = None,
) -> Iterator[VisionMLflowTracking | None]:
    if not (config.track_mlflow or os.environ.get("MLFLOW_TRACKING_URI")):
        yield None
        return
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        warnings.warn(
            "Vision MLflow tracking unavailable: MLFLOW_TRACKING_URI is not configured; continuing CV without tracking.",
            RuntimeWarning,
        )
        yield None
        return
    # Reuse the repository's tracking configuration, without imports when off.
    try:
        import mlflow
        from mlflow.exceptions import MlflowException

        from ml.tracking import configure_mlflow_tracking
    except ImportError:
        warnings.warn(
            "Vision MLflow tracking unavailable: dependency import failed; continuing CV without tracking.",
            RuntimeWarning,
        )
        yield None
        return
    try:
        configure_mlflow_tracking()
        mlflow.set_experiment(config.experiment_name)
        kwargs = {"run_id": run_id} if run_id else {"run_name": config.version}
        active_run = mlflow.start_run(**kwargs)
    except MlflowException:
        # Never catch the training body: a CV failure must still propagate.
        # Avoid printing an exception containing credentials from a server URI.
        warnings.warn(
            "Vision MLflow tracking unavailable: server run initialization failed; continuing CV without tracking.",
            RuntimeWarning,
        )
        yield None
        return
    with active_run as run:
        yield VisionMLflowTracking(mlflow, run.info.run_id, log_checkpoint=config.log_checkpoint)
