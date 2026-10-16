"""Optional MLflow logging with an explicit metadata-only artifact allowlist."""

import json
from pathlib import Path
from typing import Any

from vision.tracking._metrics import flatten_metrics, per_class_metrics

_DATASET_SCALAR_KEYS = (
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
)


class VisionMLflowTracking:
    def __init__(self, mlflow: Any, run_id: str, *, log_checkpoint: bool):
        self.mlflow, self.run_id, self.log_checkpoint = mlflow, run_id, log_checkpoint

    def log_training(self, summary: dict[str, Any], summary_path: Path, checkpoint: Path) -> None:
        self._log_training_params(summary)
        self._log_training_metrics(summary)
        paths = [summary_path]
        results_csv = checkpoint.parent.parent / "results.csv"
        if results_csv.is_file():
            paths.append(results_csv)
        self.log_metadata_artifacts(paths)
        if self.log_checkpoint:
            self.mlflow.log_artifact(str(checkpoint), artifact_path="checkpoint")
        else:
            self.mlflow.set_tag("checkpoint_reference", summary["best_checkpoint"])

    def _log_training_params(self, summary: dict[str, Any]) -> None:
        dataset = summary["dataset"]
        params = {f"config.{key}": value for key, value in summary["config"].items()}
        params.update({f"environment.{key}": json.dumps(value) for key, value in summary["environment"].items()})
        params.update({f"dataset.{split}.sha256": value for split, value in dataset["annotation_hashes"].items()})
        for key in _DATASET_SCALAR_KEYS:
            if dataset.get(key) is not None:
                params[f"dataset.{key}"] = dataset[key]
        for key in ("category_mapping", "annotation_fields"):
            if dataset.get(key) is not None:
                params[f"dataset.{key}"] = json.dumps(dataset[key], sort_keys=True)
        self._log_split_counts(params, dataset["splits"])
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

    @staticmethod
    def _log_split_counts(params: dict[str, Any], splits: dict[str, Any]) -> None:
        for split, counts in splits.items():
            params[f"dataset.{split}.images"] = counts["image_count"]
            params[f"dataset.{split}.annotations"] = counts["annotation_count"]
            for field, suffix in (
                ("annotations_per_category", "source_annotations"),
                ("converted_annotations_per_category", "converted_annotations"),
                ("images_per_category", "source_images"),
                ("converted_images_per_category", "converted_images"),
            ):
                for name, count in counts.get(field, {}).items():
                    params[f"dataset.{split}.{name}.{suffix}"] = count

    def _log_training_metrics(self, summary: dict[str, Any]) -> None:
        metrics = flatten_metrics(summary["validation_metrics"], "validation")
        metrics.update(per_class_metrics(summary["validation_metrics"], "validation"))
        for key in ("epochs_completed", "elapsed_seconds"):
            if summary.get(key) is not None:
                metrics[f"training.{key}"] = summary[key]
        self.mlflow.log_metrics(metrics)

    def log_evaluation(self, metrics: dict[str, Any], paths: list[Path]) -> None:
        logged = flatten_metrics(metrics, "test")
        logged.update(per_class_metrics(metrics, "test"))
        self.mlflow.log_metrics(logged)
        self.mlflow.set_tag("final_evaluation_split", "test")
        self.log_metadata_artifacts(paths)

    def log_metadata_artifacts(self, paths: list[Path]) -> None:
        for path in paths:
            if path.suffix.lower() not in {".json", ".csv", ".toml"}:
                raise ValueError("Vision MLflow allowlist accepts metadata JSON/CSV/TOML only; no CarDD images")
            self.mlflow.log_artifact(str(path), artifact_path="vision_metadata")