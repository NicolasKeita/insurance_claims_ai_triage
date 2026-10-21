"""Reports."""

from __future__ import annotations

from datetime import datetime, timezone

from vision.evaluation._metrics import extract_segmentation_metrics
from vision.evaluation._metrics import write_per_class_csv as _write_per_class_csv
from vision.training import (
    write_json,
)


def _evaluation_payload(request, environment, result):
    config, manifest, summary, threshold = request.config, request.manifest, request.summary, request.threshold
    return {
        "schema_version": 1,
        "split": "test",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "train/weights/best.pt",
        "checkpoint_sha256": summary["best_checkpoint_sha256"],
        "checkpoint_selection_split": "val",
        "confidence_threshold": threshold,
        "threshold_selection": "configured baseline; no TEST threshold tuning",
        "environment": environment,
        "evaluation_config": config.payload(),
        "dataset": summary["dataset"],
        "metrics_scope": manifest.get("metrics_scope", "Evaluation on the configured local COCO test split."),
        **extract_segmentation_metrics(result, manifest["category_mapping"], manifest["splits"]["test"]),
    }


def _save_evaluation(request, model, payload):
    config, manifest, summary, threshold = request.config, request.manifest, request.summary, request.threshold
    max_error_examples = request.params.max_error_examples
    include_part_errors = request.params.include_part_errors
    tracking_factory = request.params.tracking_factory
    metrics_path = config.output_dir / "test_metrics.json"
    write_json(metrics_path, payload)
    csv_path = config.output_dir / "per_class_metrics.csv"
    _write_per_class_csv(csv_path, payload["per_class"])
    if max_error_examples:
        from vision.error_analysis import ErrorExamplesConfig, save_error_examples

        analysis = save_error_examples(
            model,
            config,
            manifest,
            threshold,
            examples=ErrorExamplesConfig(max_examples=max_error_examples, include_part_errors=include_part_errors),
        )
        payload["error_analysis"] = analysis
        write_json(metrics_path, payload)
    from vision.tracking import vision_tracking_run

    tracking_factory = tracking_factory or vision_tracking_run
    with tracking_factory(config, run_id=summary.get("mlflow_run_id")) as tracking:
        if tracking is not None:
            tracking.log_evaluation(payload, [metrics_path, csv_path])
