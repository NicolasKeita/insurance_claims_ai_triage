"""Persisted evaluation artifacts, error analysis and tracking."""

from vision.evaluation._metrics import write_per_class_csv
from vision.training import write_json


def _write_base_artifacts(payload, request):
    metrics_path = request.config.output_dir / "test_metrics.json"
    write_json(metrics_path, payload)
    csv_path = request.config.output_dir / "per_class_metrics.csv"
    write_per_class_csv(csv_path, payload["per_class"])
    return metrics_path, csv_path


def _add_error_analysis(payload, model, request, params, metrics_path):
    if not params.max_error_examples:
        return
    from vision.error_analysis import ErrorExamplesConfig, save_error_examples

    examples = ErrorExamplesConfig(
        max_examples=params.max_error_examples,
        include_part_errors=params.include_part_errors,
    )
    payload["error_analysis"] = save_error_examples(
        model, request.config, request.manifest, request.threshold, examples=examples
    )
    write_json(metrics_path, payload)


def _track(payload, request, params, paths):
    from vision.tracking import vision_tracking_run

    tracking_factory = params.tracking_factory or vision_tracking_run
    with tracking_factory(request.config, run_id=request.summary.get("mlflow_run_id")) as tracking:
        if tracking is not None:
            tracking.log_evaluation(payload, list(paths))
