"""Durable JSON/CSV traces and intentionally small stable regression snapshots."""

import csv
import json
from pathlib import Path

CSV_FIELDS = [
    "scenario_id",
    "status",
    "passed_assertions",
    "failed_assertions",
    "skipped_assertions",
    "total_seconds",
    "trace_reference",
]


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _summary_document(report):
    summary = report.model_dump(mode="json", exclude={"results"})
    summary["scenarios"] = [
        {
            "scenario_id": result.scenario_id,
            "status": result.status,
            "trace_reference": result.trace_reference,
            "timings": result.timings,
        }
        for result in report.results
    ]
    return summary


def _write_failure(result, failures):
    bad = [assertion.model_dump(mode="json") for assertion in result.assertions if assertion.status == "FAIL"]
    failure_path = failures / f"{result.scenario_id}.json"
    if bad or result.errors:
        _json(
            failure_path,
            {
                "scenario_id": result.scenario_id,
                "assertions": bad,
                "errors": result.errors,
                "trace_reference": result.trace_reference,
            },
        )
    elif failure_path.exists():
        failure_path.unlink()


def _write_traces(report, traces, failures):
    for result in report.results:
        _json(traces / f"{result.scenario_id}.json", result.observation.model_dump(mode="json"))
        _write_failure(result, failures)


def _write_scenario_csv(report, output_dir):
    with (output_dir / "scenario_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in report.results:
            counts = {
                f"{word}_assertions": sum(assertion.status == status for assertion in result.assertions)
                for word, status in (("passed", "PASS"), ("failed", "FAIL"), ("skipped", "SKIP"))
            }
            writer.writerow(
                {
                    "scenario_id": result.scenario_id,
                    "status": result.status,
                    **counts,
                    "total_seconds": result.timings.get("total"),
                    "trace_reference": result.trace_reference,
                }
            )


def _write_timings_csv(report, output_dir):
    with (output_dir / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario_id", "component", "duration_seconds", "status"])
        for result in report.results:
            for component, duration in result.timings.items():
                status = (
                    result.component_results[component].status
                    if component in result.component_results
                    else result.status
                )
                writer.writerow([result.scenario_id, component, duration, status])


def write_report(report, output_dir):
    """Write stable JSON/CSV artifacts for one evaluation report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces, failures = output_dir / "traces", output_dir / "failures"
    traces.mkdir(exist_ok=True)
    failures.mkdir(exist_ok=True)
    _json(output_dir / "summary.json", _summary_document(report))
    _json(output_dir / "scenario_results.json", [result.model_dump(mode="json") for result in report.results])
    _write_traces(report, traces, failures)
    _write_scenario_csv(report, output_dir)
    _write_timings_csv(report, output_dir)
    return output_dir
