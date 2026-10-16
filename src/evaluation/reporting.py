"""Durable JSON/CSV traces and intentionally small stable regression snapshots."""

import csv
import json
from pathlib import Path


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_report(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    traces, failures = output_dir / "traces", output_dir / "failures"
    traces.mkdir(exist_ok=True)
    failures.mkdir(exist_ok=True)
    summary = report.model_dump(mode="json", exclude={"results"})
    summary["scenarios"] = [{"scenario_id": r.scenario_id, "status": r.status,
        "trace_reference": r.trace_reference, "timings": r.timings} for r in report.results]
    _json(output_dir / "summary.json", summary)
    _json(output_dir / "scenario_results.json", [r.model_dump(mode="json") for r in report.results])
    for result in report.results:
        _json(traces / f"{result.scenario_id}.json", result.observation.model_dump(mode="json"))
        bad = [a.model_dump(mode="json") for a in result.assertions if a.status == "FAIL"]
        failure_path = failures / f"{result.scenario_id}.json"
        if bad or result.errors:
            _json(failure_path, {"scenario_id": result.scenario_id, "assertions": bad,
                "errors": result.errors, "trace_reference": result.trace_reference})
        elif failure_path.exists():
            failure_path.unlink()
    with (output_dir / "scenario_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scenario_id", "status", "passed_assertions", "failed_assertions", "skipped_assertions", "total_seconds", "trace_reference"])
        writer.writeheader()
        for result in report.results:
            writer.writerow({"scenario_id": result.scenario_id, "status": result.status,
                **{f"{word}_assertions": sum(a.status == status for a in result.assertions)
                   for word, status in (("passed", "PASS"), ("failed", "FAIL"), ("skipped", "SKIP"))},
                "total_seconds": result.timings.get("total"), "trace_reference": result.trace_reference})
    with (output_dir / "timings.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario_id", "component", "duration_seconds", "status"])
        for result in report.results:
            for component, duration in result.timings.items():
                writer.writerow([result.scenario_id, component, duration,
                    result.component_results[component].status if component in result.component_results else result.status])
    return output_dir


def baseline_snapshot(report):
    """No exact latency, prose or floating point confidence is retained."""
    return {"benchmark_version": report.benchmark_version, "mode": report.mode,
        "full_stack_status": report.full_stack_status,
        "scenarios": {r.scenario_id: {
            "status": r.status,
            "structured_expectations": {a.name: {"expected": a.expected, "status": a.status}
                for a in r.assertions if a.name.startswith(("fact:", "required_signal:", "forbidden_signal:",
                    "rag:expected_sources", "rag:expected_section", "rag:citation_validity", "agent:evidence_citations",
                    "agent:acceptable_action", "agent:human_review_required", "agent:final_schema", "schema:"))},
            "signals": r.observation.signals,
            "model_versions": r.observation.model_versions,
            "policy_versions": r.observation.policy_versions,
        } for r in report.results}}


def create_baseline(report, path):
    """Withhold a real-stack baseline until the complete requested stack passed."""
    if report.mode != "real" or report.full_stack_status != "COMPLETE" or not report.results or any(r.status != "PASS" for r in report.results):
        return False
    _json(Path(path), baseline_snapshot(report))
    return True


def compare_baseline(report, baseline):
    if isinstance(baseline, (str, Path)):
        baseline = json.loads(Path(baseline).read_text(encoding="utf-8"))
    if baseline["benchmark_version"] != report.benchmark_version:
        raise ValueError("Benchmark version mismatch; scenario expectations cannot be compared")
    if baseline.get("mode") != report.mode:
        raise ValueError("Evaluation mode mismatch; real and deterministic baselines cannot be compared")
    current = baseline_snapshot(report)
    regressions = []
    for sid, previous in baseline["scenarios"].items():
        actual = current["scenarios"].get(sid)
        if actual is None:
            regressions.append({"scenario_id": sid, "kind": "missing_scenario", "expected": previous["status"], "actual": None})
            continue
        if previous["status"] == "PASS" and actual["status"] != "PASS":
            regressions.append({"scenario_id": sid, "kind": "scenario_regression", "expected": "PASS", "actual": actual["status"]})
        for name, expectation in previous["structured_expectations"].items():
            now = actual["structured_expectations"].get(name)
            if now is not None and now["expected"] != expectation["expected"]:
                regressions.append({"scenario_id": sid, "kind": "benchmark_expectation_changed",
                    "expectation": name, "expected": expectation["expected"], "actual": now["expected"]})
            if expectation["status"] == "PASS" and (now is None or now["status"] != "PASS"):
                regressions.append({"scenario_id": sid, "kind": name, "expected": expectation, "actual": now})
        # Forbidden signals are gates, not all newly observed legitimate ML signals.
        for name, now in actual["structured_expectations"].items():
            if name.startswith("forbidden_signal:") and now["status"] == "FAIL" and name not in previous["structured_expectations"]:
                regressions.append({"scenario_id": sid, "kind": name, "expected": "absent", "actual": "present"})
    return regressions
