"""Versioned summaries and bounded baseline comparison for future regressions."""

import json

from evaluation.deterministic.document_robustness import DOCUMENT_OCR_BENCHMARK_VERSION
from evaluation.deterministic.document_robustness.metrics import compare_baseline
from evaluation.deterministic.document_robustness.reporting.aggregation import aggregate, critical_fields, ratio
from evaluation.deterministic.document_robustness.reporting.artifacts import (
    document_rows,
    write_csv,
    write_debug_artifacts,
)
from evaluation.deterministic.document_robustness.runtime.consistency import consistency_results
from evaluation.deterministic.document_robustness.runtime.storage import timestamp, write_json
from evaluation.deterministic.document_robustness.transformations import VARIANTS


def write_reports(output, identity, data):
    scenarios, records = data
    consistency = consistency_results(scenarios, records)
    degraded = [row for row in consistency if row["variant"] != "native_clean"]
    evaluable = [row for row in degraded if row["evaluable"]]
    metrics = aggregate(records)
    metrics.update(
        consistency_preservation_rate=ratio(sum(row["preserved"] for row in evaluable), len(evaluable)),
        consistency_evaluable_count=len(evaluable),
        consistency_degraded_group_count=len(degraded),
    )
    summary = {
        "benchmark_version": DOCUMENT_OCR_BENCHMARK_VERSION,
        "timestamp": timestamp(),
        "identity": identity,
        "corpus": {
            "scenarios": len(scenarios),
            "documents": sum(len(s.documents) for s in scenarios),
            "variants": list(VARIANTS),
            "total_cases": sum(len(s.documents) for s in scenarios) * len(VARIANTS),
        },
        "recorded_cases": len(records),
        "metrics": metrics,
        "by_variant": {
            key: aggregate([r for r in records if r["variant"] == key])
            for key in VARIANTS
            if any(r["variant"] == key for r in records)
        },
        "by_document_type": {
            key: aggregate([r for r in records if r["document_type"] == key])
            for key in sorted({r["document_type"] for r in records})
        },
        "critical_fields": critical_fields(records),
        "consistency": consistency,
        "documents": document_rows(records),
    }
    _write_exports(output, summary, records)
    return summary


def _write_exports(output, summary, records):
    write_json(output / "summary.json", summary)
    write_csv(output / "per_document.csv", summary["documents"])
    write_csv(
        output / "per_variant.csv", [{"variant": name, **values} for name, values in summary["by_variant"].items()]
    )
    write_csv(
        output / "per_document_type.csv",
        [{"document_type": name, **values} for name, values in summary["by_document_type"].items()],
    )
    write_debug_artifacts(output, records)


def create_baseline(summary, path):
    expected = summary["corpus"]["total_cases"]
    if summary["recorded_cases"] != expected or summary["metrics"]["evaluated_count"] != expected:
        raise ValueError("Baseline requires all corpus documents and variants actually executed")
    snapshot = {
        key: summary[key]
        for key in (
            "benchmark_version",
            "timestamp",
            "identity",
            "corpus",
            "recorded_cases",
            "metrics",
            "by_variant",
            "by_document_type",
            "critical_fields",
        )
    }
    snapshot["documents"] = [
        {key: row[key] for key in ("document_id", "variant", "parse_success")} for row in summary["documents"]
    ]
    write_json(path, snapshot, exclusive=True)


def compare_saved_baseline(path, summary):
    baseline = json.loads(path.read_text(encoding="utf-8"))
    expected = summary["corpus"]["total_cases"]
    if summary["recorded_cases"] != expected or baseline.get("recorded_cases") != expected:
        raise ValueError("Baseline comparison requires two complete runs")
    previous_keys = {(row["document_id"], row["variant"]) for row in baseline["documents"]}
    current_keys = {(row["document_id"], row["variant"]) for row in summary["documents"]}
    if previous_keys != current_keys:
        raise ValueError("Baseline comparison requires identical document/variant populations")
    for key in ("identity", "recorded_cases"):
        if baseline.get(key) != summary.get(key):
            raise ValueError(f"Baseline comparison requires equal {key}")
    return compare_baseline(baseline, summary)
