"""Small serial selections and explicit baseline operations."""

import argparse
import json
from pathlib import Path

from evaluation.deterministic.document_robustness.runtime.preflight import ocr_preflight


def parser():
    result = argparse.ArgumentParser(description="CPU-only Document AI robustness evaluation")
    for name in ("document-type", "variant", "scenario"):
        result.add_argument(f"--{name}", action="append", help="Repeat to select several values; default: all")
    result.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation/document_ocr_v1"))
    result.add_argument("--resume", action="store_true", help="Reuse recorded cases without overwriting them")
    result.add_argument("--language", default="eng")
    result.add_argument("--tesseract-cmd", help="Optional existing executable; never installs system packages")
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--write-baseline", type=Path, help="Create a new baseline only after a complete run")
    result.add_argument("--baseline", type=Path, help="Compare an existing baseline from an equivalent complete run")
    return result


def main(argv=None):
    arguments = parser().parse_args(argv)
    preflight = ocr_preflight(arguments.language, arguments.tesseract_cmd)
    if arguments.preflight_only or "pypdf" in preflight.get("missing_imports", []):
        print(json.dumps(preflight, indent=2))
        return 0 if preflight["available"] else 2
    from evaluation.deterministic.document_robustness.reporting import compare_saved_baseline, create_baseline
    from evaluation.deterministic.document_robustness.reporting.artifacts import print_table
    from evaluation.deterministic.document_robustness.runtime.session import BenchmarkConfig, run_benchmark

    config = BenchmarkConfig(
        output_dir=arguments.output_dir,
        scenarios=tuple(arguments.scenario or ()),
        variants=tuple(arguments.variant or ()),
        document_types=tuple(arguments.document_type or ()),
        resume=arguments.resume,
        language=arguments.language,
        tesseract_cmd=arguments.tesseract_cmd,
    )
    try:
        summary = run_benchmark(config)
        print_table(summary)
        if arguments.write_baseline:
            create_baseline(summary, arguments.write_baseline)
        if arguments.baseline:
            regressions = compare_saved_baseline(arguments.baseline, summary)
            print(json.dumps({"regressions": regressions}, indent=2))
            return 1 if regressions else 0
        return 2 if summary["metrics"]["not_executed_count"] else 0
    except (ValueError, FileExistsError, OSError) as error:
        print(f"Document benchmark error: {error}")
        return 2
