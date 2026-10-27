"""Incremental orchestration; no vision imports, services or hidden retries."""

import json
from dataclasses import dataclass
from pathlib import Path

from evaluation.deterministic.document_robustness.corpus import build_corpus
from evaluation.deterministic.document_robustness.runtime.cases import evaluate_document
from evaluation.deterministic.document_robustness.runtime.preflight import ocr_preflight
from evaluation.deterministic.document_robustness.runtime.storage import (
    corpus_payload,
    load_records,
    prepare_run,
    run_identity,
    timestamp,
    write_json,
)
from evaluation.deterministic.document_robustness.transformations import VARIANTS


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: Path = Path("artifacts/evaluation/document_ocr_v1")
    scenarios: tuple[str, ...] = ()
    variants: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    resume: bool = False
    language: str = "eng"
    tesseract_cmd: str | None = None


def run_benchmark(config):
    from evaluation.deterministic.document_robustness.reporting import write_reports

    scenarios = build_corpus()
    _validate_selection(config, scenarios)
    preflight = ocr_preflight(config.language, config.tesseract_cmd)
    output = Path(config.output_dir).resolve()
    identity = run_identity(scenarios, preflight)
    manifest = prepare_run(output, identity, config.resume)
    if not (output / "ground_truth.json").exists():
        write_json(output / "ground_truth.json", corpus_payload(scenarios), exclusive=True)
    print(preflight["status"], flush=True)
    for reason in preflight["errors"]:
        print(reason, flush=True)
    for scenario in scenarios:
        if config.scenarios and scenario.scenario_id not in config.scenarios:
            continue
        _run_scenario(scenario, config, (output, manifest, preflight))
    records = load_records(output, manifest)
    return write_reports(output, identity, (scenarios, records))


def _validate_selection(config, scenarios):
    selections = (
        (config.scenarios, {item.scenario_id for item in scenarios}, "scenario"),
        (config.variants, set(VARIANTS), "variant"),
        (config.document_types, {"claim_form", "accident_report", "garage_quote"}, "document type"),
    )
    for selected, allowed, label in selections:
        if set(selected) - allowed:
            raise ValueError(f"Unknown {label}: {sorted(set(selected) - allowed)}")


def _run_scenario(scenario, config, state):
    output, manifest, preflight = state
    for reference in scenario.documents:
        if config.document_types and reference.document_type not in config.document_types:
            continue
        for variant in config.variants or VARIANTS:
            key = f"{reference.document_id}__{variant}"
            if key in manifest["cases"]:
                continue
            result_path = output / "results" / f"{key}.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result["case_key"] != key:
                    raise ValueError(f"Orphan case identity differs: {result_path}")
            else:
                result = evaluate_document(reference, variant, output, preflight)
                result.update(scenario_id=scenario.scenario_id, timestamp=timestamp())
                write_json(result_path, result, exclusive=True)
            manifest["cases"][key] = {name: result[name] for name in ("scenario_id", "variant", "timestamp", "status")}
            write_json(output / "manifest.json", manifest)
            print(f"{key}: {result['status']} ({result['extraction_method']})", flush=True)
