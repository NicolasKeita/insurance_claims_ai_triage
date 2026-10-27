"""Small real native runs and non-destructive resume require no OCR executable."""

import json
from dataclasses import replace

import pytest

from evaluation.deterministic.document_robustness.runtime import session
from evaluation.document_robustness import BenchmarkConfig, run_benchmark


@pytest.fixture
def offline(monkeypatch):
    result = {
        "available": False,
        "status": "OCR benchmark: NOT EXECUTED",
        "language": "eng",
        "errors": ["Tesseract: FileNotFoundError: synthetic executable unavailable"],
        "tesseract_version": None,
        "document_generation_available": True,
        "native_available": True,
        "missing_imports": [],
    }
    monkeypatch.setattr(session, "ocr_preflight", lambda *_: result)
    return result


def native_config(path):
    return BenchmarkConfig(
        output_dir=path, scenarios=("demo_01",), variants=("native_clean",), document_types=("claim_form",)
    )


def test_native_tiny_run_and_resume_preserve_results(tmp_path, monkeypatch, offline):
    config = native_config(tmp_path)
    summary = run_benchmark(config)
    assert summary["recorded_cases"] == 1
    assert summary["metrics"]["document_parse_success_rate"] == 1
    assert summary["metrics"]["critical_field_accuracy"] == 1
    result = next((tmp_path / "results").glob("*.json"))
    before = result.read_bytes()
    original = session.evaluate_document

    def only_missing(reference, variant, output, preflight):
        assert reference.document_type != "claim_form"
        return original(reference, variant, output, preflight)

    monkeypatch.setattr(session, "evaluate_document", only_missing)
    resumed = run_benchmark(replace(config, resume=True, document_types=()))
    assert resumed["recorded_cases"] == 3 and result.read_bytes() == before
    assert resumed["consistency"][0]["preserved"] is True
    assert json.loads(before)["extraction_method"] == "NATIVE"
    for name in ("summary.json", "per_document.csv", "per_variant.csv", "critical_fields.csv", "ground_truth.json"):
        assert (tmp_path / name).is_file()


@pytest.mark.parametrize("change", ("environment", "corpus"))
def test_resume_rejects_environment_or_corpus_changes(tmp_path, monkeypatch, offline, change):
    config = native_config(tmp_path)
    run_benchmark(config)
    before = (tmp_path / "manifest.json").read_bytes()
    if change == "environment":
        offline["tesseract_version"] = "changed version"
    else:
        scenarios = list(session.build_corpus())
        docs = scenarios[0].documents
        scenarios[0] = replace(scenarios[0], documents=(replace(docs[0], text=docs[0].text + " changed"), *docs[1:]))
        monkeypatch.setattr(session, "build_corpus", lambda: tuple(scenarios))
    with pytest.raises(ValueError, match="Resume identity differs"):
        run_benchmark(replace(config, resume=True))
    assert (tmp_path / "manifest.json").read_bytes() == before


def test_existing_outputs_are_not_overwritten(tmp_path, offline):
    marker = tmp_path / "existing.txt"
    marker.write_text("preserve me", encoding="utf-8")
    with pytest.raises(ValueError, match="Nonempty output directory has no manifest"):
        run_benchmark(native_config(tmp_path))
    assert marker.read_text(encoding="utf-8") == "preserve me"
    output = tmp_path / "run"
    run_benchmark(native_config(output))
    with pytest.raises(ValueError, match="Output already exists"):
        run_benchmark(native_config(output))


def test_unavailable_ocr_is_saved_with_exact_reason(tmp_path, offline, capsys):
    config = replace(native_config(tmp_path), variants=("scan_clean",))
    summary = run_benchmark(config)
    record = json.loads(next((tmp_path / "results").glob("*.json")).read_text(encoding="utf-8"))
    assert record["status"] == "not_executed" and record["errors"] == offline["errors"]
    assert record["extraction_method"] is None and record["extraction_attempts"] == 0
    assert summary["metrics"]["evaluated_count"] == 0
    assert summary["metrics"]["not_executed_count"] == 1
    assert summary["metrics"]["critical_field_accuracy"] is None
    captured = capsys.readouterr().out
    assert "OCR benchmark: NOT EXECUTED" in captured and offline["errors"][0] in captured
