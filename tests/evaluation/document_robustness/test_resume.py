"""Interrupted source recovery and baseline comparability use immutable evidence."""

import json

import pytest

from evaluation.deterministic.document_robustness.corpus import build_corpus
from evaluation.deterministic.document_robustness.reporting import compare_saved_baseline
from evaluation.deterministic.document_robustness.runtime import storage
from evaluation.deterministic.document_robustness.runtime.sources import materialize_or_reuse
from evaluation.deterministic.document_robustness.transformations import VARIANTS


@pytest.mark.parametrize("variant", VARIANTS)
def test_real_pdf_generation_reuses_identical_interrupted_sources(tmp_path, variant):
    reference = build_corpus()[0].documents[0]
    source = tmp_path / f"{variant}.pdf"
    config = materialize_or_reuse(reference, variant, source)
    original = source.read_bytes()
    assert original.startswith(b"%PDF-")
    assert materialize_or_reuse(reference, variant, source) == config
    assert source.read_bytes() == original
    assert list(tmp_path.iterdir()) == [source]


def test_differing_existing_source_is_preserved_and_temporary_render_removed(tmp_path):
    reference = build_corpus()[0].documents[0]
    source = tmp_path / "native.pdf"
    materialize_or_reuse(reference, "native_clean", source)
    altered = source.read_bytes() + b"\n"
    source.write_bytes(altered)
    with pytest.raises(ValueError, match="differs from deterministic reference"):
        materialize_or_reuse(reference, "native_clean", source)
    assert source.read_bytes() == altered
    assert list(tmp_path.iterdir()) == [source]


def _summary(variant, count):
    return {
        "benchmark_version": "document_ocr_v1",
        "identity": {"test": "same-corpus-and-config"},
        "corpus": {"total_cases": 2},
        "recorded_cases": count,
        "metrics": {"evaluated_count": count},
        "documents": [{"document_id": f"doc-{index}", "variant": variant} for index in range(count)],
    }


@pytest.mark.parametrize("count,message", [(1, "two complete runs"), (2, "identical document/variant populations")])
def test_baselines_reject_same_count_with_different_document_populations(tmp_path, count, message):
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_summary("native_clean", count)), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        compare_saved_baseline(baseline, _summary("scan_clean", count))


def test_resume_rejects_changed_implementation_without_modifying_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "implementation_fingerprint", lambda: "first-code-fingerprint")
    identity = {"benchmark_version": "document_ocr_v1"}
    initial = storage.prepare_run(tmp_path, identity, resume=False)
    manifest = tmp_path / "manifest.json"
    original = manifest.read_bytes()
    assert storage.prepare_run(tmp_path, identity, resume=True) == initial
    monkeypatch.setattr(storage, "implementation_fingerprint", lambda: "changed-code-fingerprint")
    with pytest.raises(ValueError, match="Resume implementation differs"):
        storage.prepare_run(tmp_path, identity, resume=True)
    assert manifest.read_bytes() == original
