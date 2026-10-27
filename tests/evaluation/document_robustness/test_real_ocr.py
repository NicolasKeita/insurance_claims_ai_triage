"""Small opt-in checks of actual Tesseract through the production document path."""

import os

import pytest

from evaluation.deterministic.document_robustness.corpus import build_corpus
from evaluation.deterministic.document_robustness.runtime.cases import evaluate_document
from evaluation.deterministic.document_robustness.runtime.preflight import ocr_preflight

pytestmark = [
    pytest.mark.ocr,
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_OCR_TESTS") != "1", reason="Set RUN_OCR_TESTS=1 to run real OCR tests"),
]


@pytest.fixture(scope="module")
def actual_ocr_environment():
    preflight = ocr_preflight(language="eng")
    if not preflight["available"]:
        pytest.skip(preflight["status"] + ": " + "; ".join(preflight["errors"]))
    return preflight


@pytest.mark.parametrize("document_index", [0, 1, 2], ids=["claim_form", "accident_report", "garage_quote"])
@pytest.mark.parametrize("variant, method", [("scan_clean", "OCR"), ("mixed_native_and_scan", "HYBRID")])
def test_real_ocr_clean_documents(tmp_path, actual_ocr_environment, document_index, variant, method):
    reference = build_corpus()[0].documents[document_index]
    result = evaluate_document(reference, variant, tmp_path, actual_ocr_environment)
    assert result["extraction_method"] == method, result["errors"]
    assert result["extracted_text"].strip()
    assert result["ocr_duration_seconds"] > 0
    assert result["structured_metrics"]["parse_success"], result["errors"]
    assert result["status"] == "executed", result["errors"]
