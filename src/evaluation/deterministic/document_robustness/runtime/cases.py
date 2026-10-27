"""Exercise the existing native/OCR ingestion and production field extractors."""

from pathlib import Path
from time import perf_counter

from claims.assets import DocumentAsset
from claims.document_extraction import extract_accident_report, extract_claim_form, extract_garage_quote
from claims.enums import DocumentType
from evaluation.deterministic.document_robustness.comparison import compare_fields
from evaluation.deterministic.document_robustness.metrics import text_metrics
from evaluation.deterministic.document_robustness.runtime.sources import materialize_or_reuse

EXTRACTORS = {
    "claim_form": extract_claim_form,
    "accident_report": extract_accident_report,
    "garage_quote": extract_garage_quote,
}
DOCUMENT_TYPES = {
    "claim_form": DocumentType.CLAIM_FORM,
    "accident_report": DocumentType.ACCIDENT_REPORT,
    "garage_quote": DocumentType.REPAIR_QUOTE,
}


def evaluate_document(reference, variant, output, preflight):
    key = f"{reference.document_id}__{variant}"
    source = output / "sources" / f"{key}.pdf"
    result = {
        "case_key": key,
        "document_id": reference.document_id,
        "document_type": reference.document_type,
        "variant": variant,
        "status": "executed",
        "source_variant": str(source),
        "errors": [],
        "warnings": [],
        "extraction_method": None,
        "extracted_text": "",
        "structured": None,
        "text_metrics": None,
        "native_duration_seconds": None,
        "ocr_duration_seconds": None,
        "structured_duration_seconds": None,
        "extraction_attempts": 0,
        "structured_attempts": 0,
    }
    started = perf_counter()
    actual = None
    try:
        unavailable = not preflight.get("native_available", True) or (
            variant != "native_clean" and not preflight["available"]
        )
        if unavailable:
            result.update(status="not_executed", errors=preflight["errors"])
        else:
            result["degradation_config"] = materialize_or_reuse(reference, variant, source)
            actual = _extract(reference, result, preflight)
    except Exception as error:
        result["status"] = "failed"
        result["errors"].append(f"{type(error).__name__}: {error}")
        result["initial_error"] = result["errors"][0]
    result["duration_seconds"] = perf_counter() - started
    result["structured_metrics"] = compare_fields(reference.expected, actual)
    return result


def _extract(reference, result, preflight):
    from claims.document_ingestion import extract_pdf_text

    asset = DocumentAsset(
        type=DOCUMENT_TYPES[reference.document_type],
        filename=reference.document_id + ".pdf",
        path=Path(result["source_variant"]),
    )
    result["extraction_attempts"] = 1
    if result["variant"] == "native_clean" and not preflight["available"]:
        extraction = extract_pdf_text(asset)
    else:
        from claims.document_ingestion import extract_pdf_text_auto
        from claims.ocr import TesseractOcrEngine

        extraction = extract_pdf_text_auto(asset, TesseractOcrEngine(language=preflight["language"]), dpi=300)
    result.update(
        extraction_method=extraction.method.value,
        extracted_text=extraction.text,
        native_duration_seconds=extraction.native_duration_seconds,
        ocr_duration_seconds=extraction.ocr_duration_seconds,
        text_metrics=text_metrics(reference.text, extraction.text),
    )
    expected_method = {"native_clean": "NATIVE", "mixed_native_and_scan": "HYBRID"}.get(result["variant"], "OCR")
    if result["extraction_method"] != expected_method:
        raise ValueError(f"Expected {expected_method}, got {result['extraction_method']}")
    started = perf_counter()
    result["structured_attempts"] = 1
    try:
        actual = EXTRACTORS[reference.document_type](extraction)
        result["structured"] = actual.model_dump(mode="json")
        return actual
    finally:
        result["structured_duration_seconds"] = perf_counter() - started
