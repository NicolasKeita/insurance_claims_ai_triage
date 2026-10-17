"""Document-type dispatch for LLM-assisted extraction."""

from claims.document_extraction import DocumentExtractionError
from claims.enums import DocumentType
from claims.llm import StructuredLlm
from claims.llm_document_extraction.accident import extract_accident_report_with_llm
from claims.llm_document_extraction.claim_form import extract_claim_form_with_llm
from claims.llm_document_extraction.common import LlmDocumentExtraction
from claims.llm_document_extraction.quote import extract_garage_quote_with_llm


def extract_document_with_llm(extraction, llm: StructuredLlm) -> LlmDocumentExtraction:
    match extraction.asset.type:
        case DocumentType.CLAIM_FORM:
            return extract_claim_form_with_llm(extraction, llm)
        case DocumentType.ACCIDENT_REPORT:
            return extract_accident_report_with_llm(extraction, llm)
        case DocumentType.REPAIR_QUOTE:
            return extract_garage_quote_with_llm(extraction, llm)
        case _:
            raise DocumentExtractionError(f"Unsupported document type: {extraction.asset.type}")
