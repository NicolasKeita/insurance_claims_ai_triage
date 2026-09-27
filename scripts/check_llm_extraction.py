import os
from pathlib import Path

from claims.assets import resolve_document_assets
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.llm import OllamaStructuredLlm
from claims.llm_document_extraction import (
    extract_garage_quote_with_llm,
)
from claims.loader import load_claim_case


model = os.environ.get(
    "CLAIMS_LLM_MODEL"
)

if model is None:
    raise RuntimeError(
        "CLAIMS_LLM_MODEL environment variable is required"
    )


case = load_claim_case(
    Path("data/CLAIM-2026-00001")
)

assets = resolve_document_assets(case)

asset = next(
    asset
    for asset in assets
    if asset.type == DocumentType.REPAIR_QUOTE
)

text_extraction = extract_pdf_text(asset)

llm = OllamaStructuredLlm(
    model=model
)

quote = extract_garage_quote_with_llm(
    text_extraction,
    llm,
)

print(
    quote.model_dump_json(
        indent=2
    )
)