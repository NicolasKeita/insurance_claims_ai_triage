import os
from pathlib import Path

from claims.assets import resolve_document_assets
from claims.document_ingestion import extract_pdf_text
from claims.llm import OllamaStructuredLlm
from claims.llm_document_extraction import (
    extract_document_with_llm,
)
from claims.loader import load_claim_case

model = os.environ.get(
    "CLAIMS_LLM_MODEL"
)

if model is None:
    raise RuntimeError(
        "CLAIMS_LLM_MODEL environment variable is required"
    )

llm = OllamaStructuredLlm(
    model=model
)

case = load_claim_case(
    Path("data/CLAIM-2026-00001")
)

assets = resolve_document_assets(case)

for asset in assets:
    print()
    print("=" * 60)
    print(asset.type)
    print("=" * 60)

    try:
        text_extraction = extract_pdf_text(
            asset
        )

        result = extract_document_with_llm(
            text_extraction,
            llm,
        )

        print(
            result.model_dump_json(
                indent=2
            )
        )

    except Exception as error:
        print("EXTRACTION FAILED")
        print()

        print("SOURCE TEXT:")
        print("-" * 60)
        print(text_extraction.text)

        print()
        print("ERROR:")
        print("-" * 60)
        print(error)