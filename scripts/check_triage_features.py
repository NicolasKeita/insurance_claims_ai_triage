from pathlib import Path

from claims.assets import resolve_document_assets
from claims.consistency import compare_claim_documents
from claims.document_extraction import (
    extract_accident_report,
    extract_claim_form,
    extract_garage_quote,
)
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType
from claims.loader import load_claim_case
from ml.features import build_triage_features


def main() -> None:
    case = load_claim_case(Path("data/CLAIM-2026-00001"))
    assets = resolve_document_assets(case)

    def get_asset(document_type: DocumentType):
        return next(
            asset for asset in assets if asset.type == document_type
        )

    claim_form = extract_claim_form(
        extract_pdf_text(get_asset(DocumentType.CLAIM_FORM))
    )
    accident_report = extract_accident_report(
        extract_pdf_text(get_asset(DocumentType.ACCIDENT_REPORT))
    )
    garage_quote = extract_garage_quote(
        extract_pdf_text(get_asset(DocumentType.REPAIR_QUOTE))
    )

    consistency = compare_claim_documents(
        claim_form,
        accident_report,
        garage_quote,
    )
    features = build_triage_features(
        case=case,
        claim_form=claim_form,
        garage_quote=garage_quote,
        consistency=consistency,
    )
    print(features.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
