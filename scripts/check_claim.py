from pathlib import Path

from claims.loader import load_claim_case
from claims.assets import (
    resolve_document_assets,
    resolve_image_assets,
)
from claims.document_ingestion import (
    extract_pdf_text,
)

from claims.document_extraction import (
    extract_garage_quote,
)
from claims.enums import DocumentType



claim_dir = Path("data/CLAIM-2026-00001")

case = load_claim_case(claim_dir)
claim = case.claim

print(f"Claim: {claim.claim_id}")
print(f"Type: {claim.claim_type}")
print(f"Vehicle: {claim.vehicle.make} {claim.vehicle.model}")
print(f"Incident date: {claim.incident.date}")
print(
    f"Repair estimate: "
    f"{claim.repair_estimate.amount} "
    f"{claim.repair_estimate.currency}"
)

print("\nDeclared damage:")

for damage in claim.declared_damage:
    print(f"- {damage}")

print("\nMissing documents:")

for document in claim.documents:
    if not document.available:
        print(f"- {document.type}")

documents = resolve_document_assets(case)
images = resolve_image_assets(case)

print("\nDocuments:")

for document in documents:
    print(
        f"- {document.type}: "
        f"{document.path} "
        f"({document.media_type})"
    )


print("\nImages:")

for image in images:
    print(
        f"- {image.path} "
        f"({image.media_type})"
    )

print("\nExtracted document text:")

for asset in documents:
    extraction = extract_pdf_text(asset)

    print()
    print("=" * 60)
    print(asset.filename)
    print("=" * 60)
    print(extraction.text)


repair_quote_asset = next(
    asset
    for asset in documents
    if asset.type == DocumentType.REPAIR_QUOTE
)

text_extraction = extract_pdf_text(
    repair_quote_asset
)

quote = extract_garage_quote(
    text_extraction
)

print("\nStructured garage quote:")
print(quote.model_dump_json(indent=2))