"""Document fixture text rendering and real native PDF ingestion."""

from datetime import date as document_date
from decimal import Decimal

from claims.assets import resolve_document_assets
from claims.document_extraction import extract_accident_report, extract_claim_form, extract_garage_quote
from claims.document_ingestion import extract_pdf_text
from claims.enums import DocumentType

FILENAMES = {
    "CLAIM_FORM": "claim_form.pdf",
    "ACCIDENT_REPORT": "accident_report.pdf",
    "REPAIR_QUOTE": "garage_quote.pdf",
    "POLICE_REPORT": "police_report.pdf",
}
EXTRACTORS = {
    DocumentType.CLAIM_FORM: extract_claim_form,
    DocumentType.ACCIDENT_REPORT: extract_accident_report,
    DocumentType.REPAIR_QUOTE: extract_garage_quote,
}


def document_payloads(inputs):
    """Return the declared document filenames and their claim manifest entries."""
    missing = set(inputs.get("missing_documents", []))
    entries = [
        {"type": kind, "filename": name if kind not in missing else None, "available": kind not in missing}
        for kind, name in FILENAMES.items()
    ]
    return FILENAMES, missing, entries


def _header(payload):
    damage = "\n".join(item.lower().replace("_", " ") for item in payload["declared_damage"])
    vehicle = f"{payload['vehicle']['make']} {payload['vehicle']['model']}"
    injuries = "Yes" if payload["incident"]["injuries_declared"] else "No"
    return damage, vehicle, injuries


def document_texts(payload, inputs, quoted, amount):
    """Render the synthetic document bodies, then apply controlled overrides."""
    date = document_date.fromisoformat(payload["incident"]["date"]).strftime("%d/%m/%Y")
    damage, vehicle, injuries = _header(payload)
    claim_id = payload["claim_id"]
    texts = {
        "CLAIM_FORM": f"DEMO SYNTHETIC CLAIM FORM\nClaim ID: {claim_id}\nIncident date: {date}\n"
        f"Location: {payload['incident']['location']}\nVehicle: {vehicle}\n"
        f"Vehicle year: {payload['vehicle']['year']}\nCollision type: Front collision\n"
        f"Declared damage:\n{damage}\nInjuries declared: {injuries}",
        "ACCIDENT_REPORT": f"DEMO SYNTHETIC ACCIDENT REPORT\nClaim ID: {claim_id}\nDate: {date}\n"
        f"Location: {payload['incident']['location']}\nVehicle:\n{vehicle}\n"
        f"Collision type: Front collision\nReported damage:\n{damage}\nPassengers: 1\nInjuries: {injuries}",
        "POLICE_REPORT": f"DEMO SYNTHETIC SUPPORTING RECORD\nClaim ID: {claim_id}\n"
        "No inference of misconduct is made.",
    }
    part_amount = (Decimal(amount) / Decimal(max(1, len(quoted) + 1))).quantize(Decimal("0.01"))
    labor = Decimal(amount) - part_amount * len(quoted)
    parts = "\n".join(f"{item.lower().replace('_', ' ')}: {part_amount} EUR" for item in quoted)
    texts["REPAIR_QUOTE"] = (
        f"DEMO SYNTHETIC GARAGE QUOTE\nGarage: Demo Garage\nClaim ID: {claim_id}\n{parts}\n"
        f"Labor: {labor} EUR\nTotal: {amount} EUR"
    )
    for kind, overrides in inputs.get("document_overrides", {}).items():
        if isinstance(overrides, str):
            texts[kind] = overrides
        else:
            for old, new in overrides.items():
                texts[kind] = texts[kind].replace(old, str(new))
    return texts


def extract_case_documents(case):
    """Real native PDF ingestion plus existing structured extractors."""
    ingested, extracted = {}, {}
    for asset in resolve_document_assets(case):
        text = extract_pdf_text(asset)
        ingested[asset.filename] = text
        if asset.type in EXTRACTORS:
            extracted[asset.type] = EXTRACTORS[asset.type](text)
    return extracted, ingested
