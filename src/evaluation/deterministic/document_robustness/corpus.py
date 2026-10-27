"""Five demo dossiers reuse the controlled source schema and E2E document renderer."""

import json
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

from evaluation.deterministic.document_robustness.models import (
    KINDS,
    TYPES,
    ReferenceDocument,
    Scenario,
    build_expected,
)
from evaluation.deterministic.documents import document_texts

ROOT = Path(__file__).resolve().parents[4]
ROWS = (
    ("Renault", "Clio", "Demo North", "EUR", "1500.00", ("FRONT_BUMPER", "LEFT_HEADLIGHT")),
    ("Toyota", "Corolla", "Demo East", "USD", "1600.25", ("FRONT_BUMPER",)),
    ("Ford", "Focus", "Demo West", "EUR", "3160.00", ("FRONT_BUMPER", "LEFT_HEADLIGHT")),
    ("Peugeot", "208", "Demo South", "GBP", "1200.00", ("FRONT_BUMPER", "HOOD")),
    ("Volkswagen", "Golf", "Demo Central", "EUR", "4120.80", ("FRONT_BUMPER", "LEFT_HEADLIGHT", "HOOD")),
)


def _payload(source, index, row):
    payload = deepcopy(source)
    make, model, location, currency, amount, damage = row
    payload.update(claim_id=f"DEMO-OCR-{index:03d}", declared_damage=list(damage), images=[])
    payload["customer"] = {"customer_id": f"DEMO-CUSTOMER-{index:03d}"}
    payload["policy"]["policy_id"] = f"DEMO-POLICY-{index:03d}"
    payload["vehicle"] = {"make": make, "model": model, "year": 2018 + index}
    payload["incident"].update(date=f"2026-09-{15 + index:02d}", location=location, injuries_declared=index == 5)
    payload["repair_estimate"] = {"amount": amount, "currency": currency}
    return payload


def _scenario(source, index, row):
    payload = _payload(source, index, row)
    quoted = list(payload["declared_damage"]) + (["HOOD"] if index == 3 else [])
    garage = f"Demo Garage {index}"
    incident_date = date.fromisoformat(payload["incident"]["date"])
    report_date = incident_date + timedelta(days=8) if index == 4 else incident_date
    overrides = {
        "REPAIR_QUOTE": {"Demo Garage": garage, "EUR": row[3]},
        "ACCIDENT_REPORT": {incident_date.strftime("%d/%m/%Y"): report_date.strftime("%d/%m/%Y")},
    }
    expected = build_expected(payload, quoted, report_date, garage)
    texts = document_texts(payload, {"document_overrides": overrides}, quoted, Decimal(row[4]))
    scenario_id = f"demo_{index:02d}"
    documents = tuple(
        ReferenceDocument(f"{scenario_id}_{kind}", kind, texts[str(doc_type)], value)
        for kind, doc_type, value in zip(KINDS, TYPES, expected, strict=True)
    )
    facts = dict.fromkeys(
        (
            "claim_id_match",
            "incident_date_match",
            "location_match",
            "vehicle_match",
            "collision_type_match",
            "injuries_match",
            "declared_damage_match",
        ),
        True,
    )
    facts.update(
        incident_date_match=index != 4, additional_quote_damage=["HOOD"] if index == 3 else [], unmapped_quote_items=[]
    )
    return Scenario(scenario_id, documents, facts)


def build_corpus() -> tuple[Scenario, ...]:
    """Use only the demo source claim; no PDF or extracted text supplies ground truth."""
    source = json.loads((ROOT / "data/CLAIM-2026-00001/claim.json").read_text(encoding="utf-8"))
    return tuple(_scenario(source, index, row) for index, row in enumerate(ROWS, start=1))
