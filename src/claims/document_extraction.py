import re
from decimal import Decimal
from datetime import datetime

from claims.document_models import (
    AccidentReportExtraction,
    ClaimFormExtraction,
    DocumentSource,
    GarageQuoteExtraction,
    MoneyAmount,
    QuoteLineItem,
)

from claims.enums import (
    CollisionType,
    DamageType,
    DocumentType,
)

from claims.document_ingestion import PdfTextExtraction
from claims.document_models import (
    DocumentSource,
    GarageQuoteExtraction,
    MoneyAmount,
    QuoteLineItem,
)
from claims.enums import DocumentType

class DocumentExtractionError(Exception):
    pass

MONEY_LINE_PATTERN = re.compile(
    r"^(?P<label>[^:]+):\s*"
    r"(?P<amount>\d+(?:[.,]\d{1,2})?)\s+"
    r"(?P<currency>[A-Z]{3})$"
)

DAMAGE_LABEL_MAP = {
    "front bumper": DamageType.FRONT_BUMPER,
    "left headlight": DamageType.LEFT_HEADLIGHT,
    "hood": DamageType.HOOD,
}

def _find_prefixed_value(
    lines: list[str],
    prefix: str,
) -> str:
    for line in lines:
        if line.startswith(prefix):
            value = line[len(prefix):].strip()

            if value:
                return value

    raise DocumentExtractionError(
        f"Missing field: {prefix}"
    )

def extract_garage_quote(
    extraction: PdfTextExtraction,
) -> GarageQuoteExtraction:
    if (
        extraction.asset.type
        != DocumentType.REPAIR_QUOTE
    ):
        raise DocumentExtractionError(
            "Expected a repair quote document"
        )
    
    if not extraction.has_text:
        raise DocumentExtractionError(
            "Repair quote contains no extractable text"
        )
    
    lines = [
        line.strip()
        for line in extraction.text.splitlines()
        if line.strip()
    ]

    garage = _find_prefixed_value(
        lines,
        "Garage:",
    )

    claim_id = _find_prefixed_value(
        lines,
        "Claim ID:",
    )
    parts: list[QuoteLineItem] = []

    labor: MoneyAmount | None = None
    total: MoneyAmount | None = None

    for line in lines:
        match = MONEY_LINE_PATTERN.match(line)

        if match is None:
            continue

        label = match.group("label").strip()

        amount = Decimal(
            match.group("amount").replace(",", ".")
        )

        currency = match.group("currency")

        money = MoneyAmount(
            amount=amount,
            currency=currency,
        )
        if label.lower() == "labor":
            labor = money

        elif label.lower() == "total":
            total = money

        else:
            parts.append(
                QuoteLineItem(
                    description=label,
                    price=money,
                )
            )

    if labor is None:
        raise DocumentExtractionError(
            "Missing labor amount"
        )

    if total is None:
        raise DocumentExtractionError(
            "Missing total amount"
        )
    currencies = {
        item.price.currency
        for item in parts
    }

    currencies.add(labor.currency)
    currencies.add(total.currency)

    if len(currencies) != 1:
        raise DocumentExtractionError(
            "Quote contains inconsistent currencies"
        )
    calculated_total = sum(
        (
            item.price.amount
            for item in parts
        ),
        Decimal("0"),
    )

    calculated_total += labor.amount
    if calculated_total != total.amount:
        raise DocumentExtractionError(
            "Quote total does not match line items"
        )
    return GarageQuoteExtraction(
        garage=garage,
        claim_id=claim_id,
        parts=tuple(parts),
        labor=labor,
        total=total,
        source=DocumentSource(
            document_type=extraction.asset.type,
            filename=extraction.asset.filename,
            page=1,
        ),
    )

def normalize_damage(
    value: str,
) -> DamageType:
    normalized = value.strip().lower()

    try:
        return DAMAGE_LABEL_MAP[normalized]
    except KeyError as error:
        raise DocumentExtractionError(
            f"Unknown damage type: {value}"
        ) from error

COLLISION_TYPE_MAP = {
    "front collision": CollisionType.FRONT_COLLISION,
}

def normalize_collision_type(
    value: str,
) -> CollisionType:
    normalized = value.strip().lower()

    try:
        return COLLISION_TYPE_MAP[normalized]
    except KeyError as error:
        raise DocumentExtractionError(
            f"Unknown collision type: {value}"
        ) from error

def parse_yes_no(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized == "yes":
        return True

    if normalized == "no":
        return False

    raise DocumentExtractionError(
        f"Expected Yes/No value, got: {value}"
    )

def parse_document_date(value: str):
    try:
        return datetime.strptime(
            value,
            "%d/%m/%Y",
        ).date()

    except ValueError as error:
        raise DocumentExtractionError(
            f"Invalid document date: {value}"
        ) from error

def _find_section(
    lines: list[str],
    start: str,
    stop_prefixes: tuple[str, ...],
) -> list[str]:
    try:
        start_index = lines.index(start)
    except ValueError as error:
        raise DocumentExtractionError(
            f"Missing section: {start}"
        ) from error

    values = []

    for line in lines[start_index + 1:]:
        if line.startswith(stop_prefixes):
            break

        values.append(line)

    return values

def extract_claim_form(
    extraction: PdfTextExtraction,
) -> ClaimFormExtraction:
    if extraction.asset.type != DocumentType.CLAIM_FORM:
        raise DocumentExtractionError(
            "Expected a claim form document"
        )

    if not extraction.has_text:
        raise DocumentExtractionError(
            "Claim form contains no extractable text"
        )

    lines = [
        line.strip()
        for line in extraction.text.splitlines()
        if line.strip()
    ]
    claim_id = _find_prefixed_value(
        lines,
        "Claim ID:",
    )

    incident_date = parse_document_date(
        _find_prefixed_value(
            lines,
            "Incident date:",
        )
    )

    location = _find_prefixed_value(
        lines,
        "Location:",
    )
    vehicle = _find_prefixed_value(
        lines,
        "Vehicle:",
    )

    vehicle_parts = vehicle.split(
        maxsplit=1
    )

    if len(vehicle_parts) != 2:
        raise DocumentExtractionError(
            f"Unable to parse vehicle: {vehicle}"
        )

    vehicle_make, vehicle_model = vehicle_parts

    vehicle_year = int(
        _find_prefixed_value(
            lines,
            "Vehicle year:",
        )
    )

    collision_type = normalize_collision_type(
        _find_prefixed_value(
            lines,
            "Collision type:",
        )
    )

    injuries_declared = parse_yes_no(
        _find_prefixed_value(
            lines,
            "Injuries declared:",
        )
    )
    damage_lines = _find_section(
        lines,
        "Declared damage:",
        (
            "Injuries declared:",
        ),
    )

    declared_damage = tuple(
        normalize_damage(value)
        for value in damage_lines
    )
    return ClaimFormExtraction(
        claim_id=claim_id,
        incident_date=incident_date,
        location=location,
        vehicle_make=vehicle_make,
        vehicle_model=vehicle_model,
        vehicle_year=vehicle_year,
        collision_type=collision_type,
        declared_damage=declared_damage,
        injuries_declared=injuries_declared,
        source=DocumentSource(
            document_type=extraction.asset.type,
            filename=extraction.asset.filename,
            page=1,
        ),
    )

def extract_accident_report(
    extraction: PdfTextExtraction,
) -> AccidentReportExtraction:
    if (
        extraction.asset.type
        != DocumentType.ACCIDENT_REPORT
    ):
        raise DocumentExtractionError(
            "Expected an accident report document"
        )

    if not extraction.has_text:
        raise DocumentExtractionError(
            "Accident report contains no extractable text"
        )

    lines = [
        line.strip()
        for line in extraction.text.splitlines()
        if line.strip()
    ]

    claim_id = _find_prefixed_value(
        lines,
        "Claim ID:",
    )

    incident_date = parse_document_date(
        _find_prefixed_value(
            lines,
            "Date:",
        )
    )

    location = _find_prefixed_value(
        lines,
        "Location:",
    )

    collision_type = normalize_collision_type(
        _find_prefixed_value(
            lines,
            "Collision type:",
        )
    )

    try:
        vehicle_index = lines.index(
            "Vehicle:"
        )
        vehicle = lines[vehicle_index + 1]
    except (ValueError, IndexError) as error:
        raise DocumentExtractionError(
            "Missing vehicle information"
        ) from error

    vehicle_parts = vehicle.split(
        maxsplit=1
    )

    if len(vehicle_parts) != 2:
        raise DocumentExtractionError(
            f"Unable to parse vehicle: {vehicle}"
        )

    vehicle_make, vehicle_model = vehicle_parts

    damage_lines = _find_section(
        lines,
        "Reported damage:",
        (
            "Passengers:",
            "Injuries:",
        ),
    )

    reported_damage = tuple(
        normalize_damage(value)
        for value in damage_lines
    )

    passengers = int(
        _find_prefixed_value(
            lines,
            "Passengers:",
        )
    )

    injuries_declared = parse_yes_no(
        _find_prefixed_value(
            lines,
            "Injuries:",
        )
    )

    return AccidentReportExtraction(
        claim_id=claim_id,
        incident_date=incident_date,
        location=location,
        vehicle_make=vehicle_make,
        vehicle_model=vehicle_model,
        collision_type=collision_type,
        reported_damage=reported_damage,
        passengers=passengers,
        injuries_declared=injuries_declared,
        source=DocumentSource(
            document_type=extraction.asset.type,
            filename=extraction.asset.filename,
            page=1,
        ),
    )