from decimal import Decimal

from pydantic import BaseModel, Field

from claims.enums import DocumentType

from datetime import date

from claims.enums import (
    CollisionType,
    DamageType,
)


class MoneyAmount(BaseModel):
    amount: Decimal = Field(ge=0)
    currency: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )

class QuoteLineItem(BaseModel):
    description: str
    price: MoneyAmount

class DocumentSource(BaseModel):
    document_type: DocumentType
    filename: str
    page: int = Field(ge=1)

class GarageQuoteExtraction(BaseModel):
    garage: str
    claim_id: str

    parts: tuple[QuoteLineItem, ...]
    labor: MoneyAmount
    total: MoneyAmount

    source: DocumentSource


class ClaimFormExtraction(BaseModel):
    claim_id: str

    incident_date: date
    location: str

    vehicle_make: str
    vehicle_model: str
    vehicle_year: int

    collision_type: CollisionType

    declared_damage: tuple[DamageType, ...]
    injuries_declared: bool

    source: DocumentSource

class AccidentReportExtraction(BaseModel):
    claim_id: str

    incident_date: date
    location: str

    vehicle_make: str
    vehicle_model: str

    collision_type: CollisionType

    reported_damage: tuple[DamageType, ...]

    passengers: int = Field(ge=0)
    injuries_declared: bool

    source: DocumentSource