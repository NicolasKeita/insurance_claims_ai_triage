from decimal import Decimal

from pydantic import BaseModel, Field, model_validator

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

    @model_validator(mode="after")
    def validate_financials(self):
        currencies = {
            item.price.currency
            for item in self.parts
        }

        currencies.add(self.labor.currency)
        currencies.add(self.total.currency)

        if len(currencies) != 1:
            raise ValueError(
                "Quote contains inconsistent currencies"
            )

        calculated_total = sum(
            (
                item.price.amount
                for item in self.parts
            ),
            Decimal("0"),
        )

        calculated_total += self.labor.amount

        if calculated_total != self.total.amount:
            raise ValueError(
                "Quote total does not match line items"
            )

        return self


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

class GarageQuoteLlmOutput(BaseModel):
    garage: str | None
    claim_id: str | None

    parts: tuple[LlmQuoteLineItem, ...] = ()

    labor: LlmMoneyAmount | None
    total: LlmMoneyAmount | None

class LlmMoneyAmount(BaseModel):
    amount: float = Field(ge=0)

    currency: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )
class LlmQuoteLineItem(BaseModel):
    description: str
    price: LlmMoneyAmount

class ClaimFormLlmOutput(BaseModel):
    claim_id: str | None

    incident_date: str | None
    location: str | None

    vehicle_make: str | None
    vehicle_model: str | None
    vehicle_year: int | None

    collision_type: CollisionType | None

    declared_damage: tuple[DamageType, ...] = ()

    injuries_declared: bool | None

class AccidentReportLlmOutput(BaseModel):
    claim_id: str | None

    incident_date: str | None
    location: str | None

    vehicle_make: str | None
    vehicle_model: str | None

    collision_type: CollisionType | None

    reported_damage: tuple[DamageType, ...] = ()

    passengers: int | None = Field(
        default=None,
        ge=0,
    )

    injuries_declared: bool | None