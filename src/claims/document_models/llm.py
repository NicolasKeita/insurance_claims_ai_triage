"""Raw structured-output schemas for local LLM document extraction."""

from pydantic import BaseModel, Field

from claims.enums import CollisionType, DamageType


class LlmMoneyAmount(BaseModel):
    amount: float = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")


class LlmQuoteLineItem(BaseModel):
    description: str
    price: LlmMoneyAmount


class GarageQuoteLlmOutput(BaseModel):
    garage: str | None
    claim_id: str | None
    parts: tuple[LlmQuoteLineItem, ...] = ()
    labor: LlmMoneyAmount | None
    total: LlmMoneyAmount | None


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
    passengers: int | None = Field(default=None, ge=0)
    injuries_declared: bool | None
