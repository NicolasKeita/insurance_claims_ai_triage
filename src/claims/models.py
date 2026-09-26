from datetime import date

from pydantic import BaseModel, Field, model_validator
from claims.enums import (
    ClaimStatus,
    ClaimType,
    CollisionType,
    DamageType,
    DocumentType,
)

class Vehicle(BaseModel):
    make: str
    model: str
    year: int = Field(ge=1886)

class Customer(BaseModel):
    customer_id: str

class Policy(BaseModel):
    policy_id: str
    product: str

class Incident(BaseModel):
    date: date
    location: str
    collision_type: CollisionType
    injuries_declared: bool

    @model_validator(mode="after")
    def validate_incident_date(self):
        if self.date > date.today():
            raise ValueError(
                "Incident date cannot be in the future"
            )

        return self

class ClaimDocument(BaseModel):
    type: DocumentType
    filename: str | None
    available: bool

    @model_validator(mode="after")
    def validate_available_document(self):
        if self.available and self.filename is None:
            raise ValueError(
                "An available document must have a filename"
            )

        return self

class RepairEstimate(BaseModel):
    amount: float = Field(ge=0)
    currency: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )

class Claim(BaseModel):
    claim_id: str
    claim_type: ClaimType
    status: ClaimStatus

    incident: Incident
    vehicle: Vehicle
    customer: Customer
    policy: Policy

    declared_damage: list[DamageType]
    documents: list[ClaimDocument]
    images: list[str]

    repair_estimate: RepairEstimate