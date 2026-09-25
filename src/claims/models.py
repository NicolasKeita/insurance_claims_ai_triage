from datetime import date

from pydantic import BaseModel


class Vehicle(BaseModel):
    make: str
    model: str
    year: int

class Customer(BaseModel):
    customer_id: str

class Policy(BaseModel):
    policy_id: str
    product: str

class Incident(BaseModel):
    date: date
    location: str
    collision_type: str
    injuries_declared: bool

class ClaimDocument(BaseModel):
    type: str
    filename: str | None
    available: bool

class RepairEstimate(BaseModel):
    amount: float
    currency: str

class Claim(BaseModel):
    claim_id: str
    claim_type: str
    status: str

    incident: Incident
    vehicle: Vehicle
    customer: Customer
    policy: Policy

    declared_damage: list[str]
    documents: list[ClaimDocument]
    images: list[str]

    repair_estimate: RepairEstimate