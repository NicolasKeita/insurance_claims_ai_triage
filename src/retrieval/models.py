"""ORM-free, deliberately allowlisted retrieval data."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel


@dataclass(frozen=True)
class ClaimRetrievalFacts:
    claim_type: str
    collision_type: str
    location: str
    vehicle_make: str
    vehicle_model: str
    vehicle_year: int
    declared_damage: tuple[str, ...]
    injuries_declared: bool
    repair_amount: Decimal
    repair_currency: str


@dataclass(frozen=True)
class StoredRetrievalClaim:
    point_id: UUID
    claim_id: str
    incident_date: date
    facts: ClaimRetrievalFacts


@dataclass(frozen=True)
class ClaimRetrievalDocument:
    point_id: UUID
    claim_id: str
    incident_date: date
    claim_type: str
    canonical_text: str
    representation_version: str


@dataclass(frozen=True)
class RetrievalHit:
    point_id: UUID
    similarity_score: float


class SimilarClaim(BaseModel):
    claim_id: str
    similarity_score: float
    incident_date: date
    claim_type: str
    collision_type: str
    repair_amount: Decimal
    repair_currency: str


class SimilarClaimsResponse(BaseModel):
    claim_id: str
    representation_version: str
    results: list[SimilarClaim]


class RetrievalUnavailableError(RuntimeError):
    """Safe public failure; the underlying exception is chained privately."""


class IndexCompatibilityError(RetrievalUnavailableError):
    pass
