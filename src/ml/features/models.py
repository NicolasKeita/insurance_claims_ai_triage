"""Triage feature schema and its engineering error."""

from pydantic import BaseModel, Field

from claims.enums import ClaimType, CollisionType


class TriageFeatures(BaseModel):
    claim_type: ClaimType
    collision_type: CollisionType
    vehicle_age: int = Field(ge=0)
    repair_amount: float = Field(ge=0)
    declared_damage_count: int = Field(ge=0)
    additional_damage_count: int = Field(ge=0)
    document_count: int = Field(ge=0)
    missing_document_count: int = Field(ge=0)
    document_completeness_ratio: float = Field(ge=0, le=1)
    injuries_declared: bool
    consistency_issue_count: int = Field(ge=0)


class FeatureEngineeringError(Exception):
    pass
