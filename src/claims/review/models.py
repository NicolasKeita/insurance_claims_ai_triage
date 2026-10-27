"""Presentation contracts; unavailable assessments never become negative results."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from claims.models import Claim

ReviewStatus = Literal["AVAILABLE", "UNAVAILABLE", "NOT_EXECUTED", "FAILED", "NO_EVIDENCE", "INSUFFICIENT_EVIDENCE"]


class ReviewSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ReviewStatus
    reason: str | None = None
    data: JsonValue = None
    source: str | None = None
    updated_at: datetime | None = None


class ReviewEvidence(BaseModel):
    id: str
    section: str
    title: str
    data: JsonValue = None
    source: str | None = None


class ReviewAudit(BaseModel):
    timestamps: dict[str, str] = Field(default_factory=dict)
    model_versions: dict[str, str | None] = Field(default_factory=dict)
    policy_versions: dict[str, str | None] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    execution_statuses: dict[str, ReviewStatus] = Field(default_factory=dict)


class ClaimReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: Claim
    documents: ReviewSection
    consistency: ReviewSection
    triage: ReviewSection
    anomaly: ReviewSection
    history: ReviewSection
    investigation: ReviewSection
    vision: ReviewSection
    visual_consistency: ReviewSection
    similar_claims: ReviewSection
    knowledge: ReviewSection
    agent: ReviewSection
    evidence: list[ReviewEvidence] = Field(default_factory=list)
    audit: ReviewAudit = Field(default_factory=ReviewAudit)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: Literal["persisted", "demo"] = "persisted"


class DemoClaim(BaseModel):
    claim_id: str
    title: str
    description: str
