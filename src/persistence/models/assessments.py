"""Append-only assessment, vision and agent-run snapshot tables."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Identity, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from claims.enums import TriageWorkflow
from claims.investigation.models import ReviewPriority
from persistence.models.base import Base, CreatedMixin, IdentityMixin, enum_column


class AssessmentMixin(IdentityMixin, CreatedMixin):
    # RESTRICT keeps observed history attached to its claim.
    claim_pk: Mapped[UUID] = mapped_column(ForeignKey("claims.id"))


class ModelMetadataMixin:
    registered_model: Mapped[str]
    model_version: Mapped[str]
    model_alias: Mapped[str]


class TriageAssessmentRow(AssessmentMixin, ModelMetadataMixin, Base):
    __tablename__ = "triage_assessments"
    __table_args__ = (
        Index("ix_triage_assessments_claim_date", "claim_pk", "created_at", "id"),
        CheckConstraint("confidence BETWEEN 0 AND 1", name="confidence"),
    )
    recommended_workflow: Mapped[TriageWorkflow] = mapped_column(enum_column(TriageWorkflow, "triage_workflow"))
    confidence: Mapped[float]
    probabilities: Mapped[dict[str, float]] = mapped_column(JSONB)


class AnomalyAssessmentRow(AssessmentMixin, ModelMetadataMixin, Base):
    __tablename__ = "anomaly_assessments"
    __table_args__ = (
        Index("ix_anomaly_assessments_claim_latest", "claim_pk", "created_at", "insertion_order"),
        UniqueConstraint("insertion_order"),
        CheckConstraint("reference_percentile BETWEEN 0 AND 1", name="percentile"),
    )
    is_anomalous: Mapped[bool]
    insertion_order: Mapped[int] = mapped_column(BigInteger, Identity(always=True))
    anomaly_score: Mapped[float]
    threshold: Mapped[float]
    reference_percentile: Mapped[float | None]
    unusual_features: Mapped[list[str]] = mapped_column(JSONB)


class InvestigationAssessmentRow(AssessmentMixin, Base):
    __tablename__ = "investigation_assessments"
    __table_args__ = (
        Index("ix_investigation_assessments_claim_date", "claim_pk", "created_at", "id"),
        CheckConstraint("review_score BETWEEN 0 AND 100", name="review_score"),
    )
    review_score: Mapped[int]
    review_priority: Mapped[ReviewPriority] = mapped_column(enum_column(ReviewPriority, "review_priority"))
    review_recommended: Mapped[bool]
    policy_version: Mapped[str]
    signals: Mapped[list[dict]] = mapped_column(JSONB)


class VisionAssessmentRow(AssessmentMixin, Base):
    """Append-only multimodal snapshots; normalized polygons, never raster masks."""

    __tablename__ = "vision_assessments"
    __table_args__ = (
        UniqueConstraint("insertion_order"),
        Index("ix_vision_assessments_claim_latest", "claim_pk", "created_at", "insertion_order"),
    )
    insertion_order: Mapped[int] = mapped_column(BigInteger, Identity(always=True))
    damage_model_identifier: Mapped[str]
    damage_model_version: Mapped[str | None]
    part_model_identifier: Mapped[str]
    part_model_version: Mapped[str | None]
    fusion_policy_version: Mapped[str]
    structured_results: Mapped[dict] = mapped_column(JSONB)


class AgentRunRow(IdentityMixin, Base):
    """Immutable audit snapshot; no relationship cascades or business writes."""

    __tablename__ = "agent_runs"
    __table_args__ = (
        UniqueConstraint("insertion_order"),
        Index("ix_agent_runs_claim_latest", "claim_pk", "created_at", "insertion_order"),
        CheckConstraint("status IN ('COMPLETED', 'FAILED')", name="status"),
        CheckConstraint("iteration_count >= 0 AND tool_call_count >= 0", name="nonnegative_counts"),
        CheckConstraint("elapsed_seconds >= 0", name="nonnegative_elapsed"),
        CheckConstraint("completed_at >= created_at", name="completion_time"),
        CheckConstraint(
            "(status = 'COMPLETED' AND recommendation IS NOT NULL AND error IS NULL) OR "
            "(status = 'FAILED' AND recommendation IS NULL AND error IS NOT NULL)",
            name="terminal_result",
        ),
    )
    claim_pk: Mapped[UUID] = mapped_column(ForeignKey("claims.id"))
    run_id: Mapped[str] = mapped_column(unique=True)
    insertion_order: Mapped[int] = mapped_column(BigInteger, Identity(always=True))
    objective: Mapped[str]
    status: Mapped[str]
    agent_version: Mapped[str]
    prompt_version: Mapped[str]
    llm_model: Mapped[str | None]
    iteration_count: Mapped[int]
    tool_call_count: Mapped[int]
    tools_used: Mapped[list[str]] = mapped_column(JSONB)
    evidence: Mapped[list[dict]] = mapped_column(JSONB)
    trace: Mapped[list[dict]] = mapped_column(JSONB)
    recommendation: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    elapsed_seconds: Mapped[float]
    error: Mapped[str | None]
