"""Relational business columns and JSONB evidence snapshots."""

from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, Enum, ForeignKey, Identity, Index, MetaData, Numeric,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from claims.enums import (
    ClaimStatus, ClaimType, CollisionType, DamageType, DocumentType, TriageWorkflow,
)
from investigation.models import ReviewPriority


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def enum_column(enum, name: str):
    return Enum(enum, name=name, native_enum=False, create_constraint=True,
                validate_strings=True, values_callable=lambda cls: [e.value for e in cls])


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention={
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    })


class IdentityMixin:
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)


class CreatedMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TimestampMixin(CreatedMixin):
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class CustomerRow(IdentityMixin, TimestampMixin, Base):
    __tablename__ = "customers"
    customer_id: Mapped[str] = mapped_column(unique=True)
    claims: Mapped[list["ClaimRow"]] = relationship(back_populates="customer")


class PolicyRow(IdentityMixin, TimestampMixin, Base):
    __tablename__ = "policies"
    policy_id: Mapped[str] = mapped_column(unique=True)
    product: Mapped[str]
    # Claim data does not establish exclusive policy ownership by a customer.
    claims: Mapped[list["ClaimRow"]] = relationship(back_populates="policy")


class ClaimRow(IdentityMixin, TimestampMixin, Base):
    __tablename__ = "claims"
    __table_args__ = (
        CheckConstraint("repair_estimate_amount >= 0", name="nonnegative_amount"),
        CheckConstraint("vehicle_year >= 1886", name="vehicle_year"),
        CheckConstraint("repair_estimate_currency ~ '^[A-Z]{3}$'", name="currency"),
        Index("ix_claims_customer_date", "customer_pk", "incident_date"),
    )
    claim_id: Mapped[str] = mapped_column(unique=True)
    customer_pk: Mapped[UUID] = mapped_column(ForeignKey("customers.id"))
    policy_pk: Mapped[UUID] = mapped_column(ForeignKey("policies.id"), index=True)
    claim_type: Mapped[ClaimType] = mapped_column(enum_column(ClaimType, "claim_type"))
    status: Mapped[ClaimStatus] = mapped_column(enum_column(ClaimStatus, "claim_status"))
    incident_date: Mapped[date] = mapped_column(index=True)
    incident_location: Mapped[str]
    collision_type: Mapped[CollisionType] = mapped_column(
        enum_column(CollisionType, "collision_type"), index=True
    )
    injuries_declared: Mapped[bool]
    vehicle_make: Mapped[str]
    vehicle_model: Mapped[str]
    vehicle_year: Mapped[int]
    repair_estimate_amount: Mapped[Decimal] = mapped_column(Numeric())
    repair_estimate_currency: Mapped[str]
    customer: Mapped[CustomerRow] = relationship(back_populates="claims")
    policy: Mapped[PolicyRow] = relationship(back_populates="claims")
    damages: Mapped[list["DeclaredDamageRow"]] = relationship(
        cascade="all, delete-orphan", order_by="DeclaredDamageRow.position"
    )
    documents: Mapped[list["DocumentRow"]] = relationship(
        cascade="all, delete-orphan", order_by="DocumentRow.position"
    )
    images: Mapped[list["ImageRow"]] = relationship(
        cascade="all, delete-orphan", order_by="ImageRow.position"
    )


class DeclaredDamageRow(IdentityMixin, Base):
    __tablename__ = "claim_declared_damages"
    __table_args__ = (UniqueConstraint("claim_pk", "damage_type"),)
    claim_pk: Mapped[UUID] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"))
    damage_type: Mapped[DamageType] = mapped_column(enum_column(DamageType, "damage_type"))
    position: Mapped[int]


class DocumentRow(IdentityMixin, Base):
    __tablename__ = "claim_documents"
    __table_args__ = (
        CheckConstraint("NOT available OR filename IS NOT NULL", name="available_filename"),
        Index("ix_claim_documents_claim_pk", "claim_pk"),
    )
    claim_pk: Mapped[UUID] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"))
    document_type: Mapped[DocumentType] = mapped_column(enum_column(DocumentType, "document_type"))
    filename: Mapped[str | None]
    available: Mapped[bool]
    position: Mapped[int]


class ImageRow(IdentityMixin, Base):
    __tablename__ = "claim_images"
    claim_pk: Mapped[UUID] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str]
    position: Mapped[int]


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
    recommended_workflow: Mapped[TriageWorkflow] = mapped_column(
        enum_column(TriageWorkflow, "triage_workflow")
    )
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
    review_priority: Mapped[ReviewPriority] = mapped_column(
        enum_column(ReviewPriority, "review_priority")
    )
    review_recommended: Mapped[bool]
    policy_version: Mapped[str]
    signals: Mapped[list[dict]] = mapped_column(JSONB)
