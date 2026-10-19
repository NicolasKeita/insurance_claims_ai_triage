"""Customer, policy, claim and claim-child tables."""

from datetime import date
from decimal import Decimal
from uuid import UUID

from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from claims.enums import ClaimStatus, ClaimType, CollisionType, DamageType, DocumentType
from persistence.models.base import Base, IdentityMixin, TimestampMixin, enum_column


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
    collision_type: Mapped[CollisionType] = mapped_column(enum_column(CollisionType, "collision_type"), index=True)
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
    documents: Mapped[list["DocumentRow"]] = relationship(cascade="all, delete-orphan", order_by="DocumentRow.position")
    images: Mapped[list["ImageRow"]] = relationship(cascade="all, delete-orphan", order_by="ImageRow.position")


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
    claim_pk: Mapped[UUID] = mapped_column(ForeignKey("claims.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str]
    position: Mapped[int]
