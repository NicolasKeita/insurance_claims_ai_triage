from pydantic import BaseModel, Field

from claims.enums import (
    ClaimType,
    CollisionType,
)
from claims.case import ClaimCase
from claims.consistency import (
    ClaimConsistencyReport,
)
from claims.document_models import (
    ClaimFormExtraction,
    GarageQuoteExtraction,
)


class TriageFeatures(BaseModel):
    claim_type: ClaimType
    collision_type: CollisionType

    vehicle_age: int = Field(ge=0)

    repair_amount: float = Field(ge=0)

    declared_damage_count: int = Field(ge=0)
    additional_damage_count: int = Field(ge=0)

    document_count: int = Field(ge=0)
    missing_document_count: int = Field(ge=0)

    document_completeness_ratio: float = Field(
        ge=0,
        le=1,
    )

    injuries_declared: bool

    consistency_issue_count: int = Field(
        ge=0
    )

class FeatureEngineeringError(Exception):
    pass

def calculate_vehicle_age(
    vehicle_year: int,
    incident_year: int,
) -> int:
    age = incident_year - vehicle_year

    if age < 0:
        raise FeatureEngineeringError(
            "Vehicle year is after incident year"
        )

    return age

def calculate_document_completeness(
    case: ClaimCase,
) -> tuple[int, int, float]:

    documents = case.claim.documents

    document_count = len(documents)

    if document_count == 0:
        return 0, 0, 1.0

    missing_count = sum(
        1
        for document in documents
        if not document.available
    )

    available_count = (
        document_count - missing_count
    )

    ratio = (
        available_count
        / document_count
    )

    return (
        document_count,
        missing_count,
        ratio,
    )

def count_consistency_issues(
    report: ClaimConsistencyReport,
) -> int:

    checks = (
        report.claim_id_match,
        report.incident_date_match,
        report.location_match,
        report.vehicle_match,
        report.collision_type_match,
        report.injuries_match,
        report.declared_damage_match,
    )
    mismatches = sum(
        1
        for check in checks
        if not check
    )
    return (
        mismatches
        + len(
            report.additional_quote_damage
        )
        + len(
            report.unmapped_quote_items
        )
    )

def build_triage_features(
    case: ClaimCase,
    claim_form: ClaimFormExtraction,
    garage_quote: GarageQuoteExtraction,
    consistency: ClaimConsistencyReport,
) -> TriageFeatures:

    vehicle_age = calculate_vehicle_age(
        vehicle_year=claim_form.vehicle_year,
        incident_year=(
            claim_form.incident_date.year
        ),
    )
    (
        document_count,
        missing_document_count,
        document_completeness_ratio,
    ) = calculate_document_completeness(
        case
    )
    consistency_issue_count = (
        count_consistency_issues(
            consistency
        )
    )
    return TriageFeatures(
        claim_type=case.claim.claim_type,

        collision_type=(
            claim_form.collision_type
        ),

        vehicle_age=vehicle_age,

        repair_amount=float(
            garage_quote.total.amount
        ),

        declared_damage_count=len(
            claim_form.declared_damage
        ),

        additional_damage_count=len(
            consistency.additional_quote_damage
        ),

        document_count=document_count,

        missing_document_count=(
            missing_document_count
        ),

        document_completeness_ratio=(
            document_completeness_ratio
        ),

        injuries_declared=(
            claim_form.injuries_declared
        ),

        consistency_issue_count=(
            consistency_issue_count
        ),
    )