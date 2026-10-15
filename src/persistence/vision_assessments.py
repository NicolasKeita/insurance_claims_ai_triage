"""Append/read multimodal assessments already computed by the vision service.

No detector is constructed here. The JSONB snapshot contains normalized geometry
and model/fusion metadata, rather than raster masks or dataset images.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from uuid import UUID

from sqlalchemy import select

from persistence.database import transaction
from persistence.models import VisionAssessmentRow
from persistence.repositories import ClaimRepository
from vision.models import ClaimMultimodalVisionAssessment


@dataclass(frozen=True)
class StoredVisionAssessment:
    id: UUID
    created_at: datetime
    result: ClaimMultimodalVisionAssessment


def vision_assessment_to_row(
    claim_pk: UUID, result: ClaimMultimodalVisionAssessment,
) -> VisionAssessmentRow:
    checked = ClaimMultimodalVisionAssessment.model_validate(
        result.model_dump(round_trip=True)
    )
    json.dumps(checked.model_dump(round_trip=True), default=str, allow_nan=False)
    payload = json.loads(json.dumps(
        checked.model_dump(mode="json", round_trip=True), allow_nan=False,
    ))
    return VisionAssessmentRow(
        claim_pk=claim_pk,
        damage_model_identifier=checked.damage_model_metadata.identifier,
        damage_model_version=checked.damage_model_metadata.version,
        part_model_identifier=checked.part_model_metadata.identifier,
        part_model_version=checked.part_model_metadata.version,
        fusion_policy_version=checked.fusion_policy.version,
        structured_results=payload,
    )


def vision_assessment_from_row(
    row: VisionAssessmentRow, *, expected_claim_id: str | None = None,
) -> StoredVisionAssessment:
    result = ClaimMultimodalVisionAssessment.model_validate(row.structured_results)
    if expected_claim_id is not None and result.claim_id != expected_claim_id:
        raise ValueError("Vision assessment snapshot belongs to a different claim")
    if (
        row.damage_model_identifier != result.damage_model_metadata.identifier
        or row.damage_model_version != result.damage_model_metadata.version
        or row.part_model_identifier != result.part_model_metadata.identifier
        or row.part_model_version != result.part_model_metadata.version
        or row.fusion_policy_version != result.fusion_policy.version
    ):
        raise ValueError("Vision assessment metadata differs from its evidence snapshot")
    return StoredVisionAssessment(row.id, row.created_at, result)


class VisionAssessmentRepository:
    """INSERT/SELECT only; the caller owns its transaction and commit."""

    def __init__(self, session):
        self.session = session
        self.claims = ClaimRepository(session)

    def add(self, result: ClaimMultimodalVisionAssessment) -> StoredVisionAssessment:
        row = vision_assessment_to_row(self.claims.require_pk(result.claim_id), result)
        self.session.add(row)
        self.session.flush()
        return vision_assessment_from_row(row, expected_claim_id=result.claim_id)

    def list_for_claim(self, claim_id: str, limit: int = 20) -> list[StoredVisionAssessment]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Vision assessment list limit must be between 1 and 100")
        pk = self.claims.require_pk(claim_id)
        rows = self.session.scalars(
            select(VisionAssessmentRow).where(VisionAssessmentRow.claim_pk == pk)
            .order_by(VisionAssessmentRow.created_at.desc(), VisionAssessmentRow.insertion_order.desc())
            .limit(limit)
        )
        return [vision_assessment_from_row(row, expected_claim_id=claim_id) for row in rows]

    def latest_for_claim(self, claim_id: str) -> StoredVisionAssessment | None:
        records = self.list_for_claim(claim_id, limit=1)
        return records[0] if records else None


class VisionAssessmentStore:
    """Transaction-owning append/read service without inference side effects."""

    def __init__(self, factory):
        self.factory = factory

    def save(self, result: ClaimMultimodalVisionAssessment) -> StoredVisionAssessment:
        with transaction(self.factory) as session:
            return VisionAssessmentRepository(session).add(result)

    def list_for_claim(self, claim_id: str, limit: int = 20) -> list[StoredVisionAssessment]:
        with self.factory() as session:
            return VisionAssessmentRepository(session).list_for_claim(claim_id, limit)

    def latest_for_claim(self, claim_id: str) -> StoredVisionAssessment | None:
        with self.factory() as session:
            return VisionAssessmentRepository(session).latest_for_claim(claim_id)
