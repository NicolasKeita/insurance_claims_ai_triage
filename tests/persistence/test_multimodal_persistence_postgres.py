"""Opt-in real PostgreSQL multimodal snapshots and append-only trigger."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from persistence.repositories import ClaimNotFoundError, ClaimRepository
from persistence.service import PersistenceService
from persistence.vision_assessments import (
    VisionAssessmentStore,
    vision_assessment_to_row,
)
from tests.investigation.multimodal._assessment import make_visual_assessment

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def test_vision_append_round_trip_and_latest_timestamp_tie(factory, reference_claim):
    PersistenceService(factory).import_claim(reference_claim)
    store = VisionAssessmentStore(factory)
    assert store.latest_for_claim(reference_claim.claim_id) is None
    visual = make_visual_assessment()
    timestamp = datetime(2026, 10, 14, 10, tzinfo=timezone.utc)
    with factory.begin() as session:
        pk = ClaimRepository(session).require_pk(reference_claim.claim_id)
        first = vision_assessment_to_row(pk, visual)
        second = vision_assessment_to_row(pk, visual)
        first.created_at = second.created_at = timestamp
        session.add(first)
        session.flush()
        session.add(second)
        session.flush()
        first_id, second_id = first.id, second.id
    records = store.list_for_claim(reference_claim.claim_id)
    assert [row.id for row in records] == [second_id, first_id]
    assert records[0].result == visual
    assert store.latest_for_claim(reference_claim.claim_id).id == second_id


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE vision_assessments SET part_model_version='changed' WHERE id=:id",
        "DELETE FROM vision_assessments WHERE id=:id",
    ],
)
def test_vision_trigger_forbids_update_delete(factory, reference_claim, statement):
    PersistenceService(factory).import_claim(reference_claim)
    store = VisionAssessmentStore(factory)
    saved = store.save(make_visual_assessment())
    with pytest.raises(DBAPIError, match="append-only"):
        with factory.begin() as session:
            session.execute(text(statement), {"id": saved.id})
    assert store.latest_for_claim(reference_claim.claim_id) == saved


def test_vision_unknown_claim_cannot_append_or_read(factory):
    store = VisionAssessmentStore(factory)
    with pytest.raises(ClaimNotFoundError):
        store.save(make_visual_assessment(claim_id="missing"))
    with pytest.raises(ClaimNotFoundError):
        store.latest_for_claim("missing")
