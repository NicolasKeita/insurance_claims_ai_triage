"""Test assessments."""

from datetime import datetime, timezone
from uuid import UUID

from persistence.repositories import (
    AssessmentRepository,
    ClaimRepository,
)
from persistence.service import PersistenceService
from tests.domain.persistence.mapping._support import (
    make_anomaly,
    make_investigation,
    make_triage,
)
from tests.domain.persistence.postgres._constants import pytestmark as pytestmark
from tests.domain.persistence.postgres._support import factory
from tests.domain.persistence.postgres._support import factory as factory
from tests.domain.persistence.postgres._support import postgres_engine as postgres_engine


def test_all_assessments_append_and_latest(factory, reference_claim):
    service = PersistenceService(factory)
    service.import_claim(reference_claim)
    with factory() as session:
        repo = AssessmentRepository(session)
        assert repo.get_latest_triage(reference_claim.claim_id) is None
        assert repo.get_latest_anomaly(reference_claim.claim_id) is None
        assert repo.get_latest_investigation(reference_claim.claim_id) is None
    first = service.save_assessments(
        reference_claim.claim_id,
        triage=make_triage("1"),
        anomaly=make_anomaly("1"),
        investigation=make_investigation("v1"),
    )
    second = service.save_assessments(
        reference_claim.claim_id,
        triage=make_triage("2"),
        anomaly=make_anomaly("2"),
        investigation=make_investigation("v2"),
    )
    history = service.get_history(reference_claim.claim_id)
    for kind in ("triage", "anomaly", "investigation"):
        records = getattr(history, kind)
        assert len(records) == 2
        assert records[0] == first[kind] and records[1] == second[kind]
        assert records[0].id != records[1].id
        assert records[0].created_at.tzinfo is not None
    with factory() as session:
        repository = AssessmentRepository(session)
        assert repository.get_latest_triage(reference_claim.claim_id) == second["triage"]
        assert repository.get_latest_anomaly(reference_claim.claim_id) == second["anomaly"]
        assert repository.get_latest_investigation(reference_claim.claim_id) == second["investigation"]
    assert history.triage[0].result == make_triage("1")
    assert history.anomaly[0].result == make_anomaly("1")
    assert history.investigation[0].result == make_investigation("v1")


def test_latest_tie_breaker_is_explicit(factory, reference_claim):
    service = PersistenceService(factory)
    service.import_claim(reference_claim)
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with factory.begin() as session:
        pk = ClaimRepository(session).require_pk(reference_claim.claim_id)
        from persistence.mappers import triage_to_row

        high = triage_to_row(pk, make_triage("higher_uuid"))
        high.id, high.created_at = UUID(int=2), timestamp
        low = triage_to_row(pk, make_triage("lower_uuid"))
        low.id, low.created_at = UUID(int=1), timestamp
        session.add_all([high, low])
    with factory() as session:
        assert AssessmentRepository(session).get_latest_triage(reference_claim.claim_id).id == UUID(int=2)
