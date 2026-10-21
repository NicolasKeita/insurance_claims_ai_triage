"""Test anomaly."""

from datetime import datetime, timezone
from uuid import UUID

from persistence.history import HistoricalClaimRepository
from persistence.mappers import anomaly_to_row
from persistence.repositories import (
    AssessmentRepository,
    ClaimRepository,
)
from tests.domain.investigation.history_postgres._constants import pytestmark as pytestmark
from tests.domain.investigation.history_postgres._support import add_claim
from tests.domain.persistence.mapping._support import make_anomaly, make_investigation


def test_latest_anomaly_not_alert_rows_and_insertion_order_timestamp_tie(factory, reference_claim):
    """
    Reverse UUID ordering: the newer row must still win a tie.

    Previous investigation outputs must never feed into history.
    """
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        first = add_claim(session, reference_claim, "normal-now", 1)
        second = add_claim(session, reference_claim, "anomalous-now", 2)
        add_claim(session, reference_claim, "no-assessments", 3)
        same_day = add_claim(session, reference_claim, "same-day", 0)
        repository = AssessmentRepository(session)
        for claim_id, old_flag, new_flag in ((first, True, False), (second, False, True)):
            pk = ClaimRepository(session).require_pk(claim_id)
            for i, flag in enumerate((old_flag, new_flag)):
                result = make_anomaly(str(i))
                result.assessment.is_anomalous = flag
                row = anomaly_to_row(pk, result)
                row.id = UUID(int=(20 if claim_id == first else 40) - i)
                row.created_at = timestamp
                session.add(row)
                session.flush()
            assert repository.get_latest_anomaly(claim_id).result.assessment.is_anomalous is new_flag
        repository.add_anomaly(same_day, make_anomaly())
        repository.add_anomaly(reference_claim.claim_id, make_anomaly())
        repository.add_investigation(first, make_investigation())
        facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert facts.previous_claim_count == 3
        assert facts.previous_anomalous_claim_count == 1


def test_latest_anomaly_timestamp_precedes_counter(factory, reference_claim):
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        previous = add_claim(session, reference_claim, "backdated", 1)
        pk = ClaimRepository(session).require_pk(previous)
        for year, flag in ((2026, True), (2025, False)):
            result = make_anomaly(str(year))
            result.assessment.is_anomalous = flag
            row = anomaly_to_row(pk, result)
            row.created_at = datetime(year, 1, 1, tzinfo=timezone.utc)
            session.add(row)
            session.flush()
        assert (
            HistoricalClaimRepository(session).get_profile(reference_claim.claim_id).previous_anomalous_claim_count == 1
        )
