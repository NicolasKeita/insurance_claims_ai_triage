"""Test transactions."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from persistence.models import (
    ClaimRow,
    CustomerRow,
    DeclaredDamageRow,
    DocumentRow,
    PolicyRow,
)
from persistence.repositories import (
    ClaimRepository,
)
from persistence.service import PersistenceService
from tests.domain.persistence.mapping._support import make_anomaly, make_triage
from tests.domain.persistence.postgres._constants import pytestmark as pytestmark
from tests.domain.persistence.postgres._support import factory
from tests.domain.persistence.postgres._support import factory as factory
from tests.domain.persistence.postgres._support import postgres_engine as postgres_engine


def test_failed_document_write_rolls_back_entire_import(factory, reference_claim):
    """
    Fail a later document write after all initial inserts flushed.
    """
    with pytest.raises(IntegrityError):
        with factory.begin() as session:
            ClaimRepository(session).add(reference_claim)
            row = session.scalar(select(ClaimRow))
            session.add(
                DocumentRow(claim_pk=row.id, document_type="POLICE_REPORT", available=True, filename=None, position=4)
            )
            session.flush()
    with factory() as session:
        for model in (ClaimRow, CustomerRow, PolicyRow, DocumentRow, DeclaredDamageRow):
            assert session.scalar(select(func.count()).select_from(model)) == 0


def test_assessment_bundle_is_atomic(factory, reference_claim):
    service = PersistenceService(factory)
    service.import_claim(reference_claim)
    bad_anomaly = make_anomaly()
    bad_anomaly.assessment.anomaly_score = float("nan")
    with pytest.raises(ValueError):
        service.save_assessments(reference_claim.claim_id, triage=make_triage(), anomaly=bad_anomaly)
    assert service.get_history(reference_claim.claim_id).triage == ()
