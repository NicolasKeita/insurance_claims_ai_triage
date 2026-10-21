"""Test constraints."""

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from persistence.models import (
    CustomerRow,
    DeclaredDamageRow,
)
from persistence.repositories import (
    ClaimNotFoundError,
    ClaimRepository,
    PolicyConflictError,
)
from persistence.service import PersistenceService
from tests.domain.persistence.mapping._support import make_triage
from tests.domain.persistence.postgres._constants import pytestmark as pytestmark
from tests.domain.persistence.postgres._support import factory
from tests.domain.persistence.postgres._support import factory as factory
from tests.domain.persistence.postgres._support import postgres_engine as postgres_engine


def test_missing_claim_and_conflicting_policy_are_explicit(factory, reference_claim):
    service = PersistenceService(factory)
    with pytest.raises(ClaimNotFoundError):
        service.save_assessments("missing", triage=make_triage())
    service.import_claim(reference_claim)
    second = reference_claim.model_copy(deep=True, update={"claim_id": "OTHER"})
    second.customer.customer_id = "OTHER_CUSTOMER"
    second.policy.product = "CONFLICTING_PRODUCT"
    with pytest.raises(PolicyConflictError):
        service.import_claim(second)
    with factory() as session:
        assert not ClaimRepository(session).exists("OTHER")
        assert session.scalar(select(func.count()).select_from(CustomerRow)) == 1


def test_database_rejects_duplicate_damage_pair(factory, reference_claim):
    service = PersistenceService(factory)
    service.import_claim(reference_claim)
    with pytest.raises(IntegrityError):
        with factory.begin() as session:
            pk = ClaimRepository(session).require_pk(reference_claim.claim_id)
            session.add(DeclaredDamageRow(claim_pk=pk, damage_type="FRONT_BUMPER", position=3))
