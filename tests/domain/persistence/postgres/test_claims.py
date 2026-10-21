"""Test claims."""

from decimal import Decimal

import pytest
from sqlalchemy import func, inspect, select

from persistence.models import (
    ClaimRow,
    CustomerRow,
    PolicyRow,
)
from persistence.repositories import (
    ClaimAlreadyExistsError,
    ClaimRepository,
)
from persistence.service import PersistenceService
from tests.domain.persistence.postgres._constants import pytestmark as pytestmark
from tests.domain.persistence.postgres._support import factory, postgres_engine
from tests.domain.persistence.postgres._support import factory as factory
from tests.domain.persistence.postgres._support import postgres_engine as postgres_engine


def test_migrations_upgrade_downgrade_and_metadata_match(postgres_engine):
    with postgres_engine.connect() as connection:
        columns = {c["name"]: c["type"] for c in inspect(connection).get_columns("claims")}
        assert columns["created_at"].timezone
        assert str(columns["repair_estimate_amount"]) == "NUMERIC"
        assert str(columns["id"]) == "UUID"


def test_save_reload_reuse_and_reject_duplicate(factory, reference_claim):
    service = PersistenceService(factory)
    first = service.import_claim(reference_claim)
    assert first.customer_created and first.policy_created
    assert first.document_count == 4 and first.image_count == 2
    with pytest.raises(ClaimAlreadyExistsError):
        service.import_claim(reference_claim)
    second = reference_claim.model_copy(deep=True, update={"claim_id": "CLAIM-2026-00002"})
    result = service.import_claim(second)
    assert not result.customer_created and not result.policy_created
    history = service.get_history(reference_claim.claim_id)
    assert history.claim == reference_claim
    assert history.triage == history.anomaly == history.investigation == ()
    with factory() as session:
        repository = ClaimRepository(session)
        assert repository.exists(reference_claim.claim_id)
        assert repository.get_by_claim_id("missing") is None
        assert len(repository.list_for_customer(reference_claim.customer.customer_id)) == 2
        assert len(repository.list_for_policy(reference_claim.policy.policy_id)) == 2
        assert session.scalar(select(func.count()).select_from(CustomerRow)) == 1
        assert session.scalar(select(func.count()).select_from(PolicyRow)) == 1
        claims = list(session.scalars(select(ClaimRow)))
        assert claims[0].customer_pk == claims[1].customer_pk
        assert claims[0].policy_pk == claims[1].policy_pk


def test_decimal_round_trip(factory, reference_claim):
    claim = reference_claim.model_copy(deep=True)
    claim.repair_estimate.amount = Decimal("9007199254740993.12345678")
    service = PersistenceService(factory)
    service.import_claim(claim)
    assert service.get_history(claim.claim_id).claim.repair_estimate.amount == claim.repair_estimate.amount
