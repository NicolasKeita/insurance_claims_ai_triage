"""Real PostgreSQL migrations and transactions; no SQLite substitute.

Every run owns a randomly named schema in an explicitly dedicated *_test
database. Cleanup never targets public, development or another run's schema.
"""

import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from persistence.database import require_test_database
from persistence.models import (
    Base, ClaimRow, CustomerRow, DeclaredDamageRow, DocumentRow, PolicyRow,
    TriageAssessmentRow,
)
from persistence.repositories import (
    AssessmentRepository, ClaimAlreadyExistsError, ClaimNotFoundError,
    ClaimRepository, PolicyConflictError,
)
from persistence.service import PersistenceService
from test_persistence import reference_claim, make_anomaly, make_investigation, make_triage

pytestmark = pytest.mark.postgres


@pytest.fixture(scope="module")
def postgres_engine():
    value = os.environ.get("TEST_DATABASE_URL")
    if not value:
        pytest.skip("TEST_DATABASE_URL is not set")
    config = require_test_database(value)
    schema = f"persistence_test_{uuid4().hex}"
    admin = create_engine(config.url, hide_parameters=True)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(config.url, hide_parameters=True,
                           connect_args={"options": f"-csearch_path={schema}"})
    try:
        cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
        with engine.begin() as connection:
            assert inspect(connection).get_table_names() == []
            cfg.attributes["connection"] = connection
            command.upgrade(cfg, "head")
            assert set(Base.metadata.tables) <= set(inspect(connection).get_table_names())
            command.downgrade(cfg, "base")
            assert inspect(connection).get_table_names() == ["alembic_version"]
            command.upgrade(cfg, "head")
            assert compare_metadata(MigrationContext.configure(connection), Base.metadata) == []
        yield engine
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture
def factory(postgres_engine):
    # Application commits become savepoints; an outer rollback isolates tests.
    with postgres_engine.connect() as connection:
        outer = connection.begin()
        factory = sessionmaker(connection, expire_on_commit=False,
                               join_transaction_mode="create_savepoint")
        try:
            yield factory
        finally:
            outer.rollback()


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


def test_all_assessments_append_and_latest(factory, reference_claim):
    service = PersistenceService(factory)
    service.import_claim(reference_claim)
    with factory() as session:
        repo = AssessmentRepository(session)
        assert repo.get_latest_triage(reference_claim.claim_id) is None
        assert repo.get_latest_anomaly(reference_claim.claim_id) is None
        assert repo.get_latest_investigation(reference_claim.claim_id) is None
    first = service.save_assessments(reference_claim.claim_id, triage=make_triage("1"),
                                     anomaly=make_anomaly("1"), investigation=make_investigation("v1"))
    second = service.save_assessments(reference_claim.claim_id, triage=make_triage("2"),
                                      anomaly=make_anomaly("2"), investigation=make_investigation("v2"))
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


def test_failed_document_write_rolls_back_entire_import(factory, reference_claim):
    with pytest.raises(IntegrityError):
        with factory.begin() as session:
            ClaimRepository(session).add(reference_claim)
            row = session.scalar(select(ClaimRow))
            # Fail a later document write after all initial inserts flushed.
            session.add(DocumentRow(claim_pk=row.id, document_type="POLICE_REPORT",
                                    available=True, filename=None, position=4))
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
