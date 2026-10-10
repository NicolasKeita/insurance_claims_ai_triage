"""Opt-in real PostgreSQL audit persistence and append-only guarantees."""

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DBAPIError

from persistence.agent_runs import AgentRunAlreadyExistsError, AgentRunStore
from persistence.models import (
    AgentRunRow, AnomalyAssessmentRow, ClaimRow, InvestigationAssessmentRow, TriageAssessmentRow,
)
from persistence.repositories import ClaimNotFoundError
from persistence.service import PersistenceService
from test_agent_persistence import add_policy_evidence, make_agent_run
from test_persistence import reference_claim
from test_persistence_postgres import factory, postgres_engine

pytestmark = pytest.mark.postgres


def test_agent_runs_coexist_json_round_trip_and_latest_timestamp_tie(factory, reference_claim):
    PersistenceService(factory).import_claim(reference_claim)
    store = AgentRunStore(factory)
    assert store.latest_for_claim(reference_claim.claim_id) is None
    timestamp = datetime(2026, 10, 10, 10, tzinfo=timezone.utc)
    first = add_policy_evidence(make_agent_run(reference_claim.claim_id, timestamp=timestamp))
    second = make_agent_run(reference_claim.claim_id, timestamp=timestamp)
    assert store.save(first) == first
    assert store.save(second) == second
    assert first.run_id != second.run_id
    assert store.get_by_run_id(first.run_id) == first
    assert store.get_by_run_id(second.run_id) == second
    assert store.get_by_run_id("unknown") is None
    assert store.list_for_claim(reference_claim.claim_id) == [second, first]
    assert store.list_for_claim(reference_claim.claim_id, limit=1) == [second]
    assert store.latest_for_claim(reference_claim.claim_id) == second
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(AgentRunRow)) == 2
        assert session.scalar(select(func.count()).select_from(ClaimRow)) == 1
        for assessment in (TriageAssessmentRow, AnomalyAssessmentRow, InvestigationAssessmentRow):
            assert session.scalar(select(func.count()).select_from(assessment)) == 0


def test_agent_run_list_uses_timestamp_before_identity_counter(factory, reference_claim):
    PersistenceService(factory).import_claim(reference_claim)
    store = AgentRunStore(factory)
    newer = make_agent_run(reference_claim.claim_id)
    older = make_agent_run(reference_claim.claim_id, timestamp=newer.created_at - timedelta(days=1))
    store.save(newer)
    store.save(older)
    assert store.latest_for_claim(reference_claim.claim_id) == newer
    assert store.list_for_claim(reference_claim.claim_id) == [newer, older]


def test_agent_failed_run_is_saved_with_sanitized_error(factory, reference_claim):
    PersistenceService(factory).import_claim(reference_claim)
    store = AgentRunStore(factory)
    failed = make_agent_run(reference_claim.claim_id, failed=True)
    assert store.save(failed) == failed
    loaded = store.get_by_run_id(failed.run_id)
    assert loaded.recommendation is None
    assert loaded.error == "Infrastructure tool failed."
    assert loaded.status.value == "FAILED"


def test_agent_duplicate_run_rejected_without_replacing_original(factory, reference_claim):
    PersistenceService(factory).import_claim(reference_claim)
    store = AgentRunStore(factory)
    source = make_agent_run(reference_claim.claim_id)
    store.save(source)
    with pytest.raises(AgentRunAlreadyExistsError):
        store.save(source)
    assert store.list_for_claim(reference_claim.claim_id) == [source]


def test_agent_unknown_claim_raises_existing_domain_error(factory):
    store = AgentRunStore(factory)
    with pytest.raises(ClaimNotFoundError):
        store.save(make_agent_run("missing"))
    with pytest.raises(ClaimNotFoundError):
        store.list_for_claim("missing")
    with pytest.raises(ClaimNotFoundError):
        store.latest_for_claim("missing")


@pytest.mark.parametrize("statement", [
    "UPDATE agent_runs SET objective = 'changed' WHERE run_id = :run_id",
    "DELETE FROM agent_runs WHERE run_id = :run_id",
])
def test_agent_database_trigger_forbids_update_and_delete(factory, reference_claim, statement):
    PersistenceService(factory).import_claim(reference_claim)
    source = make_agent_run(reference_claim.claim_id)
    store = AgentRunStore(factory)
    store.save(source)
    with pytest.raises(DBAPIError, match="append-only"):
        with factory.begin() as session:
            session.execute(text(statement), {"run_id": source.run_id})
    assert store.get_by_run_id(source.run_id) == source
