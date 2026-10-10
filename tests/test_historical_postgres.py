"""Business-time history and API snapshot checks against real PostgreSQL."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select

from api.main import create_app
from investigation.models import SignalCategory
from persistence.history import HistoricalClaimRepository
from persistence.mappers import anomaly_to_row
from persistence.models import AnomalyAssessmentRow, ClaimRow, InvestigationAssessmentRow
from persistence.repositories import AssessmentRepository, ClaimNotFoundError, ClaimRepository
from persistence.service import PersistenceService
from test_investigation import make_consistency
from test_persistence import make_anomaly, make_investigation, reference_claim
from test_persistence_postgres import factory, postgres_engine

pytestmark = pytest.mark.postgres


def add_claim(session, reference, suffix, days, amount="100", currency="EUR", customer=None):
    payload = reference.model_dump()
    payload["claim_id"] = f"HISTORY-TEST-{suffix}"
    payload["incident"]["date"] = reference.incident.date - timedelta(days=days)
    payload["repair_estimate"] = {"amount": amount, "currency": currency}
    if customer:
        payload["customer"] = {"customer_id": customer}
    claim = type(reference).model_validate(payload)
    ClaimRepository(session).add(claim)
    return claim.claim_id


def test_temporal_customer_same_day_and_created_at_independence(factory, reference_claim):
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        add_claim(session, reference_claim, "previous", 2)
        add_claim(session, reference_claim, "later", -1)
        add_claim(session, reference_claim, "same-day", 0)
        add_claim(session, reference_claim, "other-customer", 1, customer="OTHER")
        # Insertion time deliberately contradicts incident chronology.
        current = session.scalar(select(ClaimRow).where(ClaimRow.claim_id == reference_claim.claim_id))
        current.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert facts.previous_claim_count == facts.claims_last_30_days == 1
        assert facts.days_since_previous_claim == 2
        assert facts.total_previous_repair_amount == Decimal("100")


@pytest.mark.parametrize("offset,counts", [
    (0, (0, 0, 0)), (1, (1, 1, 1)), (29, (1, 1, 1)), (30, (1, 1, 1)),
    (31, (0, 1, 1)), (89, (0, 1, 1)), (90, (0, 1, 1)), (91, (0, 0, 1)),
    (364, (0, 0, 1)), (365, (0, 0, 1)), (366, (0, 0, 0)),
])
def test_precise_window_boundaries(factory, reference_claim, offset, counts):
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        add_claim(session, reference_claim, f"boundary-{offset}", offset)
        facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert (facts.claims_last_30_days, facts.claims_last_90_days, facts.claims_last_365_days) == counts
        assert facts.days_since_previous_claim == (offset or None)


def test_most_recent_date_and_decimal_money_with_currency_isolation(factory, reference_claim):
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        add_claim(session, reference_claim, "eur-older", 40, "0.10")
        add_claim(session, reference_claim, "eur-recent", 2, "0.20")
        add_claim(session, reference_claim, "usd", 1, "9007199254740993.12345678", "USD")
        facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert facts.previous_claim_count == 3
        assert facts.days_since_previous_claim == 1
        assert facts.repair_amount_currency == "EUR"
        assert facts.previous_claims_in_amount_currency == 2
        assert facts.total_previous_repair_amount == Decimal("0.30")
        assert facts.average_previous_repair_amount == Decimal("0.15")
        assert facts.max_previous_repair_amount == Decimal("0.20")


def test_no_compatible_currency_preserves_counts_with_null_amounts(factory, reference_claim):
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        add_claim(session, reference_claim, "usd-only", 1, "999", "USD")
        facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert facts.previous_claim_count == 1
        assert facts.previous_claims_in_amount_currency == 0
        assert facts.total_previous_repair_amount == Decimal("0")
        assert facts.average_previous_repair_amount is facts.max_previous_repair_amount is None


def test_latest_anomaly_not_alert_rows_and_insertion_order_timestamp_tie(factory, reference_claim):
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
                # Reverse UUID ordering: the newer row must still win a tie.
                row.id = UUID(int=(20 if claim_id == first else 40) - i)
                row.created_at = timestamp
                session.add(row)
                session.flush()
            assert repository.get_latest_anomaly(claim_id).result.assessment.is_anomalous is new_flag
        repository.add_anomaly(same_day, make_anomaly())
        repository.add_anomaly(reference_claim.claim_id, make_anomaly())
        # Previous investigation outputs must never feed into history.
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
        assert HistoricalClaimRepository(session).get_profile(reference_claim.claim_id).previous_anomalous_claim_count == 1


def test_empty_history_unknown_claim_and_two_queries_independent_of_volume(factory, reference_claim):
    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        empty = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        assert empty.previous_claim_count == 0 and empty.days_since_previous_claim is None
        assert empty.total_previous_repair_amount == Decimal("0")
        assert empty.average_previous_repair_amount is empty.max_previous_repair_amount is None
        assert empty.repair_amount_currency == "EUR"
        with pytest.raises(ClaimNotFoundError):
            HistoricalClaimRepository(session).get_profile("unknown")
        for i in range(30):
            previous = add_claim(session, reference_claim, str(i), i + 1)
            AssessmentRepository(session).add_anomaly(previous, make_anomaly())
        statements = []
        connection = session.connection()

        def record(conn, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        event.listen(connection, "before_cursor_execute", record)
        try:
            facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        finally:
            event.remove(connection, "before_cursor_execute", record)
        assert len(statements) == 2
        assert facts.previous_claim_count == facts.previous_anomalous_claim_count == 30
        assert "row_number() OVER" in statements[1]


def test_api_history_and_persisted_v2_signals_snapshot(factory, reference_claim):
    from scripts.seed_historical_claims import seed

    service = PersistenceService(factory)
    service.import_claim(reference_claim)
    with factory.begin() as session:
        assert len(seed(session)) == 9
        assert all("Existing fixture verified" in message for message in seed(session))

    calls = []

    class Predictor:
        def assess(self, features):
            calls.append(features)
            result = make_anomaly()
            result.assessment.is_anomalous = False
            result.assessment.reference_percentile = 0.5
            return result

    def forbidden_triage():
        raise AssertionError("History/investigation does not need a triage model")

    app = create_app(forbidden_triage, lambda: Predictor(), session_factory=factory)
    path = f"/v1/claims/{reference_claim.claim_id}"
    with TestClient(app) as client:
        response = client.get(path + "/history")
        assert response.status_code == 200
        assert response.json()["previous_claim_count"] == 6
        assert response.json()["previous_anomalous_claim_count"] == 2
        assert calls == []
        assert client.get("/v1/claims/unknown/history").status_code == 404
        assert client.post("/v1/claims/unknown/investigation/assess",
                           json={"consistency": make_consistency().model_dump(mode="json")}).status_code == 404
        payload = {"consistency": make_consistency().model_dump(mode="json")}
        assert client.post(path + "/investigation/assess", json={**payload, "previous_claim_count": 999}).status_code == 422
        response = client.post(path + "/investigation/assess", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["policy_version"] == "investigation_policy_v2"
        assert body["review_score"] == 45  # 33 history + 12 missing documents
        assert body["review_priority"] == "ELEVATED"
        assert len(calls) == 1
        assert calls[0].repair_amount == float(reference_claim.repair_estimate.amount)
        assert client.post(path + "/investigation/assess", json=payload).json() == body
    history = service.get_history(reference_claim.claim_id)
    assert len(history.investigation) == len(history.anomaly) == 2
    stored = history.investigation[0].result
    assert stored.model_dump(mode="json") == body
    historical_signals = [s for s in stored.signals if s.category == SignalCategory.HISTORICAL_CONTEXT]
    assert historical_signals[1].evidence["previous_anomalous_claim_count"] == 2
    with factory.begin() as session:
        result = make_anomaly()
        result.assessment.is_anomalous = False
        AssessmentRepository(session).add_anomaly("DEMO-STEP25-HISTORY-P03", result)
    assert service.get_historical_profile(reference_claim.claim_id).previous_anomalous_claim_count == 1
    # Re-reading the stored snapshot preserves the old evidence exactly.
    assert service.get_history(reference_claim.claim_id).investigation[0].result == stored


def test_seed_rejects_unexpected_fixture_without_overwriting(factory, reference_claim):
    from scripts.seed_historical_claims import seed

    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        collision = reference_claim.model_copy(update={"claim_id": "DEMO-STEP25-HISTORY-P03"})
        ClaimRepository(session).add(collision)
        with pytest.raises(ValueError, match="Unexpected data"):
            seed(session)
        assert ClaimRepository(session).get_by_claim_id(collision.claim_id) == collision
        assert session.scalar(select(func.count()).select_from(ClaimRow)) == 2


def test_api_rolls_back_anomaly_if_investigation_write_fails(factory, reference_claim, monkeypatch):
    from sqlalchemy.exc import IntegrityError

    PersistenceService(factory).import_claim(reference_claim)

    class Predictor:
        def assess(self, features):
            return make_anomaly()

    def fail(*args, **kwargs):
        raise IntegrityError("demo failure", {}, Exception("write rejected"))

    monkeypatch.setattr(AssessmentRepository, "add_investigation", fail)
    app = create_app(anomaly_predictor_loader=lambda: Predictor(), session_factory=factory)
    with TestClient(app) as client:
        result = client.post(f"/v1/claims/{reference_claim.claim_id}/investigation/assess",
                             json={"consistency": make_consistency().model_dump(mode="json")})
        assert result.status_code == 503
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(AnomalyAssessmentRow)) == 0
        assert session.scalar(select(func.count()).select_from(InvestigationAssessmentRow)) == 0


def test_migration_backfills_legacy_rows_and_advances_identity(factory, reference_claim):
    from pathlib import Path
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    with factory.begin() as session:
        ClaimRepository(session).add(reference_claim)
        pk = ClaimRepository(session).require_pk(reference_claim.claim_id)
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in (2, 1):
            row = anomaly_to_row(pk, make_anomaly(str(i)))
            row.id, row.created_at = UUID(int=i), timestamp
            session.add(row)
            session.flush()
        cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
        cfg.attributes["connection"] = session.connection()
        command.downgrade(cfg, "0001")
        command.upgrade(cfg, "head")
        old = list(session.execute(text("SELECT id, insertion_order FROM anomaly_assessments ORDER BY insertion_order")))
        assert old == [(UUID(int=1), 1), (UUID(int=2), 2)]
        row = anomaly_to_row(pk, make_anomaly("new"))
        row.created_at = timestamp
        session.add(row)
        session.flush()
        assert row.insertion_order == 3
        assert AssessmentRepository(session).get_latest_anomaly(reference_claim.claim_id).id == row.id
