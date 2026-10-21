"""Test queries."""

from decimal import Decimal

import pytest
from sqlalchemy import event

from persistence.history import HistoricalClaimRepository
from persistence.repositories import (
    AssessmentRepository,
    ClaimNotFoundError,
    ClaimRepository,
)
from tests.investigation.history_postgres._constants import pytestmark as pytestmark
from tests.investigation.history_postgres._support import add_claim
from tests.persistence.mapping._support import make_anomaly


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

        def record(conn, clauseelement, multiparams, params, execution_options):
            statement = str(clauseelement)
            statements.append(statement)

        event.listen(connection, "before_execute", record)
        try:
            facts = HistoricalClaimRepository(session).get_profile(reference_claim.claim_id)
        finally:
            event.remove(connection, "before_execute", record)
        assert len(statements) == 2
        assert facts.previous_claim_count == facts.previous_anomalous_claim_count == 30
        assert "row_number() OVER" in statements[1]
