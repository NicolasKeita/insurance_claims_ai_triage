"""Test migration."""

from datetime import datetime, timezone
from uuid import UUID

from persistence.mappers import anomaly_to_row
from persistence.repositories import (
    AssessmentRepository,
    ClaimRepository,
)
from tests.integration_support import PROJECT_ROOT
from tests.investigation.history_postgres._constants import pytestmark as pytestmark
from tests.persistence.mapping._support import make_anomaly


def test_migration_backfills_legacy_rows_and_advances_identity(factory, reference_claim):

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
        cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
        cfg.attributes["connection"] = session.connection()
        command.downgrade(cfg, "0001")
        command.upgrade(cfg, "head")
        old = list(
            session.execute(text("SELECT id, insertion_order FROM anomaly_assessments ORDER BY insertion_order"))
        )
        assert old == [(UUID(int=1), 1), (UUID(int=2), 2)]
        row = anomaly_to_row(pk, make_anomaly("new"))
        row.created_at = timestamp
        session.add(row)
        session.flush()
        assert row.insertion_order == 3
        assert AssessmentRepository(session).get_latest_anomaly(reference_claim.claim_id).id == row.id
