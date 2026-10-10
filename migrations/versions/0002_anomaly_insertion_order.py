"""Reliable timestamp tie-break for anomaly latest-state queries.

Revision ID: 0002
Revises: 0001
Legacy equal timestamps cannot reveal original insertion order. Backfill keeps
the old deterministic UUID tie-break only for those existing ambiguous rows.
New rows always receive a database-generated monotonic allocation counter.
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("anomaly_assessments", sa.Column("insertion_order", sa.BigInteger(), nullable=True))
    op.execute("""
        WITH numbered AS (
            SELECT id, row_number() OVER (ORDER BY created_at, id) AS ordinal
            FROM anomaly_assessments
        )
        UPDATE anomaly_assessments AS assessment
        SET insertion_order = numbered.ordinal
        FROM numbered WHERE assessment.id = numbered.id
    """)
    op.alter_column("anomaly_assessments", "insertion_order", nullable=False)
    op.execute("ALTER TABLE anomaly_assessments ALTER COLUMN insertion_order ADD GENERATED ALWAYS AS IDENTITY")
    op.execute("""
        SELECT setval(pg_get_serial_sequence('anomaly_assessments', 'insertion_order'),
                      COALESCE(MAX(insertion_order), 1), COUNT(*) > 0)
        FROM anomaly_assessments
    """)
    op.create_unique_constraint("uq_anomaly_assessments_insertion_order", "anomaly_assessments", ["insertion_order"])
    op.drop_index("ix_anomaly_assessments_claim_date", table_name="anomaly_assessments")
    op.create_index("ix_anomaly_assessments_claim_latest", "anomaly_assessments", ["claim_pk", "created_at", "insertion_order"])


def downgrade():
    op.drop_index("ix_anomaly_assessments_claim_latest", table_name="anomaly_assessments")
    op.create_index("ix_anomaly_assessments_claim_date", "anomaly_assessments", ["claim_pk", "created_at", "id"])
    op.drop_constraint("uq_anomaly_assessments_insertion_order", "anomaly_assessments", type_="unique")
    op.drop_column("anomaly_assessments", "insertion_order")
