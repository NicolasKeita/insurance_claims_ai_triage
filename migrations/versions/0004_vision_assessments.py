"""Append-only multimodal perception snapshots with independent model versions.

Revision ID: 0004
Revises: 0003
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "vision_assessments",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("claim_pk", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("insertion_order", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("damage_model_identifier", sa.String(), nullable=False),
        sa.Column("damage_model_version", sa.String(), nullable=True),
        sa.Column("part_model_identifier", sa.String(), nullable=False),
        sa.Column("part_model_version", sa.String(), nullable=True),
        sa.Column("fusion_policy_version", sa.String(), nullable=False),
        sa.Column("structured_results", postgresql.JSONB(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_vision_assessments"),
        sa.ForeignKeyConstraint(["claim_pk"], ["claims.id"],
                                name="fk_vision_assessments_claim_pk_claims"),
        sa.UniqueConstraint("insertion_order", name="uq_vision_assessments_insertion_order"),
    )
    op.create_index("ix_vision_assessments_claim_latest", "vision_assessments",
                    ["claim_pk", "created_at", "insertion_order"])
    op.execute("""
        CREATE FUNCTION forbid_vision_assessment_mutation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'vision_assessments is append-only; updates and deletes are forbidden'
                USING ERRCODE = '55000';
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER vision_assessments_append_only
        BEFORE UPDATE OR DELETE ON vision_assessments
        FOR EACH ROW EXECUTE FUNCTION forbid_vision_assessment_mutation()
    """)


def downgrade():
    op.execute("DROP TRIGGER vision_assessments_append_only ON vision_assessments")
    op.execute("DROP FUNCTION forbid_vision_assessment_mutation()")
    op.drop_index("ix_vision_assessments_claim_latest", table_name="vision_assessments")
    op.drop_table("vision_assessments")
