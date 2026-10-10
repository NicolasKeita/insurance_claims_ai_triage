"""Append-only completed/failed advisory agent audit snapshots.

Revision ID: 0003
Revises: 0002
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("claim_pk", sa.UUID(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("insertion_order", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        sa.Column("objective", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("agent_version", sa.String(), nullable=False),
        sa.Column("prompt_version", sa.String(), nullable=False),
        sa.Column("llm_model", sa.String(), nullable=True),
        sa.Column("iteration_count", sa.Integer(), nullable=False),
        sa.Column("tool_call_count", sa.Integer(), nullable=False),
        sa.Column("tools_used", postgresql.JSONB(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False),
        sa.Column("trace", postgresql.JSONB(), nullable=False),
        sa.Column("recommendation", postgresql.JSONB(none_as_null=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("elapsed_seconds", sa.Float(), nullable=False),
        sa.Column("error", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_agent_runs"),
        sa.ForeignKeyConstraint(["claim_pk"], ["claims.id"], name="fk_agent_runs_claim_pk_claims"),
        sa.UniqueConstraint("run_id", name="uq_agent_runs_run_id"),
        sa.UniqueConstraint("insertion_order", name="uq_agent_runs_insertion_order"),
        sa.CheckConstraint("status IN ('COMPLETED', 'FAILED')", name=op.f("ck_agent_runs_status")),
        sa.CheckConstraint("iteration_count >= 0 AND tool_call_count >= 0",
                           name=op.f("ck_agent_runs_nonnegative_counts")),
        sa.CheckConstraint("elapsed_seconds >= 0", name=op.f("ck_agent_runs_nonnegative_elapsed")),
        sa.CheckConstraint("completed_at >= created_at", name=op.f("ck_agent_runs_completion_time")),
        sa.CheckConstraint(
            "(status = 'COMPLETED' AND recommendation IS NOT NULL AND error IS NULL) OR "
            "(status = 'FAILED' AND recommendation IS NULL AND error IS NOT NULL)",
            name=op.f("ck_agent_runs_terminal_result"),
        ),
    )
    op.create_index("ix_agent_runs_claim_latest", "agent_runs",
                    ["claim_pk", "created_at", "insertion_order"])
    op.execute("""
        CREATE FUNCTION forbid_agent_run_mutation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'agent_runs is append-only; updates and deletes are forbidden'
                USING ERRCODE = '55000';
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER agent_runs_append_only
        BEFORE UPDATE OR DELETE ON agent_runs
        FOR EACH ROW EXECUTE FUNCTION forbid_agent_run_mutation()
    """)


def downgrade():
    op.execute("DROP TRIGGER agent_runs_append_only ON agent_runs")
    op.execute("DROP FUNCTION forbid_agent_run_mutation()")
    op.drop_index("ix_agent_runs_claim_latest", table_name="agent_runs")
    op.drop_table("agent_runs")
