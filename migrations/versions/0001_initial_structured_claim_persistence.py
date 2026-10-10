"""Initial structured claim persistence

Revision ID: 0001
Revises: none
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():

    op.create_table('customers',
    sa.Column('customer_id', sa.String(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_customers')),
    sa.UniqueConstraint('customer_id', name=op.f('uq_customers_customer_id'))
    )
    op.create_table('policies',
    sa.Column('policy_id', sa.String(), nullable=False),
    sa.Column('product', sa.String(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_policies')),
    sa.UniqueConstraint('policy_id', name=op.f('uq_policies_policy_id'))
    )
    op.create_table('claims',
    sa.Column('claim_id', sa.String(), nullable=False),
    sa.Column('customer_pk', sa.Uuid(), nullable=False),
    sa.Column('policy_pk', sa.Uuid(), nullable=False),
    sa.Column('claim_type', sa.Enum('AUTO_COLLISION', name='claim_type', native_enum=False, create_constraint=False), nullable=False),
    sa.Column('status', sa.Enum('NEW', name='claim_status', native_enum=False, create_constraint=False), nullable=False),
    sa.Column('incident_date', sa.Date(), nullable=False),
    sa.Column('incident_location', sa.String(), nullable=False),
    sa.Column('collision_type', sa.Enum('FRONT_COLLISION', 'REAR_COLLISION', 'SIDE_COLLISION', 'PARKING_DAMAGE', name='collision_type', native_enum=False, create_constraint=False), nullable=False),
    sa.Column('injuries_declared', sa.Boolean(), nullable=False),
    sa.Column('vehicle_make', sa.String(), nullable=False),
    sa.Column('vehicle_model', sa.String(), nullable=False),
    sa.Column('vehicle_year', sa.Integer(), nullable=False),
    sa.Column('repair_estimate_amount', sa.Numeric(), nullable=False),
    sa.Column('repair_estimate_currency', sa.String(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("claim_type IN ('AUTO_COLLISION')", name=op.f('ck_claims_claim_type')),
    sa.CheckConstraint("collision_type IN ('FRONT_COLLISION', 'REAR_COLLISION', 'SIDE_COLLISION', 'PARKING_DAMAGE')", name=op.f('ck_claims_collision_type')),
    sa.CheckConstraint("repair_estimate_currency ~ '^[A-Z]{3}$'", name=op.f('ck_claims_currency')),
    sa.CheckConstraint("status IN ('NEW')", name=op.f('ck_claims_claim_status')),
    sa.CheckConstraint('repair_estimate_amount >= 0', name=op.f('ck_claims_nonnegative_amount')),
    sa.CheckConstraint('vehicle_year >= 1886', name=op.f('ck_claims_vehicle_year')),
    sa.ForeignKeyConstraint(['customer_pk'], ['customers.id'], name=op.f('fk_claims_customer_pk_customers')),
    sa.ForeignKeyConstraint(['policy_pk'], ['policies.id'], name=op.f('fk_claims_policy_pk_policies')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_claims')),
    sa.UniqueConstraint('claim_id', name=op.f('uq_claims_claim_id'))
    )
    op.create_index(op.f('ix_claims_collision_type'), 'claims', ['collision_type'], unique=False)
    op.create_index('ix_claims_customer_date', 'claims', ['customer_pk', 'incident_date'], unique=False)
    op.create_index(op.f('ix_claims_incident_date'), 'claims', ['incident_date'], unique=False)
    op.create_index(op.f('ix_claims_policy_pk'), 'claims', ['policy_pk'], unique=False)
    op.create_table('anomaly_assessments',
    sa.Column('is_anomalous', sa.Boolean(), nullable=False),
    sa.Column('anomaly_score', sa.Double(), nullable=False),
    sa.Column('threshold', sa.Double(), nullable=False),
    sa.Column('reference_percentile', sa.Double(), nullable=True),
    sa.Column('unusual_features', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('claim_pk', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('registered_model', sa.String(), nullable=False),
    sa.Column('model_version', sa.String(), nullable=False),
    sa.Column('model_alias', sa.String(), nullable=False),
    sa.CheckConstraint('reference_percentile BETWEEN 0 AND 1', name=op.f('ck_anomaly_assessments_percentile')),
    sa.ForeignKeyConstraint(['claim_pk'], ['claims.id'], name=op.f('fk_anomaly_assessments_claim_pk_claims')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_anomaly_assessments'))
    )
    op.create_index('ix_anomaly_assessments_claim_date', 'anomaly_assessments', ['claim_pk', 'created_at', 'id'], unique=False)
    op.create_table('claim_declared_damages',
    sa.Column('claim_pk', sa.Uuid(), nullable=False),
    sa.Column('damage_type', sa.Enum('FRONT_BUMPER', 'LEFT_HEADLIGHT', 'HOOD', name='damage_type', native_enum=False, create_constraint=False), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.CheckConstraint("damage_type IN ('FRONT_BUMPER', 'LEFT_HEADLIGHT', 'HOOD')", name=op.f('ck_claim_declared_damages_damage_type')),
    sa.ForeignKeyConstraint(['claim_pk'], ['claims.id'], name=op.f('fk_claim_declared_damages_claim_pk_claims'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_claim_declared_damages')),
    sa.UniqueConstraint('claim_pk', 'damage_type', name=op.f('uq_claim_declared_damages_claim_pk'))
    )
    op.create_table('claim_documents',
    sa.Column('claim_pk', sa.Uuid(), nullable=False),
    sa.Column('document_type', sa.Enum('CLAIM_FORM', 'ACCIDENT_REPORT', 'REPAIR_QUOTE', 'POLICE_REPORT', name='document_type', native_enum=False, create_constraint=False), nullable=False),
    sa.Column('filename', sa.String(), nullable=True),
    sa.Column('available', sa.Boolean(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.CheckConstraint("document_type IN ('CLAIM_FORM', 'ACCIDENT_REPORT', 'REPAIR_QUOTE', 'POLICE_REPORT')", name=op.f('ck_claim_documents_document_type')),
    sa.CheckConstraint('NOT available OR filename IS NOT NULL', name=op.f('ck_claim_documents_available_filename')),
    sa.ForeignKeyConstraint(['claim_pk'], ['claims.id'], name=op.f('fk_claim_documents_claim_pk_claims'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_claim_documents'))
    )
    op.create_index('ix_claim_documents_claim_pk', 'claim_documents', ['claim_pk'], unique=False)
    op.create_table('claim_images',
    sa.Column('claim_pk', sa.Uuid(), nullable=False),
    sa.Column('filename', sa.String(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.ForeignKeyConstraint(['claim_pk'], ['claims.id'], name=op.f('fk_claim_images_claim_pk_claims'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_claim_images'))
    )
    op.create_index(op.f('ix_claim_images_claim_pk'), 'claim_images', ['claim_pk'], unique=False)
    op.create_table('investigation_assessments',
    sa.Column('review_score', sa.Integer(), nullable=False),
    sa.Column('review_priority', sa.Enum('NORMAL', 'ELEVATED', 'HIGH', name='review_priority', native_enum=False, create_constraint=False), nullable=False),
    sa.Column('review_recommended', sa.Boolean(), nullable=False),
    sa.Column('policy_version', sa.String(), nullable=False),
    sa.Column('signals', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('claim_pk', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint("review_priority IN ('NORMAL', 'ELEVATED', 'HIGH')", name=op.f('ck_investigation_assessments_review_priority')),
    sa.CheckConstraint('review_score BETWEEN 0 AND 100', name=op.f('ck_investigation_assessments_review_score')),
    sa.ForeignKeyConstraint(['claim_pk'], ['claims.id'], name=op.f('fk_investigation_assessments_claim_pk_claims')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_investigation_assessments'))
    )
    op.create_index('ix_investigation_assessments_claim_date', 'investigation_assessments', ['claim_pk', 'created_at', 'id'], unique=False)
    op.create_table('triage_assessments',
    sa.Column('recommended_workflow', sa.Enum('FAST_TRACK', 'STANDARD', 'EXPERT_REVIEW', 'INVESTIGATION', name='triage_workflow', native_enum=False, create_constraint=False), nullable=False),
    sa.Column('confidence', sa.Double(), nullable=False),
    sa.Column('probabilities', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('claim_pk', sa.Uuid(), nullable=False),
    sa.Column('id', sa.Uuid(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('registered_model', sa.String(), nullable=False),
    sa.Column('model_version', sa.String(), nullable=False),
    sa.Column('model_alias', sa.String(), nullable=False),
    sa.CheckConstraint("recommended_workflow IN ('FAST_TRACK', 'STANDARD', 'EXPERT_REVIEW', 'INVESTIGATION')", name=op.f('ck_triage_assessments_triage_workflow')),
    sa.CheckConstraint('confidence BETWEEN 0 AND 1', name=op.f('ck_triage_assessments_confidence')),
    sa.ForeignKeyConstraint(['claim_pk'], ['claims.id'], name=op.f('fk_triage_assessments_claim_pk_claims')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_triage_assessments'))
    )
    op.create_index('ix_triage_assessments_claim_date', 'triage_assessments', ['claim_pk', 'created_at', 'id'], unique=False)



def downgrade():

    op.drop_index('ix_triage_assessments_claim_date', table_name='triage_assessments')
    op.drop_table('triage_assessments')
    op.drop_index('ix_investigation_assessments_claim_date', table_name='investigation_assessments')
    op.drop_table('investigation_assessments')
    op.drop_index(op.f('ix_claim_images_claim_pk'), table_name='claim_images')
    op.drop_table('claim_images')
    op.drop_index('ix_claim_documents_claim_pk', table_name='claim_documents')
    op.drop_table('claim_documents')
    op.drop_table('claim_declared_damages')
    op.drop_index('ix_anomaly_assessments_claim_date', table_name='anomaly_assessments')
    op.drop_table('anomaly_assessments')
    op.drop_index(op.f('ix_claims_policy_pk'), table_name='claims')
    op.drop_index(op.f('ix_claims_incident_date'), table_name='claims')
    op.drop_index('ix_claims_customer_date', table_name='claims')
    op.drop_index(op.f('ix_claims_collision_type'), table_name='claims')
    op.drop_table('claims')
    op.drop_table('policies')
    op.drop_table('customers')


