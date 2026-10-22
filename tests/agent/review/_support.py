"""A stored document discrepancy, without external services or model inference."""

import json

import pytest

from claims.investigation.models import InvestigationAssessment, InvestigationSignal
from persistence.mappers import AssessmentRecord


@pytest.fixture
def stored_review(configured):
    claim, factory, calls, records = configured
    previous = records["investigation"]
    assessment = InvestigationAssessment(
        review_score=18,
        review_priority="NORMAL",
        review_recommended=False,
        policy_version="investigation_policy_v2",
        signals=(
            InvestigationSignal(
                code="ADDITIONAL_QUOTE_DAMAGE",
                category="DOCUMENT_CONSISTENCY",
                severity="MEDIUM",
                title="Additional damage in repair quote",
                explanation="The repair quote adds damage absent from the claim form.",
                evidence={"damage_types": ["HOOD"], "count": 1},
            ),
        ),
    )
    records["investigation"] = AssessmentRecord(previous.id, previous.created_at, assessment)
    return claim, factory, calls, records


def prompt_data(call):
    return json.loads(call[0].split("UNTRUSTED TASK AND TOOL EVIDENCE (JSON):\n")[1])
