"""A stored document discrepancy, without external services or model inference."""

import json

import pytest

from agent.tools import ReadOnlyAgentTools
from claims.investigation.models import InvestigationAssessment, InvestigationSignal
from persistence.mappers import AssessmentRecord
from tests.agent.tools._evidence import plan


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


@pytest.fixture
def review_evidence(stored_review):
    claim, factory, _, _ = stored_review
    return ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("GET_INVESTIGATION")).evidence[0]


def finalizer_state(item):
    return {
        "claim_id": "CLAIM-2026-00001",
        "objective": "Review the available evidence",
        "stop_reason": None,
        "evidence_registry": {item.evidence_id: item.model_dump(mode="json")},
        "observations": [],
    }
