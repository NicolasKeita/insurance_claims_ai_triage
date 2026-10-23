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


@pytest.fixture(
    params=[
        ("investigation", "ADDITIONAL_QUOTE_DAMAGE"),
        ("investigation", "UNMAPPED_QUOTE_ITEM"),
        ("investigation", "RECENT_CLAIM_FREQUENCY"),
        ("investigation", "ANOMALOUS_PATTERN"),
        ("investigation", "EXTREME_ANOMALY_PERCENTILE"),
        ("triage", "EXPERT_REVIEW"),
        ("triage", "INVESTIGATION"),
        ("anomaly", True),
    ],
    ids=lambda value: f"{value[0]}-{value[1]}",
)
def follow_up_evidence(stored_review, request):
    claim, factory, _, records = stored_review
    kind, trigger = request.param
    if kind != "investigation":
        item = ReadOnlyAgentTools(factory).execute(claim.claim_id, plan(f"GET_{kind.upper()}")).evidence[0]
        if kind == "triage":
            item.structured_data["result"]["recommended_workflow"] = trigger
        return item
    previous = records["investigation"]
    assessment = previous.result.model_dump()
    assessment["signals"][0]["code"] = trigger
    if trigger == "RECENT_CLAIM_FREQUENCY":
        assessment.update(review_score=15, review_priority="NORMAL", review_recommended=False)
        assessment["signals"][0].update(
            category="HISTORICAL_CONTEXT",
            title="Recent claim frequency",
            explanation="Three previous claims occurred in the last 30 days and five in the last 365 days.",
            evidence={"previous_claim_count": 6, "claims_last_30_days": 3, "claims_last_365_days": 5},
        )
    records["investigation"] = AssessmentRecord(
        previous.id, previous.created_at, InvestigationAssessment.model_validate(assessment)
    )
    return ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("GET_INVESTIGATION")).evidence[0]


def finalizer_state(item):
    return {
        "claim_id": "CLAIM-2026-00001",
        "objective": "Review the available evidence",
        "stop_reason": None,
        "evidence_registry": {item.evidence_id: item.model_dump(mode="json")},
        "observations": [],
    }
