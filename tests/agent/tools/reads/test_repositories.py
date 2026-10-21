"""Test repositories."""

import pytest

from agent.models import (
    ObservationStatus,
)
from agent.tools import ReadOnlyAgentTools
from persistence.repositories import ClaimNotFoundError
from tests.agent.tools._constants import ASSESSMENT_ID
from tests.agent.tools._evidence import plan
from tests.agent.tools._repositories import configured
from tests.agent.tools._repositories import configured as configured


@pytest.mark.parametrize(
    "action,prefix,source",
    [
        ("GET_CLAIM", "claim", "CLAIM_DATA"),
        ("GET_HISTORY", "history", "HISTORY"),
        ("GET_TRIAGE", "triage", "TRIAGE_ASSESSMENT"),
        ("GET_ANOMALY", "anomaly", "ANOMALY_ASSESSMENT"),
        ("GET_INVESTIGATION", "investigation", "INVESTIGATION_ASSESSMENT"),
    ],
)
def test_read_adapters_use_existing_repositories_and_stable_ids(configured, action, prefix, source):
    claim, factory, _, _ = configured
    tools = ReadOnlyAgentTools(factory)
    first = tools.execute(claim.claim_id, plan(action))
    second = tools.execute(claim.claim_id, plan(action))
    suffix = claim.claim_id if prefix in {"claim", "history"} else str(ASSESSMENT_ID)
    assert first.evidence[0].evidence_id == second.evidence[0].evidence_id == f"{prefix}:{suffix}"
    assert first.evidence[0].source_type == source
    assert first.status == ObservationStatus.OK
    assert factory.open_count == 0


def test_claim_evidence_omits_filenames_and_has_aggregated_presence(configured):
    claim, factory, _, _ = configured
    observation = ReadOnlyAgentTools(factory).execute(claim.claim_id, plan("GET_CLAIM"))
    text = observation.model_dump_json()
    assert "filename" not in text
    assert observation.data["document_count"] == len(claim.documents)
    assert observation.data["image_count"] == len(claim.images)
    assert any(d["available"] is False for d in observation.data["documents"])


@pytest.mark.parametrize(
    "action,kind", [("GET_TRIAGE", "triage"), ("GET_ANOMALY", "anomaly"), ("GET_INVESTIGATION", "investigation")]
)
def test_missing_assessment_returns_unavailable_without_computing(configured, action, kind):
    claim, factory, _, records = configured
    records[kind] = None
    observation = ReadOnlyAgentTools(factory).execute(claim.claim_id, plan(action))
    assert observation.status == ObservationStatus.NOT_AVAILABLE
    assert observation.evidence == []


def test_unknown_claim_preflight_and_retrieval_never_loads(configured):
    _, factory, _, _ = configured

    def forbidden():
        raise AssertionError("External service must not load for an unknown claim")

    tools = ReadOnlyAgentTools(factory, similar_service_loader=forbidden, knowledge_retriever_loader=forbidden)
    with pytest.raises(ClaimNotFoundError):
        tools.require_claim("CLAIM-SECRET-123")
    with pytest.raises(ClaimNotFoundError):
        tools.execute("CLAIM-SECRET-123", plan("SEARCH_POLICY", query="coverage"))
    with pytest.raises(ClaimNotFoundError):
        tools.execute("CLAIM-SECRET-123", plan("FIND_SIMILAR_CLAIMS"))
    assert factory.open_count == 0
