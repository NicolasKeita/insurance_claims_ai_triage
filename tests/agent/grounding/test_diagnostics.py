"""Failed runs keep precise application-owned causes and discard untrusted content."""

import pytest

from agent.evidence import EvidenceValidationError
from agent.graph._failure import failure
from agent.graph._finalization import validate
from agent.service import AgentRunError
from tests.agent.graph._support import service
from tests.agent.grounding._support import finalizer_state
from tests.infrastructure.support.agent_fakes import FakeRunStore, make_recommendation


@pytest.mark.parametrize("kind", ["unknown", "uncited", "versions"])
def test_grounding_failure_preserves_exact_cause_without_candidate(kind):
    recommendation = make_recommendation()
    expected = {"unknown": "UNKNOWN_EVIDENCE_ID", "uncited": "UNCITED_RATIONALE", "versions": "VERSION_MISMATCH"}[kind]
    if kind == "unknown":
        recommendation.evidence_ids.append("provider-secret-invented-id")
    elif kind == "uncited":
        recommendation.rationale[0].evidence_ids = []
    else:
        recommendation.prompt_version = "provider-secret-wrong-version"
    result = validate({**finalizer_state(), "recommendation": recommendation.model_dump(mode="json")})
    assert result["status"] == "FAILED" and result["recommendation"] is None
    assert f"[{expected}]" in result["error"] and result["failure_kind"] == "GROUNDING"
    assert "provider-secret" not in result["error"]


def test_sanitized_code_and_message_reach_the_saved_agent_run():
    store = FakeRunStore()
    candidate = make_recommendation(["secret-provider-id"])
    agent, _, _ = service(["GET_CLAIM", "FINISH"], candidate, store=store)
    with pytest.raises(AgentRunError) as caught:
        agent.review_claim("CLAIM-2026-00001")
    result = store.results[0]
    assert caught.value.failure_kind == "GROUNDING"
    assert "[UNKNOWN_EVIDENCE_ID]" in result.error
    assert result.recommendation is None and "secret-provider" not in result.model_dump_json()


def test_failure_never_uses_arbitrary_exception_or_diagnostic_text():
    error = EvidenceValidationError("secret provider output", code="secret-injected-code")
    result = failure("GROUNDING", code=error.code)
    assert "[INVALID_EVIDENCE]" in result["error"]
    assert "secret" not in str(result)
    assert "secret" not in str(failure("GROUNDING", code="secret-injected-code"))
