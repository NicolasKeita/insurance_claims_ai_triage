"""Small collected-evidence contexts for finalizer unit tests."""

from tests.infrastructure.support.agent_fakes import make_evidence


def finalizer_state(*items):
    items = items or (make_evidence(),)
    return {
        "claim_id": "CLAIM-2026-00001",
        "objective": "Review collected evidence",
        "stop_reason": None,
        "evidence_registry": {item.evidence_id: item.model_dump(mode="json") for item in items},
        "observations": [],
    }
