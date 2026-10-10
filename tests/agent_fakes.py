from agent.models import (
    AgentPlanStep, AgentRecommendation, EvidenceItem, EvidenceSourceType,
    ObservationStatus, ToolObservation,
)
from persistence.repositories import ClaimNotFoundError


def make_evidence(evidence_id="claim:CLAIM-2026-00001", *, summary="Stored claim facts."):
    return EvidenceItem(evidence_id=evidence_id, source_type=EvidenceSourceType.CLAIM_DATA,
                        source_reference="CLAIM-2026-00001", summary=summary)


def make_recommendation(ids=None):
    ids = ["claim:CLAIM-2026-00001"] if ids is None else ids
    return AgentRecommendation(
        case_summary="Available claim facts require human review.",
        recommended_next_action="REQUEST_ADDITIONAL_INFORMATION",
        rationale=[{"statement": "Review the recorded claim details.", "evidence_ids": ids}] if ids else [],
        evidence_ids=ids, uncertainties=["Repair details require human confirmation."])


def make_step(action, arguments=None):
    return AgentPlanStep(action=action, reason="Collect relevant evidence.", arguments=arguments or {})


class FakeStructuredLlm:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, *, prompt, response_model):
        self.calls.append((prompt, response_model))
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return response_model.model_validate(output.model_dump() if hasattr(output, "model_dump") else output)


class FakeAgentTools:
    def __init__(self, *, observations=None, failure=None, exists=True):
        self.observations = observations or {}
        self.failure = failure
        self.exists = exists
        self.calls = []
        self.preflight = []

    def require_claim(self, claim_id):
        self.preflight.append(claim_id)
        if not self.exists:
            raise ClaimNotFoundError(f"Claim {claim_id} does not exist")

    def execute(self, claim_id, step):
        self.calls.append((claim_id, step.action.value, step.arguments.model_dump()))
        if self.failure:
            raise self.failure
        if step.action.value in self.observations:
            return self.observations[step.action.value]
        evidence = make_evidence(f"{step.action.value.lower()}:{claim_id}")
        if step.action.value == "GET_CLAIM":
            evidence = make_evidence(f"claim:{claim_id}")
        return ToolObservation(tool_name=step.tool_name, status=ObservationStatus.OK,
                               data={"fact": "from stored data"}, evidence=[evidence])


class FakeRunStore:
    def __init__(self):
        self.results = []

    def save(self, result):
        self.results.append(result)
        return result
