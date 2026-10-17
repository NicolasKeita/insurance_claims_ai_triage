"""Agent recommendation, status and evidence-registry schema checks."""

from pydantic import ValidationError

from agent.models import AgentRecommendation
from evaluation.assertions.recording import Check


def _valid_schema(model, payload):
    try:
        model.model_validate(payload)
    except (ValidationError, ValueError):
        return False
    return True


def check_agent_schema(recorder):
    """Validate the final recommendation, status and evidence registry."""
    agent = recorder.observation.agent
    recommendation = agent.get("recommendation") or {}
    valid = _valid_schema(AgentRecommendation, recommendation)
    recorder.record(Check("agent:final_schema", "agent", "AgentRecommendation", valid, valid))
    recorder.record(
        Check("agent:completed", "agent", "COMPLETED", agent.get("status"), agent.get("status") == "COMPLETED")
    )
    evidence = agent.get("evidence", [])
    registry = {item.get("evidence_id") for item in evidence}
    recorder.record(
        Check(
            "agent:unique_evidence_ids", "agent", "unique registry IDs", len(evidence), len(registry) == len(evidence)
        )
    )
