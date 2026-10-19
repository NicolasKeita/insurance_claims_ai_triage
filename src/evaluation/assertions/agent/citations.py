"""Agent citation resolution, grounding and human-review checks."""

from pydantic import ValidationError

from agent.evidence import validate_recommendation
from agent.models import AgentRunResult
from evaluation.assertions.recording import Check


def _citations_resolve(agent):
    try:
        parsed = AgentRunResult.model_validate(agent)
        resolved = validate_recommendation(parsed.recommendation, {item.evidence_id: item for item in parsed.evidence})
        return parsed.recommendation.citations == resolved
    except (ValidationError, ValueError, AttributeError):
        return False


def check_agent_citations(recorder):
    """Every rationale must cite collected evidence IDs only."""
    agent = recorder.observation.agent
    recommendation = agent.get("recommendation") or {}
    registry = {item.get("evidence_id") for item in agent.get("evidence", [])}
    cited = set(recommendation.get("evidence_ids", []))
    for rationale in recommendation.get("rationale", []):
        cited.update(rationale.get("evidence_ids", []))
    unknown = sorted(cited - registry)
    ungrounded = [
        index
        for index, item in enumerate(recommendation.get("rationale", []))
        if registry and not item.get("evidence_ids")
    ]
    recorder.record(
        Check(
            "agent:evidence_citations",
            "agent",
            "all rationale IDs collected; each rationale grounded",
            {"unknown": unknown, "ungrounded": ungrounded},
            bool(recommendation) and not unknown and not ungrounded,
        )
    )
    rationale_ids = {eid for item in recommendation.get("rationale", []) for eid in item.get("evidence_ids", [])}
    declared_ids = set(recommendation.get("evidence_ids", []))
    recorder.record(
        Check(
            "agent:evidence_union",
            "agent",
            sorted(rationale_ids),
            sorted(declared_ids),
            rationale_ids == declared_ids,
        )
    )


def check_agent_review(recorder):
    """Human review must remain required and knowledge citations resolved."""
    recommendation = (recorder.observation.agent.get("recommendation")) or {}
    recorder.record(
        Check(
            "agent:human_review_required",
            "agent",
            True,
            recommendation.get("human_review_required"),
            recommendation.get("human_review_required") is True,
        )
    )
    citations_match = _citations_resolve(recorder.observation.agent)
    recorder.record(
        Check(
            "agent:knowledge_citations",
            "agent",
            "authoritative metadata resolved from collected evidence",
            citations_match,
            citations_match,
        )
    )
