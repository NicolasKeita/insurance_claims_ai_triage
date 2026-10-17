"""Agent result, citation and scenario expectation checks."""

from pydantic import ValidationError

from agent.evidence import validate_recommendation
from agent.models import AgentRecommendation, AgentRunResult
from evaluation.assertions.recording import Check


def _valid_schema(model, payload):
    try:
        model.model_validate(payload)
    except (ValidationError, ValueError):
        return False
    return True


def _citations_resolve(agent):
    try:
        parsed = AgentRunResult.model_validate(agent)
        resolved = validate_recommendation(parsed.recommendation, {item.evidence_id: item for item in parsed.evidence})
        return parsed.recommendation.citations == resolved
    except (ValidationError, ValueError, AttributeError):
        return False


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


def check_agent_scenario(recorder):
    """Their structured cause must be observed, plus explicit uncertainty.

    No exact final-answer wording is required.
    """
    scenario, agent = recorder.scenario, recorder.observation.agent
    recommendation = agent.get("recommendation") or {}
    if scenario.acceptable_agent_actions:
        action = recommendation.get("recommended_next_action")
        recorder.record(
            Check(
                "agent:acceptable_action",
                "agent",
                scenario.acceptable_agent_actions,
                action,
                action in scenario.acceptable_agent_actions,
            )
        )
    if scenario.expected_uncertainties:
        unavailable = [item.get("observation_status") for item in agent.get("trace", [])]
        recorder.record(
            Check(
                "agent:uncertainty",
                "agent",
                {"uncertainty_required": True, "causes": scenario.expected_uncertainties},
                {"uncertainties": recommendation.get("uncertainties", []), "tool_statuses": unavailable},
                bool(recommendation.get("uncertainties")),
            )
        )
        recorder.record(
            Check(
                "agent:unavailable_data_observed",
                "agent",
                "NOT_AVAILABLE/NOT_FOUND tool status",
                unavailable,
                any(status in {"NOT_AVAILABLE", "NOT_FOUND"} for status in unavailable),
            )
        )


def check_agent_expectations(recorder):
    """Run every agent check when a structured agent result was observed."""
    if not recorder.observation.agent:
        return
    check_agent_schema(recorder)
    check_agent_citations(recorder)
    check_agent_review(recorder)
    check_agent_scenario(recorder)
