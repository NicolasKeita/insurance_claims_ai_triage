"""Agent result, citation and scenario expectation checks."""

from evaluation.assertions.agent.citations import check_agent_citations, check_agent_review
from evaluation.assertions.agent.scenario import check_agent_scenario
from evaluation.assertions.agent.schema import check_agent_schema

__all__ = ["check_agent_citations", "check_agent_review", "check_agent_schema", "check_agent_scenario"]


def check_agent_expectations(recorder):
    """Run every agent check when a structured agent result was observed."""
    if not recorder.observation.agent:
        return
    check_agent_schema(recorder)
    check_agent_citations(recorder)
    check_agent_review(recorder)
    check_agent_scenario(recorder)
