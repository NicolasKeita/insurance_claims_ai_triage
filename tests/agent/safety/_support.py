"""Support."""

from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
from agent.service import AgentServiceConfig, ClaimsAgentService
from tests.support.agent_fakes import (
    FakeAgentTools,
    FakeStructuredLlm,
    make_recommendation,
    make_step,
)


def make_service(steps, *, tools=None, recommendation=None, store=None, max_iterations=8):
    return ClaimsAgentService(
        AgentServiceConfig(
            tools or FakeAgentTools(),
            StructuredAgentPlanner(FakeStructuredLlm([make_step(step) for step in steps])),
            StructuredAgentFinalizer(FakeStructuredLlm([recommendation or make_recommendation()])),
            run_store=store,
            max_iterations=max_iterations,
        )
    )
