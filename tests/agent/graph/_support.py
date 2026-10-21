"""Support."""

from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
from agent.service import AgentServiceConfig, ClaimsAgentService
from tests.infrastructure.support.agent_fakes import (
    FakeAgentTools,
    FakeStructuredLlm,
    make_recommendation,
    make_step,
)


def service(steps, recommendation=None, tools=None, store=None, max_iterations=8):
    planner_llm = FakeStructuredLlm([make_step(s) if isinstance(s, str) else s for s in steps])
    finalizer_llm = FakeStructuredLlm([recommendation or make_recommendation()])
    value = ClaimsAgentService(
        AgentServiceConfig(
            tools or FakeAgentTools(),
            StructuredAgentPlanner(planner_llm),
            StructuredAgentFinalizer(finalizer_llm),
            run_store=store,
            max_iterations=max_iterations,
            llm_model="fake-local",
        )
    )
    return value, planner_llm, finalizer_llm
