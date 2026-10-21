"""Test audit."""

from agent.planner import StructuredAgentFinalizer, StructuredAgentPlanner
from agent.service import AgentServiceConfig, ClaimsAgentService
from tests.agent.graph._constants import CLAIM_ID
from tests.agent.graph._support import service
from tests.support.agent_fakes import (
    FakeRunStore,
    FakeStructuredLlm,
    make_recommendation,
    make_step,
)


def test_two_runs_append_instead_of_replacing():
    """
    A fresh finalizer output is required on each run; no run state leaks.
    """
    store = FakeRunStore()
    agent, _, _ = service(["GET_CLAIM", "FINISH", "GET_CLAIM", "FINISH"], store=store)
    agent.graph = ClaimsAgentService(
        AgentServiceConfig(
            agent.tools,
            StructuredAgentPlanner(
                FakeStructuredLlm(
                    [make_step("GET_CLAIM"), make_step("FINISH"), make_step("GET_CLAIM"), make_step("FINISH")]
                )
            ),
            StructuredAgentFinalizer(FakeStructuredLlm([make_recommendation(), make_recommendation()])),
            run_store=store,
        )
    ).graph
    first, second = agent.review_claim(CLAIM_ID), agent.review_claim(CLAIM_ID)
    assert first.run_id != second.run_id
    assert len(store.results) == 2
    assert first.iteration_count == second.iteration_count == 1
