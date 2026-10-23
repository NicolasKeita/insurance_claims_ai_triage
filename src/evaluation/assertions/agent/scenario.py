"""Scenario-level agent action and uncertainty expectations."""

from evaluation.assertions.agent.action_policy import acceptable_actions
from evaluation.assertions.recording import Check

UNCERTAINTY_TOOLS = {"history": "GET_HISTORY"}
UNAVAILABLE_STATUSES = {"NOT_AVAILABLE", "NOT_FOUND"}


def check_agent_scenario(recorder):
    """Their structured cause must be observed, plus explicit uncertainty.

    No exact final-answer wording is required.
    """
    scenario, agent = recorder.scenario, recorder.observation.agent
    recommendation = agent.get("recommendation") or {}
    actions, blocked = acceptable_actions(recorder)
    if actions:
        action = recommendation.get("recommended_next_action")
        recorder.record(
            Check(
                "agent:acceptable_action",
                "agent",
                actions,
                action,
                action in actions,
                blocked=blocked,
            )
        )
    if scenario.expected_uncertainties:
        unavailable = [
            {"tool": item.get("tool_invoked"), "status": item.get("observation_status")}
            for item in agent.get("trace", [])
            if item.get("observation_status") in UNAVAILABLE_STATUSES
        ]
        required_tools = set(scenario.inputs.get("unavailable_tools", []))
        required_tools.update(
            UNCERTAINTY_TOOLS[cause] for cause in scenario.expected_uncertainties if cause in UNCERTAINTY_TOOLS
        )
        observed_tools = {item["tool"] for item in unavailable}
        recorder.record(
            Check(
                "agent:uncertainty",
                "agent",
                {"uncertainty_required": True, "causes": scenario.expected_uncertainties},
                {"uncertainties": recommendation.get("uncertainties", []), "unavailable_tools": unavailable},
                bool(recommendation.get("uncertainties")),
            )
        )
        recorder.record(
            Check(
                "agent:unavailable_data_observed",
                "agent",
                {"statuses": sorted(UNAVAILABLE_STATUSES), "required_tools": sorted(required_tools)},
                unavailable,
                bool(unavailable) and required_tools <= observed_tools,
            )
        )
