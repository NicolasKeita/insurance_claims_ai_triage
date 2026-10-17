"""Scenario-level agent action and uncertainty expectations."""

from evaluation.assertions.recording import Check


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
