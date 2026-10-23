"""Small structured observations for the action assertion alone."""

from evaluation.assertions.agent.scenario import check_agent_scenario
from evaluation.assertions.recording import AssertionRecorder
from evaluation.models import ComponentResult, EvaluationScenario, PipelineObservation

FOLLOW_UP_ACTIONS = ["REQUEST_ADDITIONAL_INFORMATION", "REFER_TO_EXPERT", "REFER_FOR_INVESTIGATION_REVIEW"]
STANDARD = "CONTINUE_STANDARD_REVIEW"


def recorder(*, action=STANDARD, workflow="STANDARD", anomalous=False, signals=(), mode="real"):
    scenario = EvaluationScenario(
        scenario_id="action_contract",
        description="An isolated action assertion, without executing a scenario.",
        claim_source="unused",
        acceptable_agent_actions=FOLLOW_UP_ACTIONS,
        real_agent_action_policy={"otherwise_acceptable_actions": [STANDARD, *FOLLOW_UP_ACTIONS]},
    )
    observation = PipelineObservation(
        facts={"triage": {"recommended_workflow": workflow}, "anomaly": {"is_anomalous": anomalous}},
        investigation={"review_priority": "NORMAL", "signals": [{"code": code} for code in signals]},
        agent={"recommendation": {"recommended_next_action": action}},
        components={name: ComponentResult(status="PASS") for name in ("triage", "anomaly", "investigation")},
    )
    return AssertionRecorder(scenario, observation, mode)


def action_check(value):
    check_agent_scenario(value)
    return next(item for item in reversed(value.checks) if item.name == "agent:acceptable_action")
