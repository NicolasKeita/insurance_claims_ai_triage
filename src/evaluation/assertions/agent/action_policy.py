"""Evaluate action expectations from observed outputs, independently of agent enforcement."""

from claims.enums import DamageType, TriageWorkflow
from claims.investigation.models import SignalCode

FOLLOW_UP_ACTIONS = ["REQUEST_ADDITIONAL_INFORMATION", "REFER_TO_EXPERT", "REFER_FOR_INVESTIGATION_REVIEW"]
FOLLOW_UP_WORKFLOWS = {TriageWorkflow.EXPERT_REVIEW, TriageWorkflow.INVESTIGATION}
FOLLOW_UP_SIGNALS = {
    SignalCode.ADDITIONAL_QUOTE_DAMAGE,
    SignalCode.UNMAPPED_QUOTE_ITEM,
    SignalCode.RECENT_CLAIM_FREQUENCY,
    SignalCode.ANOMALOUS_PATTERN,
    SignalCode.EXTREME_ANOMALY_PERCENTILE,
    SignalCode.VISUAL_ADDITIONAL_DAMAGE,
}


def _available(observation, component):
    result = observation.components.get(component)
    return result is not None and result.status == "PASS"


def _payload(value):
    return value if isinstance(value, dict) else {}


def _investigation_complete(observation, vision_required):
    result = observation.components.get("investigation")
    if result is None:
        return False
    missing = result.metadata.get("unavailable_inputs", [])
    if result.metadata.get("partial") and not missing:
        return False
    optional = set() if vision_required else {"vision"}
    return isinstance(missing, list) and not (set(missing) - optional)


def _observed_follow_up(observation, vision_required):
    """One positive input is sufficient; missing inputs cannot establish a negative."""
    triage = _payload(observation.facts.get("triage"))
    anomaly = _payload(observation.facts.get("anomaly"))
    investigation = _payload(observation.investigation)
    visual = _payload(_payload(observation.vision).get("consistency"))
    additional_damage = visual.get("additional_visual_damage")
    workflow = triage.get("recommended_workflow")
    flagged = anomaly.get("is_anomalous")
    signals = investigation.get("signals")
    valid_signals = isinstance(signals, list) and all(
        isinstance(item, dict) and item.get("code") in tuple(SignalCode) for item in signals
    )
    available = (
        _available(observation, "triage") and workflow in tuple(TriageWorkflow),
        _available(observation, "anomaly") and type(flagged) is bool,
        _available(observation, "investigation") and valid_signals,
    )
    positive = (
        available[0] and workflow in FOLLOW_UP_WORKFLOWS,
        available[1] and flagged is True,
        available[2] and any(item.get("code") in FOLLOW_UP_SIGNALS for item in signals),
        _available(observation, "vision")
        and isinstance(additional_damage, list)
        and any(damage in tuple(DamageType) for damage in additional_damage),
    )
    return any(positive), all(available) and _investigation_complete(observation, vision_required)


def acceptable_actions(recorder):
    """Keep deterministic expectations strict; real expectations follow measured inputs."""
    scenario = recorder.scenario
    policy = scenario.real_agent_action_policy
    if recorder.mode != "real" or policy is None:
        return scenario.acceptable_agent_actions, False
    follow_up, complete = _observed_follow_up(recorder.observation, scenario.vision_required)
    if follow_up:
        return FOLLOW_UP_ACTIONS, False
    return [action.value for action in policy.otherwise_acceptable_actions], not complete
