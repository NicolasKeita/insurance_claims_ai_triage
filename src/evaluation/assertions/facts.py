"""Expected and forbidden fact/signal checks over the recorded observation."""

from evaluation.assertions.matching import MISSING, fact_at, matches_fact
from evaluation.assertions.recording import Check

COMPONENT_MAP = {
    "extraction": "document",
    "features": "document",
    "consistency": "consistency",
    "triage": "triage",
    "anomaly": "anomaly",
    "history": "history",
    "vision": "vision",
    "rag": "retrieval",
    "agent": "agent",
    "investigation": "investigation",
}
BLOCKING_COMPONENTS = {"consistency": ["document"], "investigation": ["anomaly", "history"]}
MODEL_SIGNALS = {"ANOMALOUS_PATTERN", "EXTREME_ANOMALY_PERCENTILE"}
HISTORY_SIGNALS = {"RECENT_CLAIM_FREQUENCY", "PREVIOUS_ANOMALOUS_CLAIMS"}


def check_expected_facts(recorder):
    """Verify controlled structured facts, blocking on unavailable upstream data."""
    scenario, observation = recorder.scenario, recorder.observation
    expected_facts = dict(scenario.expected_facts)
    if recorder.mode == "real":
        expected_facts.update(scenario.real_expected_facts)
    for path, expected in expected_facts.items():
        if recorder.mode == "real" and path in scenario.inputs.get("deterministic_only_facts", []):
            continue
        component = COMPONENT_MAP.get(path.split(".")[0], path.split(".")[0])
        if path.startswith("rag.answer.") and "rag_generation" in observation.components:
            component = "rag_generation"
        actual = fact_at(observation.facts, path, MISSING)
        blocked = recorder.skipped(component) or (
            actual is MISSING and any(recorder.skipped(item) for item in BLOCKING_COMPONENTS.get(component, []))
        )
        if (
            recorder.mode == "real"
            and path.startswith("investigation.")
            and scenario.vision_required
            and recorder.skipped("vision")
        ):
            blocked = True
        recorder.record(
            Check(
                f"fact:{path}",
                component,
                expected,
                None if actual is MISSING else actual,
                matches_fact(expected, actual),
                blocked=blocked,
            )
        )


def _model_dependent_skip(recorder, code):
    if recorder.mode != "real":
        return False
    scenario = recorder.scenario
    return code in scenario.inputs.get("deterministic_only_signals", []) or (
        code in MODEL_SIGNALS and not scenario.inputs.get("real_anomaly_ground_truth", False)
    )


def check_expected_signals(recorder):
    observation = recorder.observation
    for code in recorder.scenario.expected_signals:
        if _model_dependent_skip(recorder, code):
            continue
        component = "vision" if code == "VISUAL_ADDITIONAL_DAMAGE" else "investigation"
        upstream_unavailable = code in HISTORY_SIGNALS and recorder.skipped("history")
        present = code in observation.signals
        recorder.record(
            Check(
                f"required_signal:{code}",
                component,
                True,
                present,
                present,
                blocked=recorder.skipped(component) or upstream_unavailable,
            )
        )


def check_forbidden_signals(recorder):
    observation = recorder.observation
    for code in recorder.scenario.forbidden_signals:
        present = code in observation.signals
        recorder.record(
            Check(
                f"forbidden_signal:{code}",
                "investigation",
                False,
                present,
                not present,
                blocked=recorder.skipped("investigation"),
            )
        )
