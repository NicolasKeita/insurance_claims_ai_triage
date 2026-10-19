"""Per-component availability accounting; unavailable requests stay visible."""

from evaluation.metrics.ratio import ratio

COMPONENTS = ("document", "vision", "retrieval", "agent", "triage", "anomaly", "history", "similar_claims")
RATE_COMPONENTS = ("document", "vision", "retrieval", "agent")


def component_completion(results, names=COMPONENTS):
    """Distinguish technical success from requested completion per component."""
    completion = {}
    for name in names:
        values = [
            result.component_results[name]
            for result in results
            if name in result.component_results and result.component_results[name].metadata.get("applicable", True)
        ]
        passed, failed, skipped = (sum(item.status == status for item in values) for status in ("PASS", "FAIL", "SKIP"))
        completion[name] = {
            "success_rate": ratio(passed, passed + failed),
            "requested_completion_rate": ratio(passed, len(values)),
            "completed": passed,
            "failed": failed,
            "skipped": skipped,
            "attempted": passed + failed,
        }
    return completion


def component_success_rates(completion, names=RATE_COMPONENTS):
    return {f"{name}_success_rate": completion[name]["success_rate"] for name in names}


def agent_completion(results):
    """Return the completion rate and the observed agents used by other metrics."""
    agents = [result.observation.agent for result in results if result.observation.agent]
    attempts = [
        result
        for result in results
        if result.component_results.get("agent")
        and result.component_results["agent"].status != "SKIP"
        and result.component_results["agent"].metadata.get("applicable", True)
    ]
    denominator = len(attempts) if attempts else len(agents)
    if attempts:
        completed = sum(
            bool(result.observation.agent)
            and result.observation.agent.get("status") == "COMPLETED"
            and result.component_results["agent"].status == "PASS"
            for result in attempts
        )
    else:
        completed = sum(agent.get("status") == "COMPLETED" for agent in agents)
    return ratio(completed, denominator), agents
