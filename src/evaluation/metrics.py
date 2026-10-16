"""Small benchmark metrics with explicit denominators and missingness."""

from statistics import mean, median


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def aggregate_metrics(results):
    """
    Small runs are diagnostics. Tail summaries require >=20 observations.
    """
    counts = {status: sum(r.status == status for r in results) for status in ("PASS", "FAIL", "SKIP")}
    assertions = [a for result in results for a in result.assertions]
    executed = [a for a in assertions if a.status != "SKIP"]

    def group(prefix):
        return [a for a in executed if a.name.startswith(prefix)]

    def rate(prefix):
        values = group(prefix)
        return _ratio(sum(a.status == "PASS" for a in values), len(values))

    facts = group("fact:")
    unexpected = 0
    for assertion in facts:
        expected = assertion.expected
        expected_set = expected.get("$set_equals") if isinstance(expected, dict) else expected
        if isinstance(expected_set, list) and isinstance(assertion.actual, list):
            unexpected += sum(item not in expected_set for item in assertion.actual)
    completion = {}
    for name in ("document", "vision", "retrieval", "agent", "triage", "anomaly", "history", "similar_claims"):
        values = [
            r.component_results[name]
            for r in results
            if name in r.component_results and r.component_results[name].metadata.get("applicable", True)
        ]
        passed, failed, skipped = (sum(v.status == s for v in values) for s in ("PASS", "FAIL", "SKIP"))
        completion[name] = {
            "success_rate": _ratio(passed, passed + failed),
            "requested_completion_rate": _ratio(passed, len(values)),
            "completed": passed,
            "failed": failed,
            "skipped": skipped,
            "attempted": passed + failed,
        }

    timings = {}
    for name in ("document", "vision", "triage", "anomaly", "ml", "retrieval", "rag_generation", "agent", "total"):
        samples = [
            r.timings[name]
            for r in results
            if name in r.timings
            and (
                name in {"ml", "total"} or r.component_results.get(name) and r.component_results[name].status != "SKIP"
            )
        ]
        timings[name] = {
            "count": len(samples),
            "individual_seconds": samples,
            "mean_seconds": mean(samples) if samples else None,
        }
        if len(samples) >= 20:
            ordered = sorted(samples)
            timings[name].update(
                median_seconds=median(samples),
                p95_seconds=ordered[max(0, __import__("math").ceil(0.95 * len(samples)) - 1)],
            )
    agents = [r.observation.agent for r in results if r.observation.agent]
    agent_attempts = [
        r
        for r in results
        if r.component_results.get("agent")
        and r.component_results["agent"].status != "SKIP"
        and r.component_results["agent"].metadata.get("applicable", True)
    ]
    completion_denominator = len(agent_attempts) if agent_attempts else len(agents)
    completed_agents = (
        sum(
            bool(r.observation.agent)
            and r.observation.agent.get("status") == "COMPLETED"
            and r.component_results["agent"].status == "PASS"
            for r in agent_attempts
        )
        if agent_attempts
        else sum(a.get("status") == "COMPLETED" for a in agents)
    )
    return {
        "scenario_counts": counts,
        "scenario_pass_rate": _ratio(counts["PASS"], counts["PASS"] + counts["FAIL"]),
        "scenario_pass_rate_denominator": counts["PASS"] + counts["FAIL"],
        "scenario_execution_coverage": _ratio(counts["PASS"] + counts["FAIL"], len(results)),
        "component_completion": completion,
        **{
            f"{name}_success_rate": completion[name]["success_rate"]
            for name in ("document", "vision", "retrieval", "agent")
        },
        "expected_fact_recall": _ratio(sum(a.status == "PASS" for a in facts), len(facts)),
        "expected_fact_count": len(facts),
        "unexpected_fact_count": unexpected,
        "required_signal_recall": rate("required_signal:"),
        "required_signal_count": len(group("required_signal:")),
        "forbidden_signal_count": sum(a.status == "FAIL" for a in group("forbidden_signal:")),
        "expected_source_retrieved_rate": rate("rag:expected_sources"),
        "citation_validity": rate("rag:citation_validity"),
        "agent_completion_rate": _ratio(completed_agents, completion_denominator),
        "agent_valid_final_schema_rate": rate("agent:final_schema"),
        "agent_valid_evidence_citations_rate": rate("agent:evidence_citations"),
        "agent_human_review_correctness": rate("agent:human_review_required"),
        "agent_valid_action_rate": rate("agent:acceptable_action"),
        "agent_tool_call_count": sum(a.get("tool_call_count", 0) for a in agents),
        "agent_duplicate_tool_calls_blocked": sum(
            sum(step.get("observation_status") == "DUPLICATE_BLOCKED" for step in a.get("trace", [])) for a in agents
        ),
        "forbidden_action_count": sum(len(a.actual or []) for a in group("safety:forbidden_actions")),
        "latencies": timings,
        "definitions": {
            "scenario_pass_rate": "PASS / (PASS + FAIL); SKIP excluded and separately counted; "
            "partial scenarios never PASS.",
            "component_success_rate": "Technical PASS / (PASS + FAIL) among attempted applicable components; "
            "requested_completion_rate includes unavailable requests.",
            "expected_fact_recall": "Matched controlled structured fact assertions / "
            "executed controlled fact assertions; "
            "no unlabelled ML accuracy.",
            "unexpected_fact_count": "Extra members only for exact expected list/set assertions; "
            "not arbitrary scalar model predictions.",
            "required_signal_recall": "Present required signals / executed required signal expectations.",
            "citation_validity": "Scenarios with valid retrieved citation IDs AND source/version/section metadata / "
            "scenarios with checked generated answers.",
        },
    }
