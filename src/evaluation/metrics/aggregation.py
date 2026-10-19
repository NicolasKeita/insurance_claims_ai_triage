"""Assembly of the benchmark metric document with explicit denominators."""

from evaluation.metrics.checks import (
    agent_metrics,
    fact_summary,
    forbidden_action_count,
    retrieval_metrics,
    signal_metrics,
)
from evaluation.metrics.completion import agent_completion, component_completion, component_success_rates
from evaluation.metrics.latencies import latency_summary
from evaluation.metrics.ratio import ratio

DEFINITIONS = {
    "scenario_pass_rate": "PASS / (PASS + FAIL); SKIP excluded and separately counted; partial scenarios never PASS.",
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
}


def scenario_counts(results):
    return {status: sum(result.status == status for result in results) for status in ("PASS", "FAIL", "SKIP")}


def aggregate_metrics(results):
    """Small runs are diagnostics. Tail summaries require >=20 observations."""
    counts = scenario_counts(results)
    executed = [check for result in results for check in result.assertions if check.status != "SKIP"]
    completion = component_completion(results)
    completion_rate, agents = agent_completion(results)
    return {
        "scenario_counts": counts,
        "scenario_pass_rate": ratio(counts["PASS"], counts["PASS"] + counts["FAIL"]),
        "scenario_pass_rate_denominator": counts["PASS"] + counts["FAIL"],
        "scenario_execution_coverage": ratio(counts["PASS"] + counts["FAIL"], len(results)),
        "component_completion": completion,
        **component_success_rates(completion),
        **fact_summary(executed),
        **signal_metrics(executed),
        **retrieval_metrics(executed),
        **agent_metrics(executed, agents, completion_rate),
        "forbidden_action_count": forbidden_action_count(executed),
        "latencies": latency_summary(results),
        "definitions": DEFINITIONS,
    }
