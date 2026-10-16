"""Assertion-derived rates, fact summaries and safety counters."""

from evaluation.metrics.ratio import ratio


def group(assertions, prefix):
    """Select executed assertions whose name starts with ``prefix``."""
    return [assertion for assertion in assertions if assertion.name.startswith(prefix)]


def pass_rate(assertions, prefix):
    values = group(assertions, prefix)
    return ratio(sum(item.status == "PASS" for item in values), len(values))


def fact_summary(assertions):
    """Recall over controlled structured facts, plus exact-list extras only."""
    facts = group(assertions, "fact:")
    unexpected = 0
    for assertion in facts:
        expected = assertion.expected
        expected_set = expected.get("$set_equals") if isinstance(expected, dict) else expected
        if isinstance(expected_set, list) and isinstance(assertion.actual, list):
            unexpected += sum(item not in expected_set for item in assertion.actual)
    return {
        "expected_fact_recall": ratio(sum(item.status == "PASS" for item in facts), len(facts)),
        "expected_fact_count": len(facts),
        "unexpected_fact_count": unexpected,
    }


def signal_metrics(assertions):
    return {
        "required_signal_recall": pass_rate(assertions, "required_signal:"),
        "required_signal_count": len(group(assertions, "required_signal:")),
        "forbidden_signal_count": sum(item.status == "FAIL" for item in group(assertions, "forbidden_signal:")),
    }


def retrieval_metrics(assertions):
    return {
        "expected_source_retrieved_rate": pass_rate(assertions, "rag:expected_sources"),
        "citation_validity": pass_rate(assertions, "rag:citation_validity"),
    }


def forbidden_action_count(assertions):
    return sum(len(item.actual or []) for item in group(assertions, "safety:forbidden_actions"))


def agent_metrics(assertions, agents, completion_rate):
    """Report agent rates, tool usage and blocked duplicate calls."""
    duplicate_blocked = sum(
        sum(step.get("observation_status") == "DUPLICATE_BLOCKED" for step in agent.get("trace", []))
        for agent in agents
    )
    return {
        "agent_completion_rate": completion_rate,
        "agent_valid_final_schema_rate": pass_rate(assertions, "agent:final_schema"),
        "agent_valid_evidence_citations_rate": pass_rate(assertions, "agent:evidence_citations"),
        "agent_human_review_correctness": pass_rate(assertions, "agent:human_review_required"),
        "agent_valid_action_rate": pass_rate(assertions, "agent:acceptable_action"),
        "agent_tool_call_count": sum(agent.get("tool_call_count", 0) for agent in agents),
        "agent_duplicate_tool_calls_blocked": duplicate_blocked,
    }
