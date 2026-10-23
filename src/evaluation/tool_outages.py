"""Declared fault injection policy shared by real preflight and execution."""

ASSESSMENT_OUTAGE_TOOLS = frozenset({"GET_TRIAGE", "GET_ANOMALY", "GET_INVESTIGATION", "GET_VISUAL_ASSESSMENT"})
CONTROLLED_OUTAGE_TOOLS = ASSESSMENT_OUTAGE_TOOLS | {"GET_HISTORY"}


def tool_outage_reason(config, scenario):
    """Reject unsupported faults before inference; never replace real service data."""
    unavailable = scenario.inputs.get("unavailable_tools", [])
    if not unavailable:
        return None
    if not isinstance(unavailable, list) or any(not isinstance(tool, str) for tool in unavailable):
        return "Controlled agent tool outages must be a list of tool names"
    unsupported = set(unavailable) - CONTROLLED_OUTAGE_TOOLS
    if unsupported:
        return "Unsupported controlled agent tool outages: " + ", ".join(sorted(unsupported))
    if not config.include_agent:
        return "Controlled agent tool outages require --include-agent"
    if not config.isolated_demo_stack:
        return "Controlled agent tool outages require --isolated-demo-stack"
    return None


def history_outage_session():
    """Fail only the selected agent's history connection through its real error handler.

    The real evaluation passes this callable to one service instance after checking
    isolation. No database state, repository result or tool observation is fabricated.
    """
    from sqlalchemy.exc import OperationalError

    raise OperationalError(None, None, RuntimeError("Controlled evaluation history connection failure"))
