"""Wiring of the inspectable PLAN / TOOL / FINALIZE / VALIDATE graph."""

from langgraph.graph import END, START, StateGraph

from agent.graph.finalizing import finalize_node, route_finalize
from agent.graph.planning import plan_node, route_plan
from agent.graph.tool_node import route_tool, tool_node
from agent.graph.validation import validate_node
from agent.models import AgentState


def build_agent_graph(tools, planner, finalizer):
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node(planner))
    graph.add_node("tool", tool_node(tools))
    graph.add_node("finalize", finalize_node(finalizer))
    graph.add_node("validate", validate_node)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_plan, {"tool": "tool", "finalize": "finalize", "end": END})
    graph.add_conditional_edges("tool", route_tool, {"plan": "plan", "end": END})
    graph.add_conditional_edges("finalize", route_finalize, {"validate": "validate", "end": END})
    graph.add_edge("validate", END)
    return graph.compile()
