from functools import partial

from langgraph.graph import END, START, StateGraph

from agent.models import AgentState

from ._finalization import finalize, route_finalize, validate
from ._planning import plan, route_plan
from ._tools import route_tool, tool


def build_agent_graph(tools, planner, finalizer):
    graph = StateGraph(AgentState)
    graph.add_node("plan", partial(plan, planner=planner))
    graph.add_node("tool", partial(tool, tools=tools))
    graph.add_node("finalize", partial(finalize, finalizer=finalizer))
    graph.add_node("validate", validate)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_plan, {"tool": "tool", "finalize": "finalize", "end": END})
    graph.add_conditional_edges("tool", route_tool, {"plan": "plan", "end": END})
    graph.add_conditional_edges("finalize", route_finalize, {"validate": "validate", "end": END})
    graph.add_edge("validate", END)
    return graph.compile()
