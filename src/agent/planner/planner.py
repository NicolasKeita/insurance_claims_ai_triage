"""Structured next-action planning through the existing LLM protocol."""

import json

from agent.models import AgentPlanStep, AgentToolName
from agent.planner.prompts import AGENT_SYSTEM_PROMPT, PLANNER_EXAMPLES, PLANNER_RULES
from claims.llm import StructuredLlm


class StructuredAgentPlanner:
    def __init__(self, llm: StructuredLlm):
        self.llm = llm

    def plan(self, state) -> AgentPlanStep:
        data = {
            "claim_id": state["claim_id"],
            "objective": state["objective"],
            "remaining_tool_steps": state["max_iterations"] - state["iteration"],
            "allowed_actions": [t.value for t in AgentToolName] + ["FINISH"],
            "previous_calls": [{"action": p["action"], "arguments": p["arguments"]} for p in state["plan_history"]],
            "observations": observability(state["observations"]),
        }
        prompt = (
            AGENT_SYSTEM_PROMPT
            + "\n\nSYSTEM RULES — PLANNER:\n"
            + PLANNER_RULES
            + "\nREQUIRED JSON SHAPE EXAMPLES (select one appropriate action):\n"
            + json.dumps(PLANNER_EXAMPLES)
            + "\n\nUNTRUSTED TASK AND TOOL EVIDENCE (JSON):\n"
            + json.dumps(data, ensure_ascii=False)
        )
        return AgentPlanStep.model_validate(self.llm.generate(prompt=prompt, response_model=AgentPlanStep).model_dump())


def observability(observations):
    """Keep tool status and bounded evidence summaries for the planner prompt."""
    return [
        {
            "tool": o["tool_name"],
            "status": o["status"],
            "message": o.get("message"),
            "evidence": [
                {"evidence_id": e["evidence_id"], "summary": e["summary"][:600]} for e in o.get("evidence", [])
            ],
        }
        for o in observations
    ]
