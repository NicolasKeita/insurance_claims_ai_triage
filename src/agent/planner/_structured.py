"""Structured."""

import json

from agent.models import AgentPlanStep, AgentRecommendation, AgentToolName
from agent.planner._constants import (
    AGENT_SYSTEM_PROMPT,
    FINALIZER_RULES,
    PLANNER_EXAMPLES,
    PLANNER_RULES,
)
from agent.planner._context import finalizer_evidence
from agent.planner._responses import recommendation_constraints
from claims.llm import StructuredLlm


class StructuredAgentPlanner:
    def __init__(self, llm: StructuredLlm, *, require_investigation: bool = False):
        self.llm = llm
        self.require_investigation = require_investigation
        self.minimum_tool_steps = 2 if require_investigation else 1

    def _required_step(self, state):
        """The application collects core stored evidence before discretionary LLM planning."""
        if self.require_investigation:
            attempted = {item["tool_name"] for item in state["observations"]}
            for action in ("GET_CLAIM", "GET_INVESTIGATION"):
                if action not in attempted:
                    return AgentPlanStep(
                        action=action,
                        arguments={},
                        reason="Required initial read of stored claim facts and investigation signals.",
                    )
        return None

    def plan(self, state) -> AgentPlanStep:
        required = self._required_step(state)
        if required is not None:
            return required
        observations = [
            {
                "tool": o["tool_name"],
                "status": o["status"],
                "message": o.get("message"),
                "evidence": [
                    {"evidence_id": e["evidence_id"], "summary": e["summary"][:600]} for e in o.get("evidence", [])
                ],
            }
            for o in state["observations"]
        ]
        data = {
            "claim_id": state["claim_id"],
            "objective": state["objective"],
            "remaining_tool_steps": state["max_iterations"] - state["iteration"],
            "allowed_actions": [t.value for t in AgentToolName] + ["FINISH"],
            "previous_calls": [{"action": p["action"], "arguments": p["arguments"]} for p in state["plan_history"]],
            "observations": observations,
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


class StructuredAgentFinalizer:
    def __init__(self, llm: StructuredLlm):
        self.llm = llm

    def finalize(self, state) -> AgentRecommendation:
        evidence, context_truncated = finalizer_evidence(state["evidence_registry"])
        response_model, constraints = recommendation_constraints(state["evidence_registry"])
        data = {
            **constraints,
            "claim_id": state["claim_id"],
            "objective": state["objective"],
            "collection_stop_reason": state["stop_reason"],
            "allowed_evidence_ids": list(state["evidence_registry"]),
            "evidence": evidence,
            "evidence_context_truncated": context_truncated,
            "observations": [
                {"tool": o["tool_name"], "status": o["status"], "message": o.get("message")}
                for o in state["observations"]
            ],
        }
        prompt = (
            AGENT_SYSTEM_PROMPT
            + "\n\nSYSTEM RULES — FINALIZER:\n"
            + FINALIZER_RULES
            + '\nRequired rationale shape: [{"statement":"A fact supported by evidence",'
            + '"evidence_ids":["copy an exact ID from allowed_evidence_ids"]}].\n'
            + "\n\nUNTRUSTED TASK AND TOOL EVIDENCE (JSON):\n"
            + json.dumps(data, ensure_ascii=False)
        )
        output = self.llm.generate(prompt=prompt, response_model=response_model)
        checked = response_model.model_validate(output.model_dump())
        recommendation = AgentRecommendation.model_validate(checked.model_dump())
        if context_truncated:
            recommendation.uncertainties = [
                "Evidence details were shortened to fit the model context.",
                *recommendation.uncertainties,
            ][:20]
        return recommendation
