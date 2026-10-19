"""Structured recommendation finalization through the existing LLM protocol."""

import json

from agent.models import AgentRecommendation
from agent.planner.context import finalizer_evidence
from agent.planner.prompts import AGENT_SYSTEM_PROMPT, FINALIZER_RULES
from claims.llm import StructuredLlm


class StructuredAgentFinalizer:
    def __init__(self, llm: StructuredLlm):
        self.llm = llm

    def finalize(self, state) -> AgentRecommendation:
        evidence, context_truncated = finalizer_evidence(state["evidence_registry"])
        data = {
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
        output = self.llm.generate(prompt=prompt, response_model=AgentRecommendation)
        recommendation = AgentRecommendation.model_validate(output.model_dump())
        if context_truncated:
            recommendation.uncertainties = [
                "Evidence details were shortened to fit the model context.",
                *recommendation.uncertainties,
            ][:20]
        return recommendation
