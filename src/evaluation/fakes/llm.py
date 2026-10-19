"""Deterministic structured generation driven only by collected evidence."""

import json

PLAN_ACTIONS = ["GET_CLAIM", "GET_TRIAGE", "GET_ANOMALY", "GET_HISTORY", "GET_INVESTIGATION"]
EVIDENCE_MARKER = "UNTRUSTED TASK AND TOOL EVIDENCE (JSON):\n"
ANSWER_MARKER = "INPUT DATA:\n"


class FakeStructuredLlm:
    """Deterministic structured generation based on actual collected evidence.

    This fixture never receives the golden expectations. Both the planner and
    finalizer run their real prompts and response schemas against this boundary.
    """

    def __init__(self, *, vision=False, rag_supported=True):
        self.vision, self.rag_supported, self.calls = vision, rag_supported, []

    def generate(self, *, prompt, response_model):
        self.calls.append(response_model.__name__)
        if response_model.__name__ == "GroundedLlmOutput":
            return self._grounded_answer(prompt, response_model)
        data = json.loads(prompt.split(EVIDENCE_MARKER, 1)[1])
        if response_model.__name__ == "AgentPlanStep":
            return self._plan_step(data, response_model)
        return self._final_result(data, response_model)

    def _grounded_answer(self, prompt, response_model):
        data = json.loads(prompt.split(ANSWER_MARKER, 1)[1])
        chunks = data["retrieved_sources"]
        selected = next((item for item in chunks if "deductible" in item["section"].casefold()), chunks[0])
        return response_model.model_validate(
            {
                "answer": selected["text"]
                if self.rag_supported
                else "The retrieved demo excerpts do not define the requested topic.",
                "cited_chunk_ids": [selected["chunk_id"]] if self.rag_supported else [],
                "insufficient_evidence": not self.rag_supported,
            }
        )

    def _plan_step(self, data, response_model):
        actions = list(PLAN_ACTIONS)
        if self.vision:
            actions.append("GET_VISUAL_ASSESSMENT")
        actions += ["FIND_SIMILAR_CLAIMS", "SEARCH_POLICY"]
        completed = {call["action"] for call in data["previous_calls"]}
        action = next((action for action in actions if action not in completed), "FINISH")
        arguments = (
            {"query": "demo collision deductible supporting evidence", "limit": 3}
            if action == "SEARCH_POLICY"
            else ({"limit": 3} if action == "FIND_SIMILAR_CLAIMS" else {})
        )
        return response_model.model_validate(
            {
                "action": action,
                "reason": "Collect the relevant stored evidence for human review.",
                "arguments": arguments,
            }
        )

    @staticmethod
    def _structured(by_type, source_type):
        return by_type.get(source_type, {}).get("structured_data", {})

    @staticmethod
    def _recommended_action(by_type, unavailable, missing):
        investigation = FakeStructuredLlm._structured(by_type, "INVESTIGATION_ASSESSMENT")
        if any(item["status"] == "NOT_AVAILABLE" for item in unavailable):
            return "NO_RECOMMENDATION"
        if missing:
            return "REQUEST_ADDITIONAL_INFORMATION"
        if investigation.get("review_recommended"):
            return "REFER_FOR_INVESTIGATION_REVIEW"
        return "REQUEST_ADDITIONAL_INFORMATION" if investigation.get("signals") else "CONTINUE_STANDARD_REVIEW"

    @staticmethod
    def _rationale(evidence):
        return [
            {
                "statement": f"Review the collected {item['source_type'].lower().replace('_', ' ')} evidence.",
                "evidence_ids": [item["evidence_id"]],
            }
            for item in evidence[:12]
        ]

    @staticmethod
    def _uncertainties(unavailable, missing, by_type):
        uncertainties = [f"{item['tool']}: {item.get('message') or 'Evidence unavailable'}" for item in unavailable]
        if missing:
            uncertainties.append("Some document entries are unavailable and need human follow-up.")
        visual = FakeStructuredLlm._structured(by_type, "VISUAL_ASSESSMENT")
        if visual.get("consistency", {}).get("declared_damage_not_visually_confirmed"):
            uncertainties.append(
                "Declared damage was not visually confirmed; non-detection is not evidence of absence."
            )
        return uncertainties

    def _final_result(self, data, response_model):
        evidence = data["evidence"]
        by_type = {item["source_type"]: item for item in evidence}
        unavailable = [item for item in data["observations"] if item["status"] in ("NOT_AVAILABLE", "NOT_FOUND")]
        claim = self._structured(by_type, "CLAIM_DATA")
        missing = any(not item["available"] for item in claim.get("documents", []))
        return response_model.model_validate(
            {
                "case_summary": "The collected evidence is available for a human claims reviewer.",
                "recommended_next_action": self._recommended_action(by_type, unavailable, missing),
                "rationale": self._rationale(evidence),
                "evidence_ids": [item["evidence_id"] for item in evidence[:12]],
                "uncertainties": self._uncertainties(unavailable, missing, by_type),
                "human_review_required": True,
            }
        )
