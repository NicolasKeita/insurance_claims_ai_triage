"""Application-owned grounding messages, safe for audit and correction feedback."""

MESSAGES = {
    "INVALID_EVIDENCE": "The recommendation violates the collected evidence contract.",
    "INVALID_REGISTRY": "Evidence registry keys do not match their source IDs.",
    "UNCITED_RATIONALE": "A rationale must cite at least one collected evidence ID.",
    "UNKNOWN_EVIDENCE_ID": "A cited ID is not in the collected evidence; copy exact allowed_evidence_ids.",
    "NO_EVIDENCE": "Without collected evidence, the recommendation must abstain.",
    "NO_SUPPORTING_RATIONALE": "A supported action needs at least one factual, cited rationale.",
    "INCOMPATIBLE_FOLLOW_UP": "Required assessment follow-up must use one of the allowed actions.",
    "FOLLOW_UP_ASSESSMENT_NOT_CITED": "A follow-up rationale must cite an ID from required_follow_up_evidence_ids.",
    "REPAIR_ASSESSMENT_NOT_CITED": "A follow-up rationale must cite an ID from required_follow_up_evidence_ids.",
    "INVALID_KNOWLEDGE_METADATA": "A knowledge source is missing authoritative citation metadata.",
    "KNOWLEDGE_REFERENCE_MISMATCH": "A knowledge reference does not match its collected chunk ID.",
    "VERSION_MISMATCH": "Use the exact application and prompt versions required by the response schema.",
}


def safe_code(code):
    return code if isinstance(code, str) and code in MESSAGES else "INVALID_EVIDENCE"


def grounding_feedback(code):
    code = safe_code(code)
    return {"code": code, "message": MESSAGES[code]}


def grounding_error_text(code):
    code = safe_code(code)
    return f"Agent grounding failed. [{code}] {MESSAGES[code]}"
