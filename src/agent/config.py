"""Versioned and bounded agent configuration; never supplied by the planner."""

CLAIMS_AGENT_VERSION = "claims_agent_v1"
CLAIMS_AGENT_PROMPT_VERSION = "claims_agent_prompt_v1"
DEFAULT_MAX_ITERATIONS = 8
MAX_ITERATIONS = 8
MAX_RETRIEVAL_LIMIT = 5
MAX_OBJECTIVE_CHARS = 2000
AGENT_LLM_CONTEXT_WINDOW = 8192
MAX_FINALIZER_EVIDENCE_CHARS = 18000
DEFAULT_OBJECTIVE = (
    "Review the available evidence for this insurance claim and recommend the next "
    "human review step. Identify relevant uncertainties and supporting evidence."
)
